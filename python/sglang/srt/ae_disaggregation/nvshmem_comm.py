"""NVSHMEM communication handlers ported directly from Janus.

Protocol:
  A2E: putmem_signal with iteration/layer/token metadata packed in signal.
  E2A: putmem_signal of a partial routed-MoE output.
  A: waits for and sums all expected expert outputs.

The only intentional divergence from Janus is the context import: modern
SGLang has no ``schedule_batch.global_server_args_dict``, so the same mutable
dictionary lives in ``ae_disaggregation.context``.
"""

import ctypes
import os

import nvshmem.bindings.nvshmem as bindings
import nvshmem.core
import torch
import torch.distributed as dist

from sglang.srt.ae_disaggregation.context import global_server_args_dict
from sglang.srt.ae_disaggregation.nvshmem_utils import (
    TorchStreamWrapper,
    torch_rank_to_pe_rank,
)

DEBUG_NVSHMEM_VERIFY = os.environ.get("DEBUG_NVSHMEM_VERIFY", "0") == "1"

ITER_SHIFT = 40
LAYER_SHIFT = 20
FIELD_MASK = (1 << 20) - 1


def _pack_signal(iteration: int, layer_index: int, tokens: int) -> int:
    return (iteration << ITER_SHIFT) | (layer_index << LAYER_SHIFT) | tokens


def _signal_threshold(iteration: int) -> int:
    return iteration << ITER_SHIFT


_cudart = None
_pinned_signal = None


def _get_cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    return _cudart


def _get_pinned_signal():
    global _pinned_signal
    if _pinned_signal is None:
        _pinned_signal = torch.empty(1, dtype=torch.int64, pin_memory=True)
    return _pinned_signal


def _read_signal_value(sig_buffer) -> tuple:
    """Janus slow path: read first-layer metadata from the GPU signal."""
    pinned = _get_pinned_signal()
    cudart = _get_cudart()
    stream = torch.cuda.current_stream().cuda_stream
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(pinned.data_ptr()),
        ctypes.c_void_p(sig_buffer.handle),
        ctypes.c_size_t(8),
        ctypes.c_int(2),
        ctypes.c_void_p(stream),
    )
    torch.cuda.current_stream().synchronize()
    value = pinned.item()
    return (value >> LAYER_SHIFT) & FIELD_MASK, value & FIELD_MASK


def _current_stream_handle():
    return torch.cuda.current_stream().cuda_stream


def _current_sw():
    return TorchStreamWrapper(torch.cuda.current_stream())


class AttnNvshmemCommunicationHandler:
    """Attention side: sends hidden states and receives summed E outputs."""

    def __init__(self):
        self.rank = dist.get_rank()
        self.att_tp_size = global_server_args_dict["tp_size"]
        self.ep_size = global_server_args_dict["ep_size"]
        self.moe_workers = list(
            range(self.att_tp_size, self.att_tp_size + self.ep_size)
        )
        self.micro_batch_num = global_server_args_dict.get("micro_batch_num", 1)

        self.map_att_to_moe = [[] for _ in range(self.att_tp_size)]
        for ep_rank in self.moe_workers:
            self.map_att_to_moe[ep_rank % self.att_tp_size].append(ep_rank)
        self.send_targets = [
            torch_rank_to_pe_rank(rank)
            for rank in self.map_att_to_moe[self.rank]
        ]
        print(f"[AttnNvshmem] rank={self.rank} send_targets(PE)={self.send_targets}")

        self.ep_group_info = global_server_args_dict["ep_group_info"]
        self.expected_ep_ranks = self.ep_group_info["send_strategy"].get(
            self.rank, []
        )
        print(
            f"[AttnNvshmem] rank={self.rank} "
            f"expected_ep_ranks={self.expected_ep_ranks}"
        )

        self.a2e_bufs = global_server_args_dict["nvshmem_a2e_bufs"]
        self.a2e_sigs = global_server_args_dict["nvshmem_a2e_sigs"]
        self.e2a_dst_slots = global_server_args_dict["nvshmem_e2a_dst_slots"]
        self.e2a_sigs = global_server_args_dict["nvshmem_e2a_sigs"]
        self._iteration = {i: 0 for i in range(self.micro_batch_num)}
        self.shape = {}

    async def send_attention_result(
        self, batch_id: int, layer_index: int, gpu_hidden_state: torch.Tensor
    ):
        self.shape[batch_id] = gpu_hidden_state.shape
        tokens = gpu_hidden_state.shape[0]
        if DEBUG_NVSHMEM_VERIFY:
            gpu_hidden_state = torch.full_like(gpu_hidden_state, layer_index)

        self._iteration[batch_id] += 1
        flag_val = _pack_signal(self._iteration[batch_id], layer_index, tokens)
        flat = gpu_hidden_state.reshape(-1)
        data_bytes = flat.numel() * flat.element_size()

        src_buf = self.a2e_bufs[batch_id]["src"]
        src_buf[: flat.numel()].copy_(flat)
        dst_buf = self.a2e_bufs[batch_id]["dst"]
        sig = self.a2e_sigs[batch_id]["sig"]
        stream = _current_stream_handle()
        for dst_rank in self.send_targets:
            bindings.putmem_signal_on_stream(
                dst_buf.data_ptr(),
                src_buf.data_ptr(),
                data_bytes,
                sig.handle,
                flag_val,
                bindings.Signal_op.SIGNAL_SET,
                dst_rank,
                stream,
            )

    async def send_skip_signal(self, batch_id: int):
        self._iteration[batch_id] += 1
        flag_val = _pack_signal(self._iteration[batch_id], 0, 0)
        dst_buf = self.a2e_bufs[batch_id]["dst"]
        sig = self.a2e_sigs[batch_id]["sig"]
        stream = _current_stream_handle()
        for dst_rank in self.send_targets:
            bindings.putmem_signal_on_stream(
                dst_buf.data_ptr(),
                dst_buf.data_ptr(),
                0,
                sig.handle,
                flag_val,
                bindings.Signal_op.SIGNAL_SET,
                dst_rank,
                stream,
            )

    async def recv_moe_result(self, layer_index: int, batch_id: int):
        shape = self.shape[batch_id]
        numel = 1
        for size in shape:
            numel *= size

        self._iteration[batch_id] += 1
        flag_val = self._iteration[batch_id]
        sw = _current_sw()
        n_peers = len(self.expected_ep_ranks)
        if n_peers == 1:
            nvshmem.core.signal_wait(
                self.e2a_sigs[batch_id][0],
                flag_val,
                bindings.Cmp_type.CMP_GE,
                stream=sw,
            )
            return self.e2a_dst_slots[batch_id][0][:numel].reshape(shape)

        accumulator = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
        for index in range(n_peers):
            nvshmem.core.signal_wait(
                self.e2a_sigs[batch_id][index],
                flag_val,
                bindings.Cmp_type.CMP_GE,
                stream=sw,
            )
            accumulator.add_(self.e2a_dst_slots[batch_id][index][:numel].reshape(shape))
        return accumulator


class MoENvshmemCommunicationHandler:
    """MoE side: receives hidden states and sends one local partial output."""

    def __init__(self, hidden_dim=5120):
        self.hidden_dim = hidden_dim
        self.rank = dist.get_rank()
        self.att_tp_size = global_server_args_dict["tp_size"]
        self.ep_size = global_server_args_dict["ep_size"]
        self.ep_rank = self.rank - self.att_tp_size
        self.micro_batch_num = global_server_args_dict.get("micro_batch_num", 1)
        self.total_layers = int(os.environ.get("NUM_HIDDEN_LAYERS", 60))
        self.last_layer = 0
        self._shutdown_requested = False
        self._shutdown_reason = ""
        self.recv_att_peer = self.rank % self.att_tp_size

        self.ep_group_info = global_server_args_dict["ep_group_info"]
        self.send_targets = []
        self.slot_index_at = {}
        for att_rank, ep_ranks in self.ep_group_info["send_strategy"].items():
            if self.rank in ep_ranks:
                att_pe = torch_rank_to_pe_rank(att_rank)
                self.send_targets.append(att_pe)
                self.slot_index_at[att_pe] = ep_ranks.index(self.rank)

        self.a2e_bufs = global_server_args_dict["nvshmem_a2e_bufs"]
        self.a2e_sigs = global_server_args_dict["nvshmem_a2e_sigs"]
        self.e2a_dst_slots = global_server_args_dict["nvshmem_e2a_dst_slots"]
        self.e2a_src = global_server_args_dict["nvshmem_e2a_src"]
        self.e2a_sigs = global_server_args_dict["nvshmem_e2a_sigs"]
        self._iteration = {i: 0 for i in range(self.micro_batch_num)}
        self._cached_tokens = {}
        self._next_layer_idx = {}

    @property
    def should_shutdown(self):
        return self._shutdown_requested

    def request_shutdown(self, reason=""):
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_reason = reason
            print(f"[MoENvshmem] rank={self.rank} shutdown requested: {reason}")

    def recv_attention_result(self, batch_id):
        self._iteration[batch_id] += 1
        threshold = _signal_threshold(self._iteration[batch_id])
        try:
            nvshmem.core.signal_wait(
                self.a2e_sigs[batch_id]["sig"],
                threshold,
                bindings.Cmp_type.CMP_GE,
                stream=_current_sw(),
            )
        except Exception as exc:
            self.request_shutdown(f"signal_wait failed: {exc}")
            return None, None

        if batch_id in self._cached_tokens:
            tokens = self._cached_tokens[batch_id]
            layer_index = self._next_layer_idx[batch_id]
            self._next_layer_idx[batch_id] = layer_index + 1
            if layer_index >= self.total_layers - 1:
                del self._cached_tokens[batch_id]
                del self._next_layer_idx[batch_id]
        else:
            layer_index, tokens = _read_signal_value(self.a2e_sigs[batch_id]["sig"])
            if tokens > 0:
                self._cached_tokens[batch_id] = tokens
                self._next_layer_idx[batch_id] = layer_index + 1

        self.last_layer = layer_index
        if tokens == 0:
            return layer_index, torch.empty(
                0, self.hidden_dim, dtype=torch.bfloat16, device="cuda"
            )
        data_numel = tokens * self.hidden_dim
        return layer_index, self.a2e_bufs[batch_id]["dst"][:data_numel].reshape(
            tokens, self.hidden_dim
        )

    def send_moe_result(self, layer_index, batch_id, result_state):
        try:
            self._iteration[batch_id] += 1
            flag_val = self._iteration[batch_id]
            if DEBUG_NVSHMEM_VERIFY:
                result_state = torch.full_like(result_state, layer_index)

            flat = result_state.reshape(-1)
            e2a_src = self.e2a_src[batch_id]
            e2a_src[: flat.numel()].copy_(flat)
            stream = _current_stream_handle()
            for dst_rank in self.send_targets:
                slot_idx = self.slot_index_at[dst_rank]
                actual_dst = self.e2a_dst_slots[batch_id][slot_idx]
                bindings.putmem_signal_on_stream(
                    actual_dst[: flat.numel()].data_ptr(),
                    e2a_src.data_ptr(),
                    flat.numel() * 2,
                    self.e2a_sigs[batch_id][slot_idx].handle,
                    flag_val,
                    bindings.Signal_op.SIGNAL_SET,
                    dst_rank,
                    stream,
                )
        except Exception as exc:
            self.request_shutdown(f"send failed: {exc}")
