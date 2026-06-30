"""Stateless expert worker, ported from Janus' ``runner/moe_runner.py``.

This runner deliberately owns no scheduler, KV cache, or request state.  It
joins the A+E torch world, allocates the same symmetric NVSHMEM buffers as the
attention runner, then services ``(layer_id, hidden_states)`` messages forever.
"""

import os

import nvshmem.core
import torch
import torch.distributed as dist

from sglang.srt.ae_disaggregation.context import global_server_args_dict
from sglang.srt.ae_disaggregation.nvshmem_comm import (
    MoENvshmemCommunicationHandler,
)
from sglang.srt.ae_disaggregation.nvshmem_utils import (
    init_nvshmem_by_torch_process_group,
    nvshmem_create_tensor,
)
from sglang.srt.ae_disaggregation.utils import get_ep_group_info
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_ae_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.model_loader import get_model_loader
from sglang.srt.model_loader.loader import DefaultModelLoader, _get_quantization_config
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.server_args import set_global_server_args_for_scheduler


class MoeRunner:
    """Janus-style single-process MoE worker for one physical expert rank."""

    def __init__(self, server_args, port_args, gpu_id: int, ep_rank: int, pipe_writer=None):
        del port_args  # Retained to keep the Janus runner construction shape.
        self.server_args = server_args
        self.model_config = ModelConfig.from_server_args(server_args)
        # Torch ranks are [attention ranks][expert ranks], exactly as Janus.
        self.rank = server_args.tp_size + ep_rank
        self.local_rank = gpu_id
        self.micro_batch_num = server_args.micro_batch_num
        self.model = None

        torch.cuda.set_device(gpu_id)
        self.ep_group_info = get_ep_group_info(
            tp_size=server_args.tp_size,
            ep_size=server_args.ep_size,
            moe_node_num=1,
            enable_ep_intra_node_reduce=False,
        )
        global_server_args_dict["ep_group_info"] = self.ep_group_info

        self.init_torch_distributed()
        self.init_nvshmem_distributed()
        self.load_moe()
        self.communicator = MoENvshmemCommunicationHandler(
            self.model_config.hf_config.hidden_size
        )
        if pipe_writer is not None:
            pipe_writer.send({"ae_role": "expert", "ep_rank": ep_rank})
        self.forward_loop()

    def init_torch_distributed(self):
        """The direct Janus global A+E communicator initialization."""
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        init_distributed_environment(
            backend="nccl",
            world_size=self.server_args.tp_size + self.server_args.ep_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method="tcp://" + self.server_args.dist_init_addr,
        )
        set_global_server_args_for_scheduler(self.server_args)
        initialize_ae_model_parallel(
            attention_rank_count=self.server_args.tp_size,
            expert_rank_count=self.server_args.ep_size,
        )

    def init_nvshmem_distributed(self):
        """Allocate buffers in the identical order used by AttentionRunner."""
        init_nvshmem_by_torch_process_group(dist.group.WORLD)

        hidden_dim = self.model_config.hf_config.hidden_size
        max_tokens = int(os.environ.get("NVSHMEM_MAX_TOKENS", 4096))
        buf_numel = max_tokens * hidden_dim
        max_ep_peers = max(
            len(peers) for peers in self.ep_group_info["send_strategy"].values()
        )

        a2e_bufs = []
        a2e_sigs = []
        for _ in range(self.micro_batch_num):
            a2e_bufs.append(
                {
                    "src": nvshmem_create_tensor((buf_numel,), torch.bfloat16),
                    "dst": nvshmem_create_tensor((buf_numel,), torch.bfloat16),
                }
            )
            a2e_sigs.append({"sig": nvshmem.core.buffer(8)})

        e2a_dst_slots = []
        e2a_src = []
        e2a_sigs = []
        for _ in range(self.micro_batch_num):
            e2a_dst_slots.append(
                [
                    nvshmem_create_tensor((buf_numel,), torch.bfloat16)
                    for _ in range(max_ep_peers)
                ]
            )
            e2a_src.append(nvshmem_create_tensor((buf_numel,), torch.bfloat16))
            e2a_sigs.append([nvshmem.core.buffer(8) for _ in range(max_ep_peers)])

        global_server_args_dict.update(
            {
                "nvshmem_a2e_bufs": a2e_bufs,
                "nvshmem_a2e_sigs": a2e_sigs,
                "nvshmem_e2a_dst_slots": e2a_dst_slots,
                "nvshmem_e2a_src": e2a_src,
                "nvshmem_e2a_sigs": e2a_sigs,
                "tp_size": self.server_args.tp_size,
                "ep_size": self.server_args.ep_size,
                "micro_batch_num": self.micro_batch_num,
            }
        )

    def load_moe(self):
        """Load this rank's GLM gate/TopK and physical routed experts."""
        if os.environ.get("DEBUG_NVSHMEM_ECHO", "0") == "1":
            return
        from sglang.srt.ae_disaggregation.glm_moe import GlmMoeLocalExperts

        load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            raise ValueError(
                "AE local-MoE loading currently supports the standard "
                f"SGLang loader, got {type(loader).__name__}."
            )

        first_sparse_layer = self.model_config.hf_config.first_k_dense_replace
        layer_ids = range(
            first_sparse_layer, self.model_config.hf_config.num_hidden_layers
        )
        quant_config = _get_quantization_config(self.model_config, load_config)
        with set_default_torch_dtype(self.model_config.dtype):
            self.model = GlmMoeLocalExperts(
                config=self.model_config.hf_config,
                layer_ids=layer_ids,
                quant_config=quant_config,
            ).cuda(self.local_rank)
            self.model.load_weights(loader._get_all_weights(self.model_config, self.model))
            loader.load_weights_and_postprocess(
                self.model, (), torch.device(self.server_args.device)
            )
        self.model.eval()

    def forward_loop(self):
        """Same sequential micro-batch service loop as Janus."""
        echo_mode = os.environ.get("DEBUG_NVSHMEM_ECHO", "0") == "1"
        while not self.communicator.should_shutdown:
            for batch_id in range(self.micro_batch_num):
                layer_index, hidden_state = self.communicator.recv_attention_result(
                    batch_id
                )
                if hidden_state is None:
                    self.communicator.request_shutdown("recv failed")
                    break
                if hidden_state.shape[0] == 0:
                    continue

                if echo_mode:
                    output = hidden_state.clone()
                else:
                    output = self.model.forward_with_gate(layer_index, hidden_state)
                self.communicator.send_moe_result(layer_index, batch_id, output)


def run_moe_runner_process(server_args, port_args, gpu_id: int, ep_rank: int, pipe_writer):
    """Multiprocessing target used by the Janus-style expert launch branch."""
    MoeRunner(server_args, port_args, gpu_id, ep_rank, pipe_writer)
