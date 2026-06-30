"""Attention-side runner following Janus' ``runner/attention_runner.py``."""

import os

import nvshmem.core
import torch
import torch.distributed as dist

from sglang.srt.ae_disaggregation.context import global_server_args_dict
from sglang.srt.ae_disaggregation.nvshmem_utils import (
    init_nvshmem_by_torch_process_group,
    nvshmem_create_tensor,
)
from sglang.srt.ae_disaggregation.utils import get_ep_group_info
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_ae_model_parallel,
    set_custom_all_reduce,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.utils import get_available_gpu_memory


class AttentionRunner(ModelRunner):
    """Normal SGLang serving runner with Janus A2E/E2A initialization."""

    def init_torch_distributed(self):
        torch.cuda.set_device(self.gpu_id)
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        self.ae_ep_size = self.server_args.ep_size
        init_distributed_environment(
            backend="nccl",
            world_size=self.server_args.tp_size + self.ae_ep_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method="tcp://" + self.server_args.dist_init_addr,
        )
        # Required before GLM decides whether shared experts are fused.
        self.server_args.disable_shared_experts_fusion = True
        set_global_server_args_for_scheduler(self.server_args)
        initialize_ae_model_parallel(
            attention_rank_count=self.server_args.tp_size,
            expert_rank_count=self.ae_ep_size,
        )
        initialize_dp_attention(
            server_args=self.server_args, model_config=self.model_config
        )
        self.init_nvshmem_distributed()
        self._ae_nvshmem_initialized = True
        return get_available_gpu_memory(
            self.device, self.gpu_id, distributed=False
        )

    def initialize(self, pre_model_load_memory: float):
        # The current SGLang ModelRunner initializes EPLB/expert-location metadata
        # before GLM layers are swapped to the AE A-side MoE block.  GLM-5.1 has
        # 256 routed experts, which is not divisible by the AE ep-size 7.  A does
        # not own routed experts, so keep ep_size=7 only for distributed/NVSHMEM
        # setup and make local model initialization see ep_size=1.
        ae_ep_size = getattr(self, "ae_ep_size", self.server_args.ep_size)
        self.server_args.ep_size = 1
        try:
            super().initialize(pre_model_load_memory)
        finally:
            self.server_args.ep_size = ae_ep_size
        if not getattr(self, "_ae_nvshmem_initialized", False):
            self.init_nvshmem_distributed()
            self._ae_nvshmem_initialized = True

    def _profile_available_bytes(self, pre_model_load_memory: int) -> int:
        """Profile A-local KV memory without involving stateless E ranks."""
        post_model_load_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=False
        )
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        return int(rest_memory * (1 << 30))

    def init_nvshmem_distributed(self):
        """Exact allocation order shared with Janus MoeRunner."""
        ep_group_info = get_ep_group_info(
            tp_size=self.server_args.tp_size,
            ep_size=self.ae_ep_size,
            moe_node_num=1,
            enable_ep_intra_node_reduce=False,
        )
        global_server_args_dict["ep_group_info"] = ep_group_info
        init_nvshmem_by_torch_process_group(dist.group.WORLD)

        hidden_dim = self.model_config.hf_config.hidden_size
        max_tokens = int(os.environ.get("NVSHMEM_MAX_TOKENS", 4096))
        buf_numel = max_tokens * hidden_dim
        micro_batch_num = getattr(self.server_args, "micro_batch_num", 1)
        max_ep_peers = max(
            len(peers) for peers in ep_group_info["send_strategy"].values()
        )

        a2e_bufs, a2e_sigs = [], []
        for _ in range(micro_batch_num):
            a2e_bufs.append(
                {
                    "src": nvshmem_create_tensor((buf_numel,), torch.bfloat16),
                    "dst": nvshmem_create_tensor((buf_numel,), torch.bfloat16),
                }
            )
            a2e_sigs.append({"sig": nvshmem.core.buffer(8)})

        e2a_dst_slots, e2a_src, e2a_sigs = [], [], []
        for _ in range(micro_batch_num):
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
                "ep_size": self.ae_ep_size,
                "micro_batch_num": micro_batch_num,
            }
        )
