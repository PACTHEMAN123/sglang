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
        init_distributed_environment(
            backend="nccl",
            world_size=self.server_args.tp_size + self.server_args.ep_size,
            rank=self.tp_rank,
            local_rank=self.gpu_id,
            distributed_init_method="tcp://" + self.server_args.dist_init_addr,
        )
        # Required before GLM decides whether shared experts are fused.
        self.server_args.disable_shared_experts_fusion = True
        set_global_server_args_for_scheduler(self.server_args)
        initialize_ae_model_parallel(
            attention_rank_count=self.server_args.tp_size,
            expert_rank_count=self.server_args.ep_size,
        )
        initialize_dp_attention(
            server_args=self.server_args, model_config=self.model_config
        )
        return get_available_gpu_memory(self.device, self.gpu_id)

    def initialize(self, pre_model_load_memory: float):
        super().initialize(pre_model_load_memory)
        self.init_nvshmem_distributed()

    def init_nvshmem_distributed(self):
        """Exact allocation order shared with Janus MoeRunner."""
        ep_group_info = get_ep_group_info(
            tp_size=self.server_args.tp_size,
            ep_size=self.server_args.ep_size,
            moe_node_num=1,
            enable_ep_intra_node_reduce=False,
        )
        global_server_args_dict["ep_group_info"] = ep_group_info
        init_nvshmem_by_torch_process_group(dist.group.WORLD)

        hidden_dim = self.model_config.hf_config.hidden_size
        max_tokens = int(os.environ.get("NVSHMEM_MAX_TOKENS", 4096))
        buf_numel = max_tokens * hidden_dim
        micro_batch_num = self.server_args.micro_batch_num
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
                "ep_size": self.server_args.ep_size,
                "micro_batch_num": micro_batch_num,
            }
        )
