"""GLM routed-MoE-only module used by the Janus-style expert runner.

The attention worker owns every non-routed component, including shared
experts.  Each expert worker therefore keeps only the FP32 gate, TopK and its
contiguous shard of physical routed experts.
"""

import copy
import re
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE, MoEGate
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    maybe_fuse_routed_scale_and_shared_add,
)
from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size


class GlmMoeLocalExperts(nn.Module):
    """A collection of GLM routed-expert layers for one E rank.

    The routed expert count does not necessarily divide the chosen number of
    E ranks.  ``FusedMoE`` consequently sees a padded virtual expert count,
    while the replicated gate and TopK retain the original 256-dimensional
    logits.  Padding IDs can never be selected and therefore require no
    checkpoint weights.
    """

    def __init__(self, config, layer_ids: Sequence[int], quant_config):
        super().__init__()
        self.config = config
        self.num_routed_experts = config.n_routed_experts
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.padded_num_routed_experts = (
            (self.num_routed_experts + self.moe_ep_size - 1) // self.moe_ep_size
        ) * self.moe_ep_size
        self.layer_ids = set(layer_ids)
        self.layers = nn.ModuleDict()

        padded_config = copy.deepcopy(config)
        padded_config.n_routed_experts = self.padded_num_routed_experts
        # Shared experts are executed on A and must not be allocated on E.
        object.__setattr__(padded_config, "n_shared_experts", None)

        for layer_id in layer_ids:
            block = DeepseekV2MoE(
                config=padded_config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"model.layers.{layer_id}.mlp",
            )
            # The block above allocates padded experts for EP=6.  Routing must
            # remain in the real 256-expert ID domain and must match the
            # GLM5.1 DSA baseline, which uses DeepseekV2MoE/MoEGate rather than
            # the older Glm4Moe gate path.
            block.gate = MoEGate(
                config=config,
                quant_config=quant_config,
                prefix=f"model.layers.{layer_id}.mlp.gate",
            )
            block.topk = TopK(
                top_k=config.num_experts_per_tok,
                layer_id=layer_id,
                renormalize=config.norm_topk_prob,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                correction_bias=block.gate.e_score_correction_bias,
                quant_config=quant_config,
                routed_scaling_factor=config.routed_scaling_factor,
                num_fused_shared_experts=0,
                apply_routed_scaling_factor_on_output=getattr(
                    block.experts, "should_fuse_routed_scaling_factor_in_topk", False
                ),
                fused_shared_experts_scaling_factor=None,
                output_format=(
                    TopKOutputFormat.STANDARD
                    if (quant_config is None)
                    and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    else None
                ),
            )
            self.layers[str(layer_id)] = block

    def forward_with_gate(self, layer_id: int, hidden_states: torch.Tensor):
        block = self.layers[str(layer_id)]
        router_logits = block.gate(hidden_states)
        topk_output = block.topk(hidden_states, router_logits)
        output = block.experts(hidden_states, topk_output)
        # Match DeepseekV2MoE.forward_normal scaling semantics.  The A worker
        # owns shared experts, so E applies only the routed-only branch of the
        # same helper used by the baseline path.
        output = maybe_fuse_routed_scale_and_shared_add(
            block.experts,
            output,
            None,
            block.routed_scaling_factor,
        )
        return output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load only gate/TopK metadata and this rank's physical experts.

        The mapping is deliberately the GLM model's existing FusedMoE mapping;
        this keeps compressed-tensors/AWQ scale naming identical to the normal
        SGLang loader instead of creating a second checkpoint format.
        """
        params = dict(self.named_parameters())
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_routed_experts,
        )
        pattern = re.compile(r"^model\.layers\.(\d+)\.mlp\.(.+)$")

        for name, weight in weights:
            match = pattern.match(name)
            if match is None or int(match.group(1)) not in self.layer_ids:
                continue
            layer_id = match.group(1)
            suffix = match.group(2)

            if suffix in {"gate.weight", "gate.e_score_correction_bias"}:
                param = params.get(f"layers.{layer_id}.{suffix}")
                if param is not None:
                    default_weight_loader(param, weight)
                continue

            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in suffix:
                    continue
                mapped_suffix = suffix.replace(weight_name, param_name)
                param = params.get(f"layers.{layer_id}.{mapped_suffix}")
                if param is not None:
                    param.weight_loader(
                        param,
                        weight,
                        f"layers.{layer_id}.{mapped_suffix}",
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                break
