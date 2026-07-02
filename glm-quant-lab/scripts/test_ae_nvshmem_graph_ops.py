#!/usr/bin/env python3
"""Smoke test for experimental AE NVSHMEM graph-friendly device signaling.

Run with two local ranks, for example:

  torchrun --standalone --nproc_per_node=2 \
    glm-quant-lab/scripts/test_ae_nvshmem_graph_ops.py

The test validates two things:
  1. device-side NVSHMEM putmem_signal/wait can be JIT-built and executed;
  2. the same ops can be replayed from a CUDA graph while iteration counters
     advance on GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import nvshmem.core


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "sglang" / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sglang.srt.ae_disaggregation.nvshmem_graph_ops import (  # noqa: E402
    _load_module,
    exchange_signal,
)
from sglang.srt.ae_disaggregation.nvshmem_utils import (  # noqa: E402
    init_nvshmem_by_torch_process_group,
    nvshmem_create_tensor,
    torch_rank_to_pe_rank,
)


def _init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
    torch.cuda.set_device(local_rank)


def _assert_equal(name: str, got: torch.Tensor, expected: torch.Tensor):
    if not torch.equal(got.cpu(), expected.cpu()):
        raise AssertionError(f"{name}: got={got.cpu().tolist()} expected={expected.cpu().tolist()}")


def main():
    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"expected world_size=2, got {world}")

    init_nvshmem_by_torch_process_group(dist.group.WORLD)
    peer_rank = 1 - rank
    peer_pe = torch_rank_to_pe_rank(peer_rank)

    if rank == 0:
        _load_module()
    dist.barrier()
    if rank != 0:
        _load_module()
    dist.barrier()

    nelems = 8
    src = nvshmem_create_tensor((nelems,), torch.int64)
    dst = nvshmem_create_tensor((nelems,), torch.int64)
    sig = nvshmem.core.buffer(8)

    counter = torch.zeros(1, dtype=torch.int64, device="cuda")
    metadata = torch.zeros(4, dtype=torch.int64, device="cuda")
    debug = torch.zeros(2, dtype=torch.int64, device="cuda")

    if rank == 0:
        src.copy_(torch.arange(nelems, dtype=torch.int64, device="cuda") + 100)
    else:
        src.zero_()
        dst.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    exchange_signal(
        dst=dst.data_ptr(),
        src=src.data_ptr(),
        nbytes=src.numel() * src.element_size(),
        sig_addr=sig.handle,
        counter=counter,
        layer=7,
        tokens=nelems,
        peer_pe=peer_pe,
        role=0 if rank == 0 else 1,
        metadata=metadata,
        debug=debug,
    )
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 1:
        _assert_equal(
            "eager dst",
            dst,
            torch.arange(nelems, dtype=torch.int64, device="cuda") + 100,
        )
        _assert_equal(
            "eager metadata",
            metadata,
            torch.tensor([1, 7, nelems, (1 << 40) | (7 << 20) | nelems], device="cuda"),
        )

    # Graph replay: each replay must advance the GPU-side counter and therefore
    # use a fresh signal value even though Python passes no new iteration scalar.
    if rank == 0:
        src.add_(1000)
    else:
        metadata.zero_()
        dst.zero_()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        exchange_signal(
            dst=dst.data_ptr(),
            src=src.data_ptr(),
            nbytes=src.numel() * src.element_size(),
            sig_addr=sig.handle,
            counter=counter,
            layer=8,
            tokens=nelems,
            peer_pe=peer_pe,
            role=0 if rank == 0 else 1,
            metadata=metadata,
            debug=debug,
        )

    for replay_idx in range(3):
        if rank == 0:
            src.fill_(2000 + replay_idx)
        else:
            dst.zero_()
            metadata.zero_()
        torch.cuda.synchronize()
        dist.barrier()

        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 1:
            expected_iter = replay_idx + 2
            expected_signal = (expected_iter << 40) | (8 << 20) | nelems
            _assert_equal(
                f"graph dst {replay_idx}",
                dst,
                torch.full((nelems,), 2000 + replay_idx, dtype=torch.int64, device="cuda"),
            )
            _assert_equal(
                f"graph metadata {replay_idx}",
                metadata,
                torch.tensor([expected_iter, 8, nelems, expected_signal], device="cuda"),
            )

    if rank == 0:
        print("AE NVSHMEM graph ops smoke test passed", flush=True)


if __name__ == "__main__":
    main()
