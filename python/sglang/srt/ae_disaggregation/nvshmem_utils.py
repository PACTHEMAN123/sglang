"""NVSHMEM helpers ported from Janus/CUHKSZ/disaggregate/communication."""

import os

import nvshmem.core
import torch


def torch_rank_to_pe_rank(torch_rank: int) -> int:
    """Map torch ranks to node-contiguous NVSHMEM PE ranks."""
    tp_size = int(os.environ.get("NVSHMEM_TP_SIZE", 0))
    ep_per_expert_node = int(os.environ.get("NVSHMEM_EP_PER_EXPERT_NODE", 0))
    target_per_node = int(os.environ.get("NVSHMEM_TARGET_PE_PER_NODE", 0))
    dummy_pes = int(os.environ.get("NVSHMEM_DUMMY_PES", 0))
    attn_node_num = int(os.environ.get("NVSHMEM_ATTN_NODE_NUM", 1))

    if dummy_pes > 0 and (
        tp_size <= 0 or target_per_node <= 0 or ep_per_expert_node <= 0
    ):
        raise RuntimeError(
            "NVSHMEM_DUMMY_PES requires NVSHMEM_TP_SIZE, "
            "NVSHMEM_EP_PER_EXPERT_NODE and NVSHMEM_TARGET_PE_PER_NODE."
        )
    if tp_size <= 0 or target_per_node <= 0 or torch_rank < tp_size:
        return torch_rank
    if ep_per_expert_node <= 0:
        return torch_rank

    ep_local = torch_rank - tp_size
    expert_node_k = ep_local // ep_per_expert_node
    local_pos = ep_local % ep_per_expert_node
    return (expert_node_k + attn_node_num) * target_per_node + local_pos


def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    """Initialize NVSHMEM from the same torch process group as Janus."""
    rank_id = pg.rank()
    dummy_pes = int(os.environ.get("NVSHMEM_DUMMY_PES", 0))
    nvshmem_nranks = pg.size() + dummy_pes
    my_pe_rank = torch_rank_to_pe_rank(rank_id)
    if not 0 <= my_pe_rank < nvshmem_nranks:
        raise RuntimeError(
            f"NVSHMEM PE rank {my_pe_rank} is outside [0, {nvshmem_nranks})."
        )

    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)

    uid = broadcast_objects[0]
    uid_file = os.environ.get("NVSHMEM_UID_FILE")
    if uid_file and rank_id == 0:
        # Keep this byte-for-byte compatible with the Janus bootstrap path.
        # The 1:7 deployment does not use dummy PEs, but retaining it makes
        # the same launch/debug tooling usable when it does.
        import pickle

        with open(uid_file, "wb") as file:
            pickle.dump(uid, file)
        print(f"[nvshmem_utils] UID written to {uid_file}", flush=True)

    print(
        f"[nvshmem_utils] torch_rank={rank_id} -> pe_rank={my_pe_rank} "
        f"(nranks={nvshmem_nranks})",
        flush=True,
    )

    from cuda.core.experimental import Device

    nvshmem.core.init(
        device=Device(torch.cuda.current_device()),
        uid=uid,
        rank=my_pe_rank,
        nranks=nvshmem_nranks,
        initializer_method="uid",
    )


def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def nvshmem_free_tensor_sync(tensor):
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


class TorchStreamWrapper:
    def __init__(self, pt_stream: torch.cuda.Stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)
