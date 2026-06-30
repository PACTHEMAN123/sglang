"""Janus-compatible AE expert-group metadata."""


def get_ep_group_info(
    tp_size: int,
    ep_size: int,
    moe_node_num: int,
    enable_ep_intra_node_reduce: bool,
):
    """Build the same ``ep_group_info`` structure used by Janus.

    For the first GLM prototype ``tp_size=1``, ``ep_size=7`` and one E group,
    so the only attention rank sends to and receives from all seven experts.
    """
    world_size = tp_size + ep_size
    ep_ranks = list(range(tp_size, world_size))
    att_ranks = list(range(tp_size))
    ranks_per_node = ep_size // moe_node_num
    ep_groups = []
    if ranks_per_node <= 0:
        ep_groups = [ep_ranks]
    else:
        for index in range(moe_node_num):
            start = index * ranks_per_node
            end = ep_size if index == moe_node_num - 1 else start + ranks_per_node
            ep_groups.append(ep_ranks[start:end])

    send_strategy = {}
    if enable_ep_intra_node_reduce and ep_groups:
        for att_rank in att_ranks:
            send_strategy[att_rank] = [
                group[att_rank % len(group)] for group in ep_groups if group
            ]
    else:
        for att_rank in att_ranks:
            send_strategy[att_rank] = ep_ranks

    return {
        "ep_groups": ep_groups,
        "ep_ranks": ep_ranks,
        "att_ranks": att_ranks,
        "send_strategy": send_strategy,
        "enable_ep_intra_node_reduce": enable_ep_intra_node_reduce,
        "moe_node_num": moe_node_num,
        "tp_size": tp_size,
        "ep_size": ep_size,
    }
