from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Triton implementation
@triton.jit
def _act_quant_kernel(
    X_ptr,
    Y_ptr,
    S_ptr,
    M,
    N,
    group_size: tl.constexpr,
    round_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for activation quantization.

    Each block processes BLOCK_M rows and group_size columns.
    """
    # Get block IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # FP8 constants
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1.0 / fp8_max

    # Calculate row and column offsets
    row_start = pid_m * BLOCK_M
    col_start = pid_n * group_size

    # Create offset arrays
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Mask for valid rows and columns
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    # Load input data
    x_ptrs = X_ptr + rows[:, None] * N + cols[None, :]
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute absolute max along columns (group_size dimension) for each row
    x_abs = tl.abs(x)
    amax = tl.max(x_abs, axis=1)  # Shape: (BLOCK_M,)

    # Clamp amax to avoid division by zero
    amax = tl.maximum(amax, 1e-4)

    # Compute scale
    if round_scale:
        # Fast round scale using bit manipulation approximation
        # This is a simplified version - the exact bit manipulation is harder in Triton
        # Using log2 + ceil + pow2 as approximation
        log_val = tl.log2(amax * fp8_max_inv)
        log_ceil = tl.ceil(log_val)
        scale = tl.exp2(log_ceil)
    else:
        scale = amax * fp8_max_inv

    # Quantize: y = clamp(x / scale, fp8_min, fp8_max)
    scale_broadcast = scale[:, None]
    y = x / scale_broadcast
    y = tl.minimum(tl.maximum(y, fp8_min), fp8_max)

    # Store quantized output
    y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
    tl.store(y_ptrs, y, mask=mask)

    # Store scales
    s_cols = pid_n
    s_ptrs = S_ptr + rows * (N // group_size) + s_cols
    s_mask = row_mask
    tl.store(s_ptrs, scale, mask=s_mask)


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization with Triton.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.
        scale_fmt (Optional[str], optional): The format of the scale. Default is None.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"

    # Flatten all dims except last
    N = x.size(-1)
    x_flat = x.view(-1, N)
    M = x_flat.size(0)

    # Allocate output tensors
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    y_flat = y.view(-1, N)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    s_flat = s.view(-1, N // block_size)

    # Launch kernel
    BLOCK_M = 32
    BLOCK_N = block_size
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_size))
    round_scale = scale_fmt is not None

    _act_quant_kernel[grid](
        x_flat,
        y_flat,
        s_flat,
        M,
        N,
        group_size=block_size,
        round_scale=round_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_stages=0 if round_scale else 2,
    )

    return y, s


@triton.jit
def _get_valid_kv_indices_kernel(
    page_table_ptr,  # [bs, topk]
    kv_indptr_ptr,  # [bs + 1]
    kv_indices_ptr,  # [bs * topk] output buffer
    bs: tl.constexpr,
    topk: tl.constexpr,
):
    """
    Extract valid indices (non -1) from page_table into kv_indices.
    Each program handles one batch.
    """
    batch_id = tl.program_id(0)

    # Get the start position for this batch in kv_indices
    dst_start = tl.load(kv_indptr_ptr + batch_id)

    # Load all topk indices for this batch
    src_offset = batch_id * topk
    offsets = tl.arange(0, topk)
    indices = tl.load(page_table_ptr + src_offset + offsets)

    # Count valid indices and compact them
    mask = indices != -1

    # Use prefix sum to compute destination positions for valid elements
    # For each position, count how many valid elements are before it
    prefix_sum = tl.cumsum(mask.to(tl.int32), axis=0) - 1

    # Store valid indices to their compacted positions
    dst_positions = dst_start + prefix_sum
    tl.store(kv_indices_ptr + dst_positions, indices, mask=mask)


def get_valid_kv_indices(
    page_table_1: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    bs: int,
):
    """
    Extract valid indices from page_table_1 into kv_indices buffer.

    Args:
        page_table_1: [bs, topk] page table with -1 as invalid
        kv_indptr: [bs + 1] cumulative count of valid indices per batch
        kv_indices: [bs * topk] pre-allocated output buffer
        bs: batch size
    """
    topk = page_table_1.shape[1]
    grid = (bs,)
    _get_valid_kv_indices_kernel[grid](
        page_table_1,
        kv_indptr,
        kv_indices,
        bs,
        topk,
    )


@triton.jit
def _sparse_mla_fwd_kernel(
    q_ptr,
    kv_ptr,
    page_table_ptr,
    out_ptr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    topk: tl.constexpr,
    head_dim: tl.constexpr,
    v_head_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    sm_scale: tl.constexpr,
    logit_cap: tl.constexpr,
    has_logit_cap: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_v = tl.program_id(1)
    pid_h = tl.program_id(2) * BLOCK_H

    v_offsets = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offsets < v_head_dim
    h_offsets = pid_h + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < num_heads

    acc = tl.zeros((BLOCK_H, BLOCK_V), dtype=tl.float32)
    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_H,), 0.0, dtype=tl.float32)

    for k_start in range(0, topk, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < topk
        indices = tl.load(
            page_table_ptr + pid_t * topk + k_offsets,
            mask=k_mask,
            other=-1,
        )
        valid = k_mask & (indices >= 0)
        safe_indices = tl.maximum(indices, 0)

        scores = tl.zeros((BLOCK_H, BLOCK_K), dtype=tl.float32)
        for d_start in range(0, head_dim, BLOCK_D):
            d_offsets = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offsets < head_dim
            q = tl.load(
                q_ptr
                + (pid_t * num_heads + h_offsets[:, None]) * head_dim
                + d_offsets[None, :],
                mask=h_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            keys = tl.load(
                kv_ptr + d_offsets[:, None] + safe_indices[None, :] * kv_dim,
                mask=d_mask[:, None] & valid[None, :],
                other=0.0,
            )
            scores += tl.dot(q, keys)

        scores *= sm_scale
        if has_logit_cap:
            capped = scores / logit_cap
            scores = logit_cap * (2.0 / (1.0 + tl.exp(-2.0 * capped)) - 1.0)
        scores = tl.where(valid[None, :] & h_mask[:, None], scores, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_ij = l_i * alpha + tl.sum(p, axis=1)

        values = tl.load(
            kv_ptr + safe_indices[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(values.dtype), values)
        m_i = m_ij
        l_i = l_ij

    acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr
        + (pid_t * num_heads + h_offsets[:, None]) * v_head_dim
        + v_offsets[None, :],
        acc,
        mask=h_mask[:, None] & v_mask[None, :],
    )


def sparse_mla_sm80(
    q: torch.Tensor,
    kv: torch.Tensor,
    page_table_1: torch.Tensor,
    sm_scale: float,
    v_head_dim: int,
    logit_cap: Optional[float],
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """SM80-friendly sparse MLA for the NSA torch fallback path."""
    assert q.is_cuda and kv.is_cuda and page_table_1.is_cuda
    assert q.dim() == 3 and kv.dim() == 2 and page_table_1.dim() == 2
    assert q.is_contiguous() and kv.is_contiguous() and page_table_1.is_contiguous()

    num_tokens, num_heads, head_dim = q.shape
    topk = page_table_1.shape[1]
    kv_dim = kv.shape[1]
    if out is None:
        out = torch.empty(
            (num_tokens, num_heads, v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
    else:
        assert out.shape == (num_tokens, num_heads, v_head_dim)
        assert out.dtype == q.dtype and out.device == q.device
        assert out.is_contiguous()

    block_k = 32
    block_d = 64
    block_h = 16
    has_logit_cap = logit_cap is not None and logit_cap > 0
    block_v = 128 if v_head_dim >= 128 else triton.next_power_of_2(v_head_dim)
    _sparse_mla_fwd_kernel[
        (num_tokens, triton.cdiv(v_head_dim, block_v), triton.cdiv(num_heads, block_h))
    ](
        q,
        kv,
        page_table_1,
        out,
        num_tokens,
        num_heads,
        topk,
        head_dim,
        v_head_dim,
        kv_dim,
        float(sm_scale),
        float(logit_cap or 0.0),
        has_logit_cap,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        BLOCK_V=block_v,
        num_warps=4,
    )
    return out
