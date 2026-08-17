# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DeepSeek V4 C4 indexer fallback using FP16 HMMA on SM70."""

import os
import warnings

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32,
)
from vllm.triton_utils import tl, triton

_INDEX_HEAD_DIM = 128
_INDEX_CACHE_BYTES = _INDEX_HEAD_DIM + 4

# Cap (MiB) on the decode gathered_k workspace. Uncapped, the allocation is
# [rows, max_seq_len, 128] fp16 - with a full spec-decode batch and a long
# max_seq_len this can transiently request several GiB in one shot, which is
# enough to OOM-kill all TP workers on a V100 where only a few hundred MiB
# are typically free during decode. Chunking by rows is numerically exact
# (each bmm row is independent). 0 restores the original single-shot
# allocation. Read once at import time - no getenv in the hot path.
try:
    _DECODE_GATHER_CAP_MB = int(os.getenv("VLLM_SM70_INDEXER_CHUNK_MB", "256"))
except ValueError:
    _bad_value = os.getenv("VLLM_SM70_INDEXER_CHUNK_MB")
    warnings.warn(
        f"VLLM_SM70_INDEXER_CHUNK_MB is not an integer ({_bad_value!r}); "
        "using the default of 256 MiB.",
        stacklevel=1,
    )
    _DECODE_GATHER_CAP_MB = 256


@triton.jit
def _weighted_query_kernel(
    q_ptr,
    weights_ptr,
    out_ptr,
    q_stride0,
    q_stride1,
    weights_stride0,
    out_stride0,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    head_offsets = tl.arange(0, num_heads)
    dim_offsets = block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < head_dim
    q = tl.load(
        q_ptr
        + token_idx * q_stride0
        + head_offsets[:, None] * q_stride1
        + dim_offsets[None, :],
        mask=dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(weights_ptr + token_idx * weights_stride0 + head_offsets).to(
        tl.float32
    )
    combined = tl.sum(q * weights[:, None], axis=0)
    tl.store(
        out_ptr + token_idx * out_stride0 + dim_offsets,
        combined.to(out_ptr.type.element_ty),
        mask=dim_mask,
    )


@triton.jit
def _dequant_contiguous_index_k_kernel(
    k_ptr,
    scale_ptr,
    out_ptr,
    k_stride0,
    out_stride0,
    head_dim: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, head_dim)
    packed_ptr = k_ptr.to(tl.pointer_type(tl.uint8))
    values = tl.load(packed_ptr + row_idx * k_stride0 + offsets)
    scale = tl.load(scale_ptr + row_idx).to(tl.float32)
    dequant = fp8_e4m3fn_bits_to_fp32(values) * scale
    tl.store(
        out_ptr + row_idx * out_stride0 + offsets,
        dequant.to(out_ptr.type.element_ty),
    )


@triton.jit
def _dequant_paged_index_k_kernel(
    cache_ptr,
    block_table_ptr,
    seq_lens_ptr,
    out_ptr,
    cache_stride0,
    cache_stride1,
    block_table_stride0,
    out_stride0,
    out_stride1,
    cache_block_size,
    max_seq_len,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    key_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, head_dim)
    seq_len = tl.load(seq_lens_ptr + row_idx)
    valid = (key_offsets < seq_len) & (key_offsets < max_seq_len)
    block_in_seq = key_offsets // cache_block_size
    pos_in_block = key_offsets % cache_block_size
    physical_block = tl.load(
        block_table_ptr + row_idx * block_table_stride0 + block_in_seq,
        mask=valid,
        other=0,
    )
    token_ptr = (
        cache_ptr
        + physical_block.to(tl.int64) * cache_stride0
        + pos_in_block * cache_stride1
    )
    packed = tl.load(
        token_ptr[:, None] + dim_offsets[None, :],
        mask=valid[:, None],
        other=0,
    )
    fp8 = fp8_e4m3fn_bits_to_fp32(packed)
    scale_ptr = (token_ptr + head_dim).to(tl.pointer_type(tl.float32))
    scale = tl.load(scale_ptr, mask=valid, other=0.0).to(tl.float32)
    dequant = fp8 * scale[:, None]
    tl.store(
        out_ptr
        + row_idx * out_stride0
        + key_offsets[:, None] * out_stride1
        + dim_offsets[None, :],
        dequant.to(out_ptr.type.element_ty),
        mask=valid[:, None],
    )


def _combine_index_queries(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    assert q.dtype == torch.float16 and q.ndim == 3
    assert q.shape[-1] == _INDEX_HEAD_DIM
    assert weights.shape == q.shape[:2]
    out = torch.empty((q.shape[0], q.shape[-1]), dtype=torch.float16, device=q.device)
    block_d = 32
    _weighted_query_kernel[(q.shape[0], triton.cdiv(q.shape[-1], block_d))](
        q,
        weights,
        out,
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        out.stride(0),
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        BLOCK_D=block_d,
        num_warps=4,
    )
    return out


def sm70_indexer_prefill_logits(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale_storage: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute all prefill index scores; caller supplies causal row bounds."""
    assert k_quant.dtype == torch.float8_e4m3fn
    assert k_quant.ndim == 2 and k_quant.shape[1] == _INDEX_HEAD_DIM
    k_scales = k_scale_storage.view(torch.float32).reshape(-1)
    assert k_scales.shape[0] == k_quant.shape[0]

    weighted_q = _combine_index_queries(q, weights)
    k_fp16 = torch.empty(k_quant.shape, dtype=torch.float16, device=k_quant.device)
    _dequant_contiguous_index_k_kernel[(k_quant.shape[0],)](
        k_quant.view(torch.uint8),
        k_scales,
        k_fp16,
        k_quant.stride(0),
        k_fp16.stride(0),
        head_dim=_INDEX_HEAD_DIM,
        num_warps=4,
    )
    return torch.mm(weighted_q, k_fp16.t(), out_dtype=torch.float32)


def sm70_indexer_decode_logits(
    q: torch.Tensor,
    cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Gather paged FP8 index keys and compute batched decode scores."""
    assert cache.dtype == torch.uint8 and cache.ndim == 3
    assert cache.shape[-1] >= _INDEX_CACHE_BYTES
    weighted_q = _combine_index_queries(q, weights)

    if seq_lens.ndim == 2:
        next_n = seq_lens.shape[1]
        flat_lens = seq_lens.reshape(-1).to(torch.int32)
        block_table = block_table.repeat_interleave(next_n, dim=0)
    else:
        flat_lens = seq_lens.reshape(-1).to(torch.int32)
    assert flat_lens.shape[0] == weighted_q.shape[0]
    assert block_table.shape[0] == weighted_q.shape[0]

    max_seq_len = max(1, int(max_seq_len))
    total_rows = weighted_q.shape[0]
    block_n = 16

    row_bytes = max_seq_len * _INDEX_HEAD_DIM * 2  # fp16
    if _DECODE_GATHER_CAP_MB > 0:
        rows_per_chunk = max(1, (_DECODE_GATHER_CAP_MB * 1024 * 1024) // row_bytes)
    else:
        rows_per_chunk = total_rows

    if rows_per_chunk >= total_rows:
        # Single chunk covering every row - identical to the pre-chunking
        # behavior (same allocation, same kernel, same bmm).
        row_chunks = [(0, total_rows)]
    else:
        row_chunks = [
            (start, min(start + rows_per_chunk, total_rows))
            for start in range(0, total_rows, rows_per_chunk)
        ]

    out = torch.empty(
        (total_rows, max_seq_len), dtype=torch.float32, device=q.device
    )
    # Reused across chunks to avoid allocator churn under memory pressure.
    gathered_k = torch.empty(
        (min(rows_per_chunk, total_rows), max_seq_len, _INDEX_HEAD_DIM),
        dtype=torch.float16,
        device=q.device,
    )
    multi_chunk = len(row_chunks) > 1
    for start, end in row_chunks:
        rows = end - start
        gathered_view = gathered_k[:rows]
        if multi_chunk:
            # The kernel only writes positions < seq_len for each row, so a
            # reused buffer would otherwise carry stale K from a previous
            # chunk's request into this chunk's masked-out tail slots.
            # Downstream masking by context_lens already covers this, but
            # the memset (~0.3ms) removes the risk class outright.
            gathered_view.zero_()
        _dequant_paged_index_k_kernel[(rows, triton.cdiv(max_seq_len, block_n))](
            cache,
            block_table[start:end],
            flat_lens[start:end],
            gathered_view,
            cache.stride(0),
            cache.stride(1),
            block_table.stride(0),
            gathered_view.stride(0),
            gathered_view.stride(1),
            cache.shape[1],
            max_seq_len,
            head_dim=_INDEX_HEAD_DIM,
            BLOCK_N=block_n,
            num_warps=4,
        )
        out[start:end] = torch.bmm(
            weighted_q[start:end].unsqueeze(1),
            gathered_view.transpose(1, 2),
            out_dtype=torch.float32,
        ).squeeze(1)
    return out
