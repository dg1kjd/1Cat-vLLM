# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP16 sparse MLA kernels for DeepSeek V4 on Volta."""

import torch

from vllm.models.deepseek_v4.common.ops.fp8_software import (
    fp8_e4m3fn_bits_to_fp32,
)
from vllm.triton_utils import tl, triton

_HEAD_DIM = 512
_NOPE_DIM = 448
_ROPE_DIM = 64


@triton.jit
def _sm70_sparse_gathered_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    lengths_ptr,
    sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    kv_stride_n,
    indices_stride_t,
    out_stride_t,
    out_stride_h,
    num_heads,
    num_kv,
    scale,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dim_offsets = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    running_max = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)
    valid_len = tl.load(lengths_ptr + query_idx)
    key_offsets = tl.arange(0, BLOCK_K)

    for start in tl.range(0, INDEX_WIDTH, BLOCK_K):
        positions = start + key_offsets
        in_range = positions < valid_len
        slots = tl.load(
            indices_ptr + query_idx * indices_stride_t + positions,
            mask=positions < INDEX_WIDTH,
            other=-1,
        )
        valid = in_range & (slots >= 0) & (slots < num_kv)
        safe_slots = tl.where(valid, slots, 0)
        kv = tl.load(
            kv_ptr + safe_slots[:, None] * kv_stride_n + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(kv.dtype), kv)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
        tl.float32
    )
    final_max = tl.maximum(running_max, sink)
    alpha = tl.exp(running_max - final_max)
    final_sum = running_sum * alpha + tl.exp(sink - final_max)
    denom = tl.maximum(final_sum, 1.0e-30)
    result = tl.where(
        final_sum[:, None] > 0.0,
        acc * alpha[:, None] / denom[:, None],
        0.0,
    )
    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :],
        result,
        mask=head_mask[:, None],
    )


@triton.jit
def _sm70_sparse_paged_fp8_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    out_stride_t,
    out_stride_h,
    main_cache_stride0,
    extra_cache_stride0,
    main_indices_stride0,
    extra_indices_stride0,
    main_num_rows,
    extra_num_rows,
    main_block_size,
    extra_block_size,
    scale,
    num_heads,
    HAS_EXTRA: tl.constexpr,
    MAIN_WIDTH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    NOPE_BLOCK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_BLOCK)
    nope_mask = nope_offsets < NOPE_DIM
    rope_offsets = tl.arange(0, ROPE_DIM)

    q_row = q_ptr + query_idx * q_stride_t + head_offsets[:, None] * q_stride_h
    q_nope = tl.load(
        q_row + nope_offsets[None, :],
        mask=head_mask[:, None] & nope_mask[None, :],
        other=0.0,
    )
    q_rope = tl.load(
        q_row + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    running_max = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc_nope = tl.zeros((BLOCK_H, NOPE_BLOCK), dtype=tl.float32)
    acc_rope = tl.zeros((BLOCK_H, ROPE_DIM), dtype=tl.float32)
    key_offsets = tl.arange(0, BLOCK_K)
    main_len = tl.load(main_lengths_ptr + query_idx)

    for start in tl.range(0, MAIN_WIDTH, BLOCK_K):
        positions = start + key_offsets
        in_range = positions < main_len
        slots = tl.load(
            main_indices_ptr + query_idx * main_indices_stride0 + positions,
            mask=positions < MAIN_WIDTH,
            other=-1,
        )
        valid = in_range & (slots >= 0) & (slots < main_num_rows)
        safe_slots = tl.where(valid, slots, 0)
        block_idx = safe_slots // main_block_size
        pos_in_block = safe_slots % main_block_size
        cache_block = main_cache_ptr + block_idx.to(tl.int64) * main_cache_stride0
        token_data = cache_block + pos_in_block * 576
        token_scales = cache_block + main_block_size * 576 + pos_in_block * 8

        packed = tl.load(
            token_data[:, None] + nope_offsets[None, :],
            mask=valid[:, None] & nope_mask[None, :],
            other=0,
        )
        fp8 = fp8_e4m3fn_bits_to_fp32(packed)
        encoded_scale = tl.load(
            token_scales[:, None] + nope_offsets[None, :] // 64,
            mask=valid[:, None] & nope_mask[None, :],
            other=127,
        )
        dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        k_nope = fp8.to(tl.float16) * dequant_scale.to(tl.float16)
        k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, 0.0)

        rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
        k_rope = tl.load(
            rope_ptr[:, None] + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float16)

        scores = tl.dot(q_nope, tl.trans(k_nope))
        scores += tl.dot(q_rope, tl.trans(k_rope))
        scores *= scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        alpha = tl.exp(running_max - new_max)
        probs = tl.exp(scores - new_max[:, None])
        probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
        acc_nope = acc_nope * alpha[:, None] + tl.dot(probs.to(k_nope.dtype), k_nope)
        acc_rope = acc_rope * alpha[:, None] + tl.dot(probs.to(k_rope.dtype), k_rope)
        running_sum = running_sum * alpha + tl.sum(probs, axis=1)
        running_max = new_max

    if HAS_EXTRA:
        extra_len = tl.load(extra_lengths_ptr + query_idx)
        # The C128 logical width changes with context length. Drive this loop
        # from device metadata so one FULL CUDA Graph remains valid as context
        # grows, while the underlying row stride stays fixed.
        for start in range(0, extra_len, BLOCK_K):
            positions = start + key_offsets
            in_range = positions < extra_len
            slots = tl.load(
                extra_indices_ptr + query_idx * extra_indices_stride0 + positions,
                mask=in_range,
                other=-1,
            )
            valid = in_range & (slots >= 0) & (slots < extra_num_rows)
            safe_slots = tl.where(valid, slots, 0)
            block_idx = safe_slots // extra_block_size
            pos_in_block = safe_slots % extra_block_size
            cache_block = extra_cache_ptr + block_idx.to(tl.int64) * extra_cache_stride0
            token_data = cache_block + pos_in_block * 576
            token_scales = cache_block + extra_block_size * 576 + pos_in_block * 8

            packed = tl.load(
                token_data[:, None] + nope_offsets[None, :],
                mask=valid[:, None] & nope_mask[None, :],
                other=0,
            )
            fp8 = fp8_e4m3fn_bits_to_fp32(packed)
            encoded_scale = tl.load(
                token_scales[:, None] + nope_offsets[None, :] // 64,
                mask=valid[:, None] & nope_mask[None, :],
                other=127,
            )
            dequant_scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
            k_nope = fp8.to(tl.float16) * dequant_scale.to(tl.float16)
            k_nope = tl.where(valid[:, None] & nope_mask[None, :], k_nope, 0.0)

            rope_ptr = (token_data + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
            k_rope = tl.load(
                rope_ptr[:, None] + rope_offsets[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float16)

            scores = tl.dot(q_nope, tl.trans(k_nope))
            scores += tl.dot(q_rope, tl.trans(k_rope))
            scores *= scale
            scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            alpha = tl.exp(running_max - new_max)
            probs = tl.exp(scores - new_max[:, None])
            probs = tl.where(head_mask[:, None] & valid[None, :], probs, 0.0)
            acc_nope = acc_nope * alpha[:, None] + tl.dot(
                probs.to(k_nope.dtype), k_nope
            )
            acc_rope = acc_rope * alpha[:, None] + tl.dot(
                probs.to(k_rope.dtype), k_rope
            )
            running_sum = running_sum * alpha + tl.sum(probs, axis=1)
            running_max = new_max

    sink = tl.load(sink_ptr + head_offsets, mask=head_mask, other=neg_large).to(
        tl.float32
    )
    final_max = tl.maximum(running_max, sink)
    alpha = tl.exp(running_max - final_max)
    final_sum = running_sum * alpha + tl.exp(sink - final_max)
    denom = tl.maximum(final_sum, 1.0e-30)
    out_nope = tl.where(
        final_sum[:, None] > 0.0,
        acc_nope * alpha[:, None] / denom[:, None],
        0.0,
    )
    out_rope = tl.where(
        final_sum[:, None] > 0.0,
        acc_rope * alpha[:, None] / denom[:, None],
        0.0,
    )
    out_row = out_ptr + query_idx * out_stride_t + head_offsets[:, None] * out_stride_h
    tl.store(
        out_row + nope_offsets[None, :],
        out_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        out_row + NOPE_DIM + rope_offsets[None, :],
        out_rope,
        mask=head_mask[:, None],
    )


def sm70_sparse_attention_gathered(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Sparse FP16 attention over a contiguous gathered KV workspace."""
    assert q.dtype == kv.dtype == out.dtype == torch.float16
    assert q.shape[-1] == kv.shape[-1] == out.shape[-1] == _HEAD_DIM
    kv_2d = kv.reshape(-1, _HEAD_DIM)
    indices_2d = indices.reshape(indices.shape[0], -1)
    lengths = lengths.reshape(-1).to(torch.int32)
    assert indices_2d.shape[0] == q.shape[0] == lengths.shape[0]

    block_h = 8
    _sm70_sparse_gathered_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        kv_2d,
        indices_2d,
        lengths,
        attn_sink.contiguous(),
        out,
        q.stride(0),
        q.stride(1),
        kv_2d.stride(0),
        indices_2d.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[1],
        kv_2d.shape[0],
        float(scale),
        INDEX_WIDTH=indices_2d.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=16,
        BLOCK_D=_HEAD_DIM,
        num_warps=4,
    )


def sm70_sparse_attention_paged_fp8(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
) -> None:
    """Decode directly from the packed DeepSeek FP8 paged-cache layout."""
    assert q.dtype == out.dtype == torch.float16
    assert q.shape == out.shape and q.shape[-1] == _HEAD_DIM
    assert main_cache.dtype == torch.uint8 and main_cache.ndim == 3

    main_indices_2d = main_indices.reshape(q.shape[0], -1)
    main_lengths = main_lengths.reshape(-1).to(torch.int32)
    has_extra = (
        extra_cache is not None
        and extra_indices is not None
        and extra_lengths is not None
    )
    if has_extra:
        assert extra_cache is not None
        assert extra_indices is not None
        assert extra_lengths is not None
        assert extra_cache.dtype == torch.uint8 and extra_cache.ndim == 3
        extra_indices_2d = extra_indices.reshape(q.shape[0], -1)
        extra_lengths_1d = extra_lengths.reshape(-1).to(torch.int32)
    else:
        extra_cache = main_cache
        extra_indices_2d = main_indices_2d[:, :1]
        extra_lengths_1d = torch.zeros_like(main_lengths)

    block_h = 8
    _sm70_sparse_paged_fp8_kernel[(q.shape[0], triton.cdiv(q.shape[1], block_h))](
        q,
        main_cache,
        main_indices_2d,
        main_lengths,
        extra_cache,
        extra_indices_2d,
        extra_lengths_1d,
        attn_sink.contiguous(),
        out,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_indices_2d.stride(0),
        extra_indices_2d.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_cache.shape[1],
        extra_cache.shape[1],
        float(scale),
        q.shape[1],
        HAS_EXTRA=has_extra,
        MAIN_WIDTH=main_indices_2d.shape[1],
        BLOCK_H=block_h,
        BLOCK_K=16,
        NOPE_DIM=_NOPE_DIM,
        NOPE_BLOCK=_HEAD_DIM,
        ROPE_DIM=_ROPE_DIM,
        num_warps=4,
    )
