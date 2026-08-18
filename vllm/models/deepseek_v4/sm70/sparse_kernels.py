# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FP16 sparse MLA kernels for DeepSeek V4 on Volta."""

import torch

from vllm import envs
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


@triton.jit
def _sm70_sparse_paged_fp8_split_kernel(
    q_ptr,
    main_cache_ptr,
    main_indices_ptr,
    main_lengths_ptr,
    extra_cache_ptr,
    extra_indices_ptr,
    extra_lengths_ptr,
    pacc_ptr,
    pmax_ptr,
    psum_ptr,
    q_stride_t,
    q_stride_h,
    pacc_stride_t,
    pacc_stride_h,
    pacc_stride_s,
    pms_stride_t,
    pms_stride_h,
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
    NUM_SPLITS: tl.constexpr,
):
    """One split of the key axis, emitting an unnormalised online-softmax partial.

    Same arithmetic as `_sm70_sparse_paged_fp8_kernel`, but this program only
    visits its own slice of the main and extra key ranges and writes
    (acc, running_max, running_sum) instead of the normalised result. The
    attention sink is deliberately NOT applied here -- it belongs to the
    denominator exactly once, so the combine pass owns it.
    """
    query_idx = tl.program_id(0)
    head_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split_id = tl.program_id(2)
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

    # MAIN_WIDTH is a compile-time bound, so each split gets a fixed, equal
    # number of BLOCK_K chunks and the trip count stays static.
    MAIN_CHUNKS: tl.constexpr = (MAIN_WIDTH + BLOCK_K - 1) // BLOCK_K
    MAIN_PER_SPLIT: tl.constexpr = (MAIN_CHUNKS + NUM_SPLITS - 1) // NUM_SPLITS

    for i in tl.static_range(MAIN_PER_SPLIT):
        start = (split_id * MAIN_PER_SPLIT + i) * BLOCK_K
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
        # The C128 logical width changes with context length, so this split's
        # slice has to be derived on device to keep one FULL CUDA Graph valid.
        # The grid itself stays static -- only the trip count varies.
        extra_chunks = (extra_len + BLOCK_K - 1) // BLOCK_K
        chunks_per_split = (extra_chunks + NUM_SPLITS - 1) // NUM_SPLITS
        lo = split_id * chunks_per_split * BLOCK_K
        hi = tl.minimum(lo + chunks_per_split * BLOCK_K, extra_len)
        for start in range(lo, hi, BLOCK_K):
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

    pacc_row = (
        pacc_ptr
        + query_idx * pacc_stride_t
        + head_offsets[:, None] * pacc_stride_h
        + split_id * pacc_stride_s
    )
    tl.store(
        pacc_row + nope_offsets[None, :],
        acc_nope,
        mask=head_mask[:, None] & nope_mask[None, :],
    )
    tl.store(
        pacc_row + NOPE_DIM + rope_offsets[None, :],
        acc_rope,
        mask=head_mask[:, None],
    )
    pms_off = query_idx * pms_stride_t + head_offsets * pms_stride_h + split_id
    tl.store(pmax_ptr + pms_off, running_max, mask=head_mask)
    tl.store(psum_ptr + pms_off, running_sum, mask=head_mask)


@triton.jit
def _sm70_sparse_split_combine_kernel(
    pacc_ptr,
    pmax_ptr,
    psum_ptr,
    sink_ptr,
    out_ptr,
    pacc_stride_t,
    pacc_stride_h,
    pacc_stride_s,
    pms_stride_t,
    pms_stride_h,
    out_stride_t,
    out_stride_h,
    HEAD_BLOCK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Merge the per-split partials into the final normalised output.

    Folds in the attention sink here, once, as a logit that contributes
    exp(sink) to the denominator and nothing to the numerator -- matching the
    single-block kernel's tail exactly.
    """
    query_idx = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, NUM_SPLITS)
    dims = tl.arange(0, HEAD_BLOCK)

    pms_off = query_idx * pms_stride_t + head * pms_stride_h + splits
    pmax = tl.load(pmax_ptr + pms_off)
    psum = tl.load(psum_ptr + pms_off)
    sink = tl.load(sink_ptr + head).to(tl.float32)

    # final_max >= sink, so exp(sink - final_max) can never overflow.
    final_max = tl.maximum(tl.max(pmax, axis=0), sink)
    weight = tl.exp(pmax - final_max)
    final_sum = tl.sum(psum * weight, axis=0) + tl.exp(sink - final_max)

    acc = tl.load(
        pacc_ptr
        + query_idx * pacc_stride_t
        + head * pacc_stride_h
        + splits[:, None] * pacc_stride_s
        + dims[None, :]
    )
    merged = tl.sum(acc * weight[:, None], axis=0)
    denom = tl.maximum(final_sum, 1.0e-30)
    result = tl.where(final_sum > 0.0, merged / denom, 0.0)
    tl.store(
        out_ptr + query_idx * out_stride_t + head * out_stride_h + dims,
        result,
    )


# Split-KV scratch. Held at module scope and only ever grown, so the pointers
# baked into a captured CUDA Graph stay valid: vLLM captures decode shapes
# largest-first, which sizes this before any replay.
_split_workspace: dict[tuple[int, str], torch.Tensor] = {}


def _split_workspace_for(
    num_tokens: int, num_heads: int, splits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_numel = num_tokens * num_heads * splits * _HEAD_DIM
    ms_numel = num_tokens * num_heads * splits
    need = acc_numel + 2 * ms_numel

    # Reserve generously on the first call: round the token count up to a power
    # of two (min 8) and always reserve the maximum split count. Growing the
    # buffer *during* CUDA graph capture would bake a pointer from that graph's
    # private pool into every later replay, so the goal is to never grow at all
    # once decode has started.
    cap_tokens = max(8, 1 << max(0, num_tokens - 1).bit_length())
    cap_splits = max(splits, envs.VLLM_SM70_SPARSE_KV_MAX_SPLITS)
    capacity = cap_tokens * num_heads * cap_splits * (_HEAD_DIM + 2)

    key = (device.index if device.index is not None else 0, str(device.type))
    buf = _split_workspace.get(key)
    if buf is None or buf.numel() < need:
        buf = torch.empty(max(need, capacity), dtype=torch.float32, device=device)
        _split_workspace[key] = buf
    pacc = buf[:acc_numel].view(num_tokens, num_heads, splits, _HEAD_DIM)
    pmax = buf[acc_numel : acc_numel + ms_numel].view(num_tokens, num_heads, splits)
    psum = buf[acc_numel + ms_numel : need].view(num_tokens, num_heads, splits)
    return pacc, pmax, psum


def _choose_kv_splits(num_tokens: int, head_blocks: int, main_width: int) -> int:
    """How many key-axis splits to run, or 1 to keep the single-pass kernel.

    Decided purely from host-side static shapes so one captured CUDA Graph
    stays valid; the dynamic `extra_len` only changes each split's trip count.
    """
    if not envs.VLLM_SM70_SPARSE_SPLIT_KV:
        return 1
    blocks = num_tokens * head_blocks
    target = envs.VLLM_SM70_SPARSE_KV_SPLIT_TARGET_BLOCKS
    if blocks >= target:
        return 1
    splits = 1
    while splits * 2 * blocks <= target and splits < envs.VLLM_SM70_SPARSE_KV_MAX_SPLITS:
        splits *= 2
    # More splits than there are main chunks only helps the (dynamic) extra
    # pass; that is still a win at long context, so only clamp the degenerate
    # case where there is nothing to split at all.
    return splits if splits > 1 and main_width > 0 else 1


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
    num_tokens, num_heads = q.shape[0], q.shape[1]
    head_blocks = triton.cdiv(num_heads, block_h)
    main_width = main_indices_2d.shape[1]

    splits = _choose_kv_splits(num_tokens, head_blocks, main_width)
    if splits > 1:
        # Flash-decoding: the single-pass kernel parallelises only over
        # (token, head block), which at batch-1 decode is ONE block of 4 warps
        # on an 80-SM GPU -- measured at 0.0125 blocks/SM and 39% of the step.
        # Splitting the key axis gives every block a disjoint slice, so total
        # KV traffic is unchanged (unlike splitting by head, which re-reads it).
        pacc, pmax, psum = _split_workspace_for(
            num_tokens, num_heads, splits, q.device
        )
        _sm70_sparse_paged_fp8_split_kernel[(num_tokens, head_blocks, splits)](
            q,
            main_cache,
            main_indices_2d,
            main_lengths,
            extra_cache,
            extra_indices_2d,
            extra_lengths_1d,
            pacc,
            pmax,
            psum,
            q.stride(0),
            q.stride(1),
            pacc.stride(0),
            pacc.stride(1),
            pacc.stride(2),
            pmax.stride(0),
            pmax.stride(1),
            main_cache.stride(0),
            extra_cache.stride(0),
            main_indices_2d.stride(0),
            extra_indices_2d.stride(0),
            main_cache.shape[0] * main_cache.shape[1],
            extra_cache.shape[0] * extra_cache.shape[1],
            main_cache.shape[1],
            extra_cache.shape[1],
            float(scale),
            num_heads,
            HAS_EXTRA=has_extra,
            MAIN_WIDTH=main_width,
            BLOCK_H=block_h,
            BLOCK_K=16,
            NOPE_DIM=_NOPE_DIM,
            NOPE_BLOCK=_HEAD_DIM,
            ROPE_DIM=_ROPE_DIM,
            NUM_SPLITS=splits,
            num_warps=4,
        )
        _sm70_sparse_split_combine_kernel[(num_tokens, num_heads)](
            pacc,
            pmax,
            psum,
            attn_sink.contiguous(),
            out,
            pacc.stride(0),
            pacc.stride(1),
            pacc.stride(2),
            pmax.stride(0),
            pmax.stride(1),
            out.stride(0),
            out.stride(1),
            HEAD_BLOCK=_HEAD_DIM,
            NUM_SPLITS=splits,
            num_warps=4,
        )
        return

    _sm70_sparse_paged_fp8_kernel[(num_tokens, head_blocks)](
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
