# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

# Diagnostic: "lo:hi", a half-open band of *compressed* key indices (the units
# the indexer scores in, i.e. token_index // compress_ratio). For the last
# query row of every prefill chunk this logs where that band ranks among all
# causally valid keys, and how many of its entries survived the top-k. A
# needle-in-a-haystack miss is otherwise unattributable: it cannot distinguish
# "the indexer never selected the needle" from "it selected it and the model
# still could not read it". Off unless set; the check costs a device sync.
_INDEXER_PROBE = os.getenv("VLLM_INDEXER_PROBE", "")

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024

# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _is_exact_sm70_cuda() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability((7, 0))


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def _log_indexer_probe_rank(prefix, logits, topk_indices, cu_seqlen_ks, cu_seqlen_ke):
    """Report where each `_INDEXER_PROBE` band ranks for the chunk's last row."""
    ks = int(cu_seqlen_ks[-1].item())
    ke = int(cu_seqlen_ke[-1].item())
    row = logits[-1]
    selected = topk_indices[-1]
    for spec in _INDEXER_PROBE.split(","):
        lo, hi = (int(v) for v in spec.split(":"))
        band_lo, band_hi = max(lo, ks), min(hi, ke)
        if band_hi <= band_lo:
            continue
        best = row[band_lo:band_hi].max()
        # Rank among causally valid keys only; everything outside [ks, ke) is
        # masked by the top-k kernel and must not count as a competitor.
        rank = int((row[ks:ke] > best).sum().item())
        kept = int(((selected >= lo) & (selected < hi)).sum().item())
        logger.info(
            "indexer-probe %s keys=%d band=[%d,%d) best_rank=%d kept=%d",
            prefix,
            ke - ks,
            lo,
            hi,
            rank,
            kept,
        )


def _log_indexer_probe_decode(prefix, logits, topk_indices, seq_lens):
    """`_log_indexer_probe_rank` for the decode path.

    Decode selects through a different kernel than prefill (`persistent_topk`
    / `top_k_per_row_decode` rather than `top_k_per_row_prefill`) and a
    different index mapping, so a prefill-only probe says nothing about the
    tokens the model actually generates. Requires --enforce-eager: the syncs
    here cannot run inside a captured decode graph.
    """
    lens = seq_lens.reshape(-1)
    num_keys = int(lens[0].item())
    row = logits[0]
    selected = topk_indices[0]
    for spec in _INDEXER_PROBE.split(","):
        lo, hi = (int(v) for v in spec.split(":"))
        band_hi = min(hi, num_keys)
        if band_hi <= lo:
            continue
        best = row[lo:band_hi].max()
        rank = int((row[:num_keys] > best).sum().item())
        kept = int(((selected >= lo) & (selected < hi)).sum().item())
        logger.info(
            "indexer-probe-decode %s keys=%d band=[%d,%d) best_rank=%d kept=%d",
            prefix,
            num_keys,
            lo,
            hi,
            rank,
            kept,
        )


def _verify_sm70_decode_logits(prefix, logits, q, weights, kv_cache, block_table, seq_lens):
    """In-situ parity of the SM70 decode logits against a torch reference.

    Isolated kernel tests pass on synthetic tensors and still miss layout,
    stride and block-table mistakes that only exist in the live engine, so
    this re-derives row 0's scores straight from the paged cache with plain
    torch ops and compares both the values and the induced top-512 set.
    """
    lens = seq_lens.reshape(-1)
    n = int(lens[0].item())
    if n < 2:
        return
    bs = kv_cache.shape[1]
    idx = torch.arange(n, device=kv_cache.device)
    # Block-major addressing, mirroring indexer_k_quant_and_cache: values for
    # every token of the block, then that block's FP32 scales.
    base = block_table[0][idx // bs].long() * kv_cache.stride(0)
    pos = (idx % bs).long()
    # NOT reshape(-1): MLAAttentionSpec pads the indexer page to 576 B
    # (attention.py, alignment=576), so for block_size=64 the page is 8640 B
    # against 64*132=8448 B of payload and the tensor is non-contiguous.
    # reshape(-1) would quietly return a *compacted copy* while the offsets
    # below still use the real 8640 B stride, drifting 192 B per block --
    # which reads FP8 value bytes as the FP32 scale and manufactures negative
    # ~1e36 scales and phantom e4m3 NaNs. as_strided keeps the padding.
    flat = kv_cache.as_strided((kv_cache.shape[0] * kv_cache.stride(0),), (1,))
    lane = torch.arange(128, device=kv_cache.device)
    tokens = flat[(base + pos * 128)[:, None] + lane[None, :]]
    k = tokens.contiguous().view(torch.float8_e4m3fn).float()
    k_scale = (
        flat[(base + bs * 128 + pos * 4)[:, None] + lane[None, :4]]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    qf = q[0].float()
    wf = weights[0].float()
    dot = qf @ k.t()
    ref = ((dot * k_scale).relu() * wf[:, None]).sum(dim=0)
    # The kernel hoists the per-key scale out past the relu, which is only
    # valid for a positive scale. Scoring it both ways separates a scale-sign
    # problem from an arithmetic one.
    ref_hoisted = (dot.relu() * wf[:, None]).sum(dim=0) * k_scale

    # The compressor writes a ue8m0 scale (`scale_val = tl.exp2(exponent)`), so
    # every scale of a slot that was actually written is an exact power of two
    # with a zero fp32 mantissa. Any other bit pattern means we are reading a
    # slot the compressor never wrote — which is a different failure from a
    # quantiser that overflows, and the two are easy to confuse because both
    # surface as e4m3 NaN bytes.
    sbits = k_scale.view(torch.int32)
    written = (sbits & 0x007FFFFF == 0) & (k_scale > 0) & torch.isfinite(k_scale)
    nan_byte = ((tokens == 0x7F) | (tokens == 0xFF)).any(dim=1)
    zero_row = (tokens == 0).all(dim=1)
    # If unwritten slots are the cause, "carries a NaN byte" and "scale is not
    # a power of two" should pick out the same keys.
    agree = int((nan_byte == ~written).sum().item())
    bad = (~written).nonzero().flatten()
    if bad.numel():
        span = f"{int(bad[0])}..{int(bad[-1])}"
        # Unwritten slots from a paging or boundary mistake cluster by position
        # in the block; a quantiser bug would not.
        resid = torch.bincount(bad % bs, minlength=bs)
        hot = int(resid.argmax().item())
        shape = f"span={span} distinct_resid={int((resid > 0).sum())}/{bs} hot_resid={hot}x{int(resid[hot])}"
    else:
        shape = "span=- "
    logger.info(
        "indexer-cache %s keys=%d shape=%s stride0=%d unwritten=%d(%.1f%%) "
        "nanbyte_keys=%d zero_rows=%d nan_vs_unwritten_agree=%.1f%% %s",
        prefix,
        n,
        tuple(kv_cache.shape),
        kv_cache.stride(0),
        int((~written).sum().item()),
        100.0 * float((~written).sum().item()) / n,
        int(nan_byte.sum().item()),
        int(zero_row.sum().item()),
        100.0 * agree / n,
        shape,
    )

    got = logits[0][:n].float()
    finite = torch.isfinite(got) & torch.isfinite(ref)
    denom = ref[finite].abs().max().clamp_min(1e-9)
    diff = (got - ref).abs().where(finite, torch.zeros_like(got))
    err = (diff.max() / denom).item()
    worst = int(diff.argmax().item())
    k_top = min(512, n)
    a = set(got.topk(k_top).indices.tolist())
    b = set(ref.topk(k_top).indices.tolist())
    c = set(ref_hoisted.topk(k_top).indices.tolist())
    raw = tokens
    logger.info(
        "indexer-verify %s keys=%d rel_err=%.3e worst_at=%d(%.1f%%) "
        "nan_got=%d nan_ref=%d big=%d top%d_overlap=%.1f%% "
        "hoisted_overlap=%.1f%% | fp8nan=%d scale_neg=%d scale_bad=%d "
        "scale_rng=[%.3g,%.3g] qnan=%d wnan=%d",
        prefix,
        n,
        err,
        worst,
        100.0 * worst / n,
        int((~torch.isfinite(got)).sum().item()),
        int((~torch.isfinite(ref)).sum().item()),
        int((diff > 0.01 * denom).sum().item()),
        k_top,
        100.0 * len(a & b) / k_top,
        100.0 * len(a & c) / k_top,
        int(((raw == 0x7F) | (raw == 0xFF)).sum().item()),
        int((k_scale < 0).sum().item()),
        int((~torch.isfinite(k_scale)).sum().item()),
        k_scale[torch.isfinite(k_scale)].min().item(),
        k_scale[torch.isfinite(k_scale)].max().item(),
        int((~torch.isfinite(qf)).sum().item()),
        int((~torch.isfinite(wf)).sum().item()),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_fp4_cache,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
    sm70_fp16_indexer = _is_exact_sm70_cuda()

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if sm70_fp16_indexer:
        assert q_quant.dtype == torch.float16
        assert q_scale is None
        assert not use_fp4_cache, "SM70 requires the FP8 indexer cache"
    elif use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks:
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            if sm70_fp16_indexer:
                from vllm.models.deepseek_v4.sm70.indexer import (
                    sm70_indexer_prefill_logits,
                )

                logits = sm70_indexer_prefill_logits(
                    q_slice,
                    k_quant,
                    k_scale,
                    weights[chunk.token_start : chunk.token_end],
                )
            elif current_platform.is_xpu():
                if q_scale_slice is not None:
                    raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                    q_slice_cast,
                    k_quant_cast,
                    k_scale_cast,
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            else:
                logits = fp8_fp4_mqa_logits(
                    (q_slice_cast, q_scale_slice),
                    (k_quant_cast, k_scale_cast),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            ops.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

            if _INDEXER_PROBE:
                _log_indexer_probe_rank(
                    k_cache_prefix,
                    logits,
                    topk_indices,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        raw_kv_cache = kv_cache
        if not sm70_fp16_indexer:
            kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = (
                pack_seq_triton(weights[:num_decode_tokens], decode_lens, pad_value=0)
                if sm70_fp16_indexer
                else None
            )
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = (
                weights[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, weights.shape[-1]
                )
                if sm70_fp16_indexer
                else None
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if sm70_fp16_indexer:
            from vllm.models.deepseek_v4.sm70.indexer import (
                sm70_indexer_decode_logits,
            )

            assert padded_weights is not None
            logits = sm70_indexer_decode_logits(
                padded_q_quant_decode_tokens.reshape(
                    num_padded_tokens,
                    padded_q_quant_decode_tokens.shape[-2],
                    padded_q_quant_decode_tokens.shape[-1],
                ),
                raw_kv_cache,
                padded_weights.reshape(num_padded_tokens, -1),
                seq_lens,
                decode_metadata.block_table,
                attn_metadata_narrowed.max_seq_len,
            )
            if _INDEXER_PROBE:
                _verify_sm70_decode_logits(
                    k_cache_prefix,
                    logits,
                    padded_q_quant_decode_tokens.reshape(
                        num_padded_tokens,
                        padded_q_quant_decode_tokens.shape[-2],
                        padded_q_quant_decode_tokens.shape[-1],
                    ),
                    padded_weights.reshape(num_padded_tokens, -1),
                    raw_kv_cache,
                    decode_metadata.block_table,
                    seq_lens,
                )
        elif current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and topk_tokens in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            ops.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

        if _INDEXER_PROBE:
            _log_indexer_probe_decode(k_cache_prefix, logits, topk_indices, seq_lens)

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if (
            current_platform.is_cuda()
            and not _is_exact_sm70_cuda()
            and not has_deep_gemm()
        ):
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_quant, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path is only supported on AITER. "
            "Please enable aiter with VLLM_ROCM_USE_AITER=1"
        )
