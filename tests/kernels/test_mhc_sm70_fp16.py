# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.kernels.mhc import tilelang as mhc_tilelang
from vllm.platforms.interface import DeviceCapability
from vllm.utils.import_utils import has_tilelang


class _FakeCudaPlatform:
    def __init__(self, supports_bf16: bool) -> None:
        self.supports_bf16 = supports_bf16

    def is_cuda(self) -> bool:
        return True

    def get_device_capability(self) -> DeviceCapability:
        return DeviceCapability(8 if self.supports_bf16 else 7, 0)


def test_mhc_sm70_requires_explicit_fp16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(False))

    fp16 = torch.empty((1,), dtype=torch.float16)
    assert mhc_tilelang._require_mhc_activation_dtype(fp16) == torch.float16

    with pytest.raises(RuntimeError, match="SM70 has no native BF16"):
        mhc_tilelang._require_mhc_activation_dtype(
            torch.empty((1,), dtype=torch.bfloat16)
        )


def test_mhc_bf16_is_preserved_on_supported_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(True))

    bf16 = torch.empty((1,), dtype=torch.bfloat16)
    assert mhc_tilelang._require_mhc_activation_dtype(bf16) == torch.bfloat16


def test_mhc_fp16_rejects_implicit_norm_weight_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mhc_tilelang, "current_platform", _FakeCudaPlatform(False))

    with pytest.raises(ValueError, match="FP16 requires norm_weight"):
        mhc_tilelang._prepare_mhc_norm_weight(
            torch.empty((1,), dtype=torch.bfloat16), torch.float16
        )


def test_mhc_native_bf16_preserves_norm_weight_conversion() -> None:
    prepared = mhc_tilelang._prepare_mhc_norm_weight(
        torch.empty((1,), dtype=torch.float32), torch.bfloat16
    )
    assert prepared is not None
    assert prepared.dtype == torch.bfloat16


def test_mhc_fake_paths_preserve_fp16_graph_metadata() -> None:
    residual = torch.empty((2, 4, 128), dtype=torch.float16, device="meta")
    x = torch.empty((2, 128), dtype=torch.float16, device="meta")
    fn = torch.empty((24, 512), dtype=torch.float32, device="meta")
    scale = torch.empty((3,), dtype=torch.float32, device="meta")
    base = torch.empty((24,), dtype=torch.float32, device="meta")
    post = torch.empty((2, 4, 1), dtype=torch.float32, device="meta")
    comb = torch.empty((2, 4, 4), dtype=torch.float32, device="meta")

    pre = torch.ops.vllm.mhc_pre_tilelang(
        residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
    )
    fused = torch.ops.vllm.mhc_fused_post_pre_tilelang(
        x, residual, post, comb, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
    )
    head = torch.ops.vllm.hc_head_fused_kernel_tilelang(
        residual,
        torch.empty((4, 512), dtype=torch.float32, device="meta"),
        torch.empty((1,), dtype=torch.float32, device="meta"),
        torch.empty((4,), dtype=torch.float32, device="meta"),
        1e-6,
        1e-6,
    )

    assert pre[2].dtype == torch.float16
    assert pre[2].shape == (2, 128)
    assert fused[0].dtype == torch.float16
    assert fused[3].dtype == torch.float16
    assert fused[3].shape == (2, 128)
    assert head.dtype == torch.float16
    assert head.shape == (2, 128)


def test_mhc_fp16_fake_path_compiles_fullgraph() -> None:
    def run(
        residual: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.mhc_pre_tilelang(
            residual, fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 2
        )

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    output = compiled(
        torch.empty((2, 4, 128), dtype=torch.float16, device="meta"),
        torch.empty((24, 512), dtype=torch.float32, device="meta"),
        torch.empty((3,), dtype=torch.float32, device="meta"),
        torch.empty((24,), dtype=torch.float32, device="meta"),
    )

    assert output[2].dtype == torch.float16
    assert output[2].shape == (2, 128)


@pytest.mark.skipif(
    not (mhc_tilelang.current_platform.is_cuda() and has_tilelang()),
    reason="CUDA and TileLang required",
)
def test_mhc_sm70_fp16_block_m_prenorm_keeps_rows_independent() -> None:
    """Exercise the >=1024-token block-M path with non-identical row pairs."""
    hidden_size = 128
    hc_mult = 4
    num_tokens = 1024
    hc_hidden_size = hc_mult * hidden_size
    n_out = 2 * hc_mult + hc_mult * hc_mult
    device = mhc_tilelang.current_platform.device_type

    # The prior scalar reduction could carry the first row into the second.
    # Exact integer sums make that failure unambiguous: 512/1536, not 512/2048.
    x = torch.ones((num_tokens, hc_hidden_size), dtype=torch.float16, device=device)
    x[1::2].fill_(3)
    fn = torch.ones((n_out, hc_hidden_size), dtype=torch.float32, device=device)
    out = torch.empty((1, num_tokens, n_out), dtype=torch.float32, device=device)
    sqrsum = torch.empty((1, num_tokens), dtype=torch.float32, device=device)
    out_ref = torch.empty_like(out)
    sqrsum_ref = torch.empty_like(sqrsum)

    mhc_tilelang._torch_hc_prenorm_gemm(x, fn, out_ref, sqrsum_ref)
    mhc_tilelang._tilelang_hc_prenorm_gemm(x, fn, out, sqrsum, hidden_size, hc_mult)

    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(sqrsum, sqrsum_ref, rtol=0, atol=0)
