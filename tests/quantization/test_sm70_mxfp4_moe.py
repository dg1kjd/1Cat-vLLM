# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.mxfp4 import (
    make_deepseek_v4_mxfp4_moe_method,
)
from vllm.model_executor.layers.quantization.mxfp4_sm70_moe import (
    Mxfp4SM70MoEMethod,
    validate_mxfp4_sm70_moe_contract,
    validate_mxfp4_sm70_moe_weight_layout,
)


def _v4_flash_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        num_experts=256,
        experts_per_token=6,
        hidden_dim=4096,
        intermediate_size_per_partition=256,
        num_local_experts=256,
        num_logical_experts=256,
        activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.DeepseekV4,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=8,
            pcp_size=1,
            dp_size=1,
            ep_size=1,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=False,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        in_dtype=torch.float16,
        swiglu_limit=7.0,
    )


@pytest.mark.parametrize(
    ("tp_size", "intermediate_size_per_partition"),
    [(8, 256), (4, 512)],
)
def test_mxfp4_sm70_contract_accepts_v4_flash_tp_shapes(
    tp_size: int, intermediate_size_per_partition: int
):
    validate_mxfp4_sm70_moe_contract(
        global_num_experts=256,
        top_k=6,
        hidden_size=4096,
        intermediate_size_per_partition=intermediate_size_per_partition,
        tp_size=tp_size,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"global_num_experts": 128}, "256 global experts"),
        ({"top_k": 8}, "top-k=6"),
        ({"hidden_size": 8192}, "hidden size 4096"),
        ({"intermediate_size_per_partition": 512}, "intermediate size 2048"),
    ],
)
def test_mxfp4_sm70_contract_rejects_non_v4_flash_shapes(kwargs, match):
    values = {
        "global_num_experts": 256,
        "top_k": 6,
        "hidden_size": 4096,
        "intermediate_size_per_partition": 256,
        "tp_size": 8,
    }
    values.update(kwargs)
    with pytest.raises(NotImplementedError, match=match):
        validate_mxfp4_sm70_moe_contract(**values)


@pytest.mark.parametrize("intermediate_size", [256, 512])
def test_mxfp4_sm70_weight_layout_accepts_packed_v4_flash_tp_tensors(
    intermediate_size: int,
):
    local_experts = 256
    hidden_size = 4096
    validate_mxfp4_sm70_moe_weight_layout(
        local_num_experts=local_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size,
        w13_weight=torch.empty(
            local_experts,
            2 * intermediate_size,
            hidden_size // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        w13_weight_scale=torch.empty(
            local_experts,
            2 * intermediate_size,
            hidden_size // 32,
            dtype=torch.uint8,
            device="meta",
        ),
        w2_weight=torch.empty(
            local_experts,
            hidden_size,
            intermediate_size // 2,
            dtype=torch.uint8,
            device="meta",
        ),
        w2_weight_scale=torch.empty(
            local_experts,
            hidden_size,
            intermediate_size // 32,
            dtype=torch.uint8,
            device="meta",
        ),
    )


def test_mxfp4_sm70_weight_layout_rejects_wrong_ue8m0_scale_shape():
    with pytest.raises(ValueError, match="w2_weight_scale"):
        validate_mxfp4_sm70_moe_weight_layout(
            local_num_experts=1,
            hidden_size=4096,
            intermediate_size_per_partition=256,
            w13_weight=torch.empty(1, 512, 2048, dtype=torch.uint8, device="meta"),
            w13_weight_scale=torch.empty(1, 512, 128, dtype=torch.uint8, device="meta"),
            w2_weight=torch.empty(1, 4096, 128, dtype=torch.uint8, device="meta"),
            w2_weight_scale=torch.empty(1, 4096, 7, dtype=torch.uint8, device="meta"),
        )


def test_mxfp4_sm70_platform_gate_is_exact(monkeypatch):
    monkeypatch.setattr(sm70_tm.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sm70_tm.current_platform,
        "is_device_capability",
        lambda capability: capability == (7, 0),
    )
    assert sm70_tm.is_exact_sm70_cuda_platform()

    monkeypatch.setattr(
        sm70_tm.current_platform,
        "is_device_capability",
        lambda capability: False,
    )
    assert not sm70_tm.is_exact_sm70_cuda_platform()


def test_mxfp4_sm70_factory_selects_native_route(monkeypatch):
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(sm70_tm, "should_use_mxfp4_moe_turbomind", lambda: True)

    method = make_deepseek_v4_mxfp4_moe_method(_v4_flash_moe_config())

    assert isinstance(method, Mxfp4SM70MoEMethod)
    assert not method.skip_forward_padding


def test_mxfp4_sm70_factory_rejects_marlin_or_emulation(monkeypatch):
    monkeypatch.setattr(sm70_tm, "is_exact_sm70_cuda_platform", lambda: True)
    monkeypatch.setattr(sm70_tm, "should_use_mxfp4_moe_turbomind", lambda: False)

    with pytest.raises(NotImplementedError, match="Marlin"):
        make_deepseek_v4_mxfp4_moe_method(_v4_flash_moe_config())
