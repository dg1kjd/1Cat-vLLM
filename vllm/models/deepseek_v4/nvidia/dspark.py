# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark block-parallel draft model for DeepSeek-V4-Flash.

The checkpoint ships DSpark, not a V3-style MTP: three full decoder stages
(``mtp.0/1/2``) that consume a *block* of ``dspark_block_size`` token slots in
one forward pass and emit that many draft tokens, rather than one autoregressive
step per spec token. See ``DSPARK_ARCHITECTURE.md`` and the reference at
``DeepSeek-V4-Flash-0731/inference/model.py:743-880``.

Shape of one drafting step:

    main_x = main_norm(main_proj(cat(h_40, h_41, h_42)))   # stage 0 only
    ids    = [accepted_token, NOISE, NOISE, NOISE, NOISE]
    x      = embed(ids)                                    # embeddings only
    for stage in mtp.0, mtp.1, mtp.2:  x = stage(x, main_x)

``main_x`` never enters the hidden stream; it reaches the stages *only* as
attention keys/values. That is what lets this ride on vLLM's DFlash proposer,
which precomputes context K/V from target hidden states and then runs the query
block through the draft model with ``input_ids`` alone.

Each stage's attention is sliding-window only (``compress_ratios`` is 0 for the
three DSpark entries, and ``DeepseekV4MultiHeadLatentAttentionWrapper`` forces
``compress_ratio = 1`` for ``layer_id >= num_hidden_layers``), so there is no
Lightning Indexer and no compressor: the drafter costs the same at 32k context
as at 1k.
"""

import typing
from collections.abc import Callable, Iterable

import regex as re
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform

from .model import (
    DeepseekV4DecoderLayer,
    make_deepseek_v4_expert_params_mapping,
)

logger = init_logger(__name__)

# Mirrors the suffix logic in the main V4 loader: MXFP4 experts register
# ``w{1,2,3}_weight_scale`` while FP8 block-quant experts register
# ``..._weight_scale_inv``. Everything else (including shared experts and the
# fp8 linears) always uses ``.weight_scale_inv``.
_EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.w[123]\.scale$")


class DSparkMarkovHead(nn.Module):
    """Bigram refinement applied sequentially across the draft block.

    ``markov_w1`` embeds the previous token to ``dspark_markov_rank`` and
    ``markov_w2`` expands that back to a full-vocab logit bias. Cheap relative
    to the block forward, and it is what makes slot i+1 conditional on the token
    actually sampled at slot i even though the block itself is non-causal.
    """

    def __init__(self, vocab_size: int, rank: int, prefix: str = "") -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(
            vocab_size, rank, prefix=maybe_prefix(prefix, "markov_w1")
        )
        self.markov_w2 = ParallelLMHead(
            vocab_size, rank, prefix=maybe_prefix(prefix, "markov_w2")
        )

    def forward(
        self, token_ids: torch.Tensor, logits_processor: LogitsProcessor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.markov_w1(token_ids)
        bias = logits_processor(self.markov_w2, embed)
        return bias, embed


class DSparkConfidenceHead(nn.Module):
    """Per-slot fp32 confidence score.

    The checkpoint stores ``proj`` in bf16; the reference keeps the parameter in
    fp32 and casts the input, so we do the same — this is a scalar per slot and
    the fp32 cost is irrelevant.

    NOTE: this score is used only to *truncate* a proposal (drop trailing slots
    we do not believe), never to accept a token. Acceptance goes through the
    standard rejection sampler so the output distribution is preserved.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(1, input_dim, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self, hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([hidden, markov_embed], dim=-1).float()
        return torch.nn.functional.linear(x, self.weight).squeeze(-1)


class DeepSeekV4DSparkModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.num_stages = getattr(config, "dspark_num_stages", 3)
        self.first_layer_idx = config.num_hidden_layers

        # Reserved indexer buffer, as in DeepseekV4Model. DSpark stages are
        # SWA-only so no indexer actually fires, but DeepseekV4DecoderLayer
        # requires the argument.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # Keyed by absolute layer index so extract_layer_index() in the
        # attention wrapper sees layer_id >= num_hidden_layers and selects the
        # SWA-only route.
        self.layers = torch.nn.ModuleDict(
            {
                str(self.first_layer_idx + i): DeepseekV4DecoderLayer(
                    vllm_config,
                    prefix=f"{prefix}.layers.{self.first_layer_idx + i}",
                    topk_indices_buffer=self.topk_indices_buffer,
                    aux_stream_list=aux_stream_list,
                )
                for i in range(self.num_stages)
            }
        )
        self.stage_list = [
            self.layers[str(self.first_layer_idx + i)]
            for i in range(self.num_stages)
        ]

        # Stage 0 only: fuse the three captured target layers into main_x.
        self.main_proj = ReplicatedLinear(
            self.hidden_size * len(self.target_layer_ids),
            self.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.main_proj",
        )
        self.main_norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        # Last stage only: the output head stack.
        self.norm = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.hc_head_fn = nn.Parameter(
            torch.empty(self.hc_mult, self.hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(self.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )
        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            config.dspark_markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
        )
        self.confidence_head = DSparkConfidenceHead(
            self.hidden_size + config.dspark_markov_rank
        )

        # DSpark shares embed/head with the target in the reference; vLLM loads
        # the draft model separately, so we keep our own copies (~265 MB/rank
        # at TP8 for the pair). Same tradeoff the existing MTP path makes.
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            self.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.head = ParallelLMHead(
            config.vocab_size,
            self.hidden_size,
            prefix=maybe_prefix(prefix, "head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        self._use_sm70_path = current_platform.is_cuda() and (
            current_platform.is_device_capability((7, 0))
        )

    # ---------------------------------------------------------------- inputs

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Project target hidden states to per-stage K/V and page them in.

        ``context_states`` is the concatenation of the captured target layers,
        (num_context, hidden * len(target_layer_ids)). This is the only route by
        which the target's state reaches the drafter.

        With ``context_slot_mapping`` None (dummy_run) the projections still run
        so shapes and timings are representative, but nothing is written.
        """
        main_x = self.main_norm(self.main_proj(context_states))

        for stage in self.stage_list:
            # DeepseekV4Attention holds the projections but delegates to the
            # wrapper, which is where the KV-only helpers live.
            mla = stage.attn.mla_attn
            kv = mla.kv_from_hidden(main_x)
            slot_mapping = (
                None
                if context_slot_mapping is None
                else context_slot_mapping.get(mla.swa_cache_layer.prefix)
            )
            mla.insert_kv_into_swa_cache(kv, context_positions, slot_mapping)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        # Same mHC chain as the main stack: the first stage broadcasts the 2D
        # embedding into hc_mult streams (matching the reference's
        # `.unsqueeze(2).repeat(...)` in forward_embed), then each stage fuses
        # its post-mix with the next stage's pre-mix.
        residual, post_mix, res_mix = None, None, None
        for stage in self.stage_list:
            x, residual, post_mix, res_mix = stage(
                x, positions, None, post_mix, res_mix, residual
            )
        hidden_states = mhc_post_tilelang(x, residual, post_mix, res_mix)
        # Flat pre-hc_head residual; hc_head is deferred to compute_logits so
        # the proposer can hold on to the hidden state for the confidence head.
        return hidden_states.flatten(1)

    def _dense_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(T, hc_dim) pre-hc_head residual -> (T, hidden) dense state."""
        hidden_states = hidden_states.view(-1, self.hc_mult, self.hidden_size)
        return hc_head_fused_kernel_tilelang(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.head, self.norm(self._dense_hidden(hidden_states)))

    def compute_logits_and_dense(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the dense hidden state the confidence head needs."""
        dense = self._dense_hidden(hidden_states)
        return self.logits_processor(self.head, self.norm(dense)), dense

    def markov_bias(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.markov_head(token_ids, self.logits_processor)

    def confidence(
        self, dense_hidden: torch.Tensor, markov_embed: torch.Tensor
    ) -> torch.Tensor:
        return self.confidence_head(dense_hidden, markov_embed)

    # --------------------------------------------------------------- weights

    def finalize_weights(self) -> None:
        """Derived tensors that must exist before the first forward."""
        for stage in self.stage_list:
            stage.ffn.finalize_mega_moe_weights()
        # Stage 0 receives a 2D embedding, so it takes the broadcast mHC-pre
        # route and needs the summed fn variant (cf.
        # DeepseekV4Model.finalize_mhc_broadcast_weights).
        first = self.stage_list[0]
        first.hc_attn_fn_broadcast = (
            first.hc_attn_fn.detach()
            .view(-1, first.hc_mult, first.hidden_size)
            .sum(dim=1)
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Dotted shard names on purpose: a bare "w1" also matches the markov
        # head's `markov_w1`, which would rewrite it to `markov_gate_up_proj`
        # and fail the parameter lookup.
        stacked_params_mapping = [
            (".gate_up_proj", ".w1", 0),
            (".gate_up_proj", ".w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        first_stage = self.stage_list[0]
        if first_stage.ffn.use_mega_moe:
            expert_mapping = make_deepseek_v4_expert_params_mapping(
                self.config.n_routed_experts
            )
        else:
            expert_mapping = FusedMoE.make_expert_params_mapping(
                self,
                ckpt_gate_proj_name="w1",
                ckpt_down_proj_name="w2",
                ckpt_up_proj_name="w3",
                num_experts=self.config.n_routed_experts,
            )

        expert_scale_suffix = (
            ".weight_scale"
            if getattr(self.config, "expert_dtype", "fp4") == "fp4"
            else ".weight_scale_inv"
        )

        for name, loaded_weight in weights:
            name = self._rewrite_checkpoint_name(name)
            if name is None:
                continue

            if name.endswith(".scale"):
                suffix = (
                    expert_scale_suffix
                    if _EXPERT_SCALE_RE.search(name)
                    else ".weight_scale_inv"
                )
                name = name.removesuffix(".scale") + suffix

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if ".experts." in name or weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales must be reinterpreted, not converted: a
                    # numeric copy_() would zero the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        if weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        ):
                            loaded_params.add(name_mapped)
                            break
                    continue
                if "attn_sink" in name:
                    narrow = loaded_weight[head_rank_start:head_rank_end]
                    params_dict[name][: narrow.shape[0]].copy_(narrow)
                    loaded_params.add(name)
                    continue
                if name == "confidence_head.weight":
                    params_dict[name].copy_(loaded_weight.float())
                    loaded_params.add(name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        self._check_all_stages_loaded(loaded_params)
        self.finalize_weights()
        logger.info("DSpark draft model loaded: %d params", len(loaded_params))
        return loaded_params

    def _rewrite_checkpoint_name(self, name: str) -> str | None:
        """Reference-format checkpoint name -> this module's parameter name.

        The DSV4 checkpoint is in the reference layout (``mtp.0.attn.wkv``),
        not HF layout. Everything outside ``mtp.*`` is dropped except the
        embedding and output head, which DSpark shares with the target.
        """
        if name in ("embed.weight", "model.embed_tokens.weight"):
            return "embed_tokens.weight"
        if name in ("head.weight", "lm_head.weight"):
            return "head.weight"
        if not name.startswith("mtp."):
            return None

        stage_str, rest = name[4:].split(".", 1)
        stage = int(stage_str)
        if stage >= self.num_stages:
            return None
        layer_key = str(self.first_layer_idx + stage)

        # Stage-scoped extras live at the top of this module, not inside the
        # decoder layer: main_proj/main_norm on stage 0, the head stack on the
        # last stage.
        for shared in (
            "main_proj",
            "main_norm",
            "markov_head",
            "confidence_head",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
        ):
            if rest.startswith(shared):
                if shared == "confidence_head":
                    return "confidence_head.weight"
                return rest
        if rest.startswith("norm."):
            return rest

        if rest.endswith(".ffn.gate.bias") or rest == "ffn.gate.bias":
            rest = rest.replace("ffn.gate.bias", "ffn.gate.e_score_correction_bias")
        if ".shared_experts.w2" in rest:
            rest = rest.replace(".shared_experts.w2", ".shared_experts.down_proj")
        return f"layers.{layer_key}.{rest}"

    def _check_all_stages_loaded(self, loaded_params: set[str]) -> None:
        for i in range(self.num_stages):
            prefix = f"layers.{self.first_layer_idx + i}."
            if not any(p.startswith(prefix) for p in loaded_params):
                raise ValueError(
                    f"DSpark stage {i} (mtp.{i}.*) is missing from the "
                    f"checkpoint. Disable speculative decoding or use a "
                    f"checkpoint that ships the DSpark weights."
                )


class DeepSeekV4DSpark(nn.Module):
    """Thin wrapper matching the draft-model interface vLLM's proposer uses."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DeepSeekV4DSparkModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    @property
    def block_size(self) -> int:
        return self.model.block_size

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def precompute_and_store_context_kv(self, *args, **kwargs) -> None:
        self.model.precompute_and_store_context_kv(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.model.load_weights(weights)
