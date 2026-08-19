# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark proposer: DFlash's block-parallel machinery plus the markov head.

The DSpark draft block is non-causal — every slot attends to every other one —
so a single forward gives all `block_size` slots at once, but nothing in that
forward tells slot i+1 which token was actually chosen at slot i. The markov
head is what closes that loop: a bigram bias, applied sequentially over the
block, that conditions each slot's logits on the token sampled just before it.

Without it the block degenerates into independent guesses. Measured on
DSV4-Flash before this was wired: per-position acceptance 0.222, 0.044, 0.000,
0.000 (6.7% overall) — only the first slot, which sees the real token directly
through the block's attention, carried any signal at all.

The other thing this class fixes is a slot off-by-one. DFlash and DSpark build
the same query block — slot 0 holds the bonus token, slots 1..N hold the
noise/mask token — but they read it differently:

  * DFlash is mask-style: the mask at position p *is* the token at position p,
    so slot j yields draft j and slot 0's output is redundant with the target's.
  * DSpark is next-token at every slot. The reference walks
    ``output_ids[:, i + 1] = sample(logits[:, i])`` with
    ``output_ids[:, 0] = input_ids``, and slot i carries rope position
    ``start_pos + 1 + i`` — so slot i predicts the token at position
    ``start_pos + 2 + i``, i.e. draft i+1. The first draft comes from slot 0.

So we shift `token_indices_to_sample` left by one and let the final slot fall
off the end. The block stays `block_size` wide (in-distribution); we just read
slots 0..N-1 instead of 1..N.

Reference: `DSparkBlock.forward_head`, inference/model.py:865-880;
`DSparkAttention.forward`, inference/model.py:750-792 (rope positions).
"""

import os

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)

# The markov walk costs one full-vocab logits gather per slot. Set
# VLLM_DSPARK_MARKOV=0 to fall back to DFlash's independent per-slot sampling
# (which can use the local-argmax reduction and skips the gather entirely) to
# see what the walk is actually buying.
_MARKOV_ENABLED = os.getenv("VLLM_DSPARK_MARKOV", "1") == "1"
_PROFILE = os.getenv("VLLM_DSPARK_PROFILE", "0") == "1"
_PROFILE_EVERY = int(os.getenv("VLLM_DSPARK_PROFILE_EVERY", "200"))


class DSparkProposer(DFlashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)
        # Seeds the markov chain: the token the target actually committed, which
        # is slot 0 of the reference's `output_ids`.
        self._bonus_token_ids: torch.Tensor | None = None
        self._prof_calls = 0
        self._prof_logits_ms = 0.0
        self._prof_markov_ms = 0.0

        self._prof_propose_ms = 0.0

    def _record(self, logits_ms: float, markov_ms: float) -> None:
        self._prof_calls += 1
        self._prof_logits_ms += logits_ms
        self._prof_markov_ms += markov_ms

    @override
    def propose(self, *args, **kwargs):
        if not _PROFILE:
            return super().propose(*args, **kwargs)
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        out = super().propose(*args, **kwargs)
        ev_end.record()
        ev_end.synchronize()
        self._prof_propose_ms += ev_start.elapsed_time(ev_end)
        n = self._prof_calls
        if n and n % _PROFILE_EVERY == 0:
            logger.info(
                "DSpark profile over %d calls: propose %.2f ms, of which "
                "logits %.2f ms and markov walk %.2f ms",
                n,
                self._prof_propose_ms / n,
                self._prof_logits_ms / n,
                self._prof_markov_ms / n,
            )
            self._prof_calls = 0
            self._prof_propose_ms = 0.0
            self._prof_logits_ms = 0.0
            self._prof_markov_ms = 0.0
        return out

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        self._bonus_token_ids = next_token_ids
        num_query_total, sample_indices, new_cad = super().set_inputs_first_pass(
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        )
        # DFlash's kernel points at query offsets 1..N; DSpark reads 0..N-1.
        # The buffer is freshly allocated per call, so this is safe in place.
        sample_indices -= 1
        return num_query_total, sample_indices, new_cad

    @override
    def _sample_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logits: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bonus_token_ids = self._bonus_token_ids
        self._bonus_token_ids = None

        if not _MARKOV_ENABLED:
            return super()._sample_draft_tokens(
                hidden_states, sampling_metadata, logits, spec_step_idx
            )

        if _PROFILE:
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_logits = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()

        num_spec = self.num_speculative_tokens
        if logits is None:
            logits = self._compute_logits_for_step(hidden_states, spec_step_idx)
        total_rows = logits.shape[0]
        batch_size, remainder = divmod(total_rows, num_spec)

        if _PROFILE:
            ev_logits.record()

        if (
            bonus_token_ids is None
            or remainder != 0
            or bonus_token_ids.shape[0] < batch_size
        ):
            # Dummy/profile run, or a shape we cannot walk: fall back to the
            # independent per-slot sampling DFlash does.
            return super()._sample_draft_tokens(
                hidden_states, sampling_metadata, logits, spec_step_idx
            )

        # Rows are laid out request-major, slot-minor (see
        # `copy_and_expand_dflash_inputs_kernel`: sample_out_idx =
        # req_idx * num_spec + slot), so dim 1 is the axis the chain walks.
        logits = logits.view(batch_size, num_spec, -1)
        greedy = (
            sampling_metadata.all_greedy or not self._enable_probabilistic_draft_probs
        )

        prev_token = bonus_token_ids[:batch_size].to(torch.long)
        draft_tokens = []
        draft_probs = [] if not greedy else None

        for slot in range(num_spec):
            bias, _ = self.model.model.markov_bias(prev_token)
            slot_logits = logits[:, slot] + bias
            if greedy:
                prev_token = slot_logits.argmax(dim=-1)
            else:
                token, probs = self._sample_from_logits(
                    slot_logits, sampling_metadata
                )
                prev_token = token.view(-1)
                draft_probs.append(probs)
            draft_tokens.append(prev_token)

        draft_token_ids = torch.stack(draft_tokens, dim=1).view(-1)
        if _PROFILE:
            ev_end.record()
            ev_end.synchronize()
            self._record(
                ev_start.elapsed_time(ev_logits), ev_logits.elapsed_time(ev_end)
            )
        if greedy:
            return draft_token_ids, None
        return draft_token_ids, torch.stack(draft_probs, dim=1).view(
            batch_size * num_spec, -1
        )
