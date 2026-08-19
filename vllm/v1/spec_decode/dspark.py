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

import math
import os

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
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

# Confidence-based truncation. `DSparkBlock.forward_head` (inference/model.py:865)
# emits a per-slot fp32 score that the reference computes and then never reads.
# Set VLLM_DSPARK_CONF_THRESHOLD to a probability in (0, 1) to drop the trailing
# slots we do not believe, so the target verifies a shorter block.
#
# This only ever *shortens* a proposal. Acceptance still runs through vLLM's
# rejection sampler over the tokens that remain, so the output distribution is
# untouched; the trade is draft length against target work. It pays only if the
# dropped slots were going to be rejected anyway — hence VLLM_DSPARK_CONF_STATS,
# which measures exactly that before you pick a threshold.
#
# MEASURED ON SM70 (8xV100, 2026-08-19) AND OFF BY DEFAULT: the head predicts
# well (per-slot acceptance is monotone in the score and roughly calibrated —
# slot 3 ran 92% accepted at p>=0.9 and 0/8 below p<0.3) but truncating loses,
# 37.8 tok/s against 59.1 baseline at threshold 0.9. Two independent reasons,
# both likely to hold on any weight-bound decode:
#   1. A verify row is nearly free. The TP8 MoE decode step is dominated by
#      weight traffic, so 5 rows cost what 3 do; the tokens given up buy
#      nothing back.
#   2. Async scheduling is on, and the trim can only land in _update_states
#      (the scheduler committed a full-width block before the drafter ran).
#      That needs a CPU sync on the previous step's D2H copy at the head of
#      every step, which kills the CPU/GPU overlap: step time went UP,
#      72 -> 83 ms, while verifying FEWER rows.
# Worth revisiting only where rows actually cost something: large batches, or a
# compute-bound backend.
_CONF_THRESHOLD = float(os.getenv("VLLM_DSPARK_CONF_THRESHOLD", "0") or 0.0)
_CONF_ENABLED = 0.0 < _CONF_THRESHOLD < 1.0
# The score is a logit; compare in logit space and skip the sigmoid.
_CONF_LOGIT = (
    math.log(_CONF_THRESHOLD / (1.0 - _CONF_THRESHOLD)) if _CONF_ENABLED else 0.0
)
# Never truncate below this many slots. 0 lets a proposal be dropped entirely
# (the request then runs a plain decode step, which is a legal schedule).
_CONF_MIN_DRAFT = int(os.getenv("VLLM_DSPARK_CONF_MIN_DRAFT", "1"))

# Calibration: bucket each slot's confidence and count how often the target
# actually accepted it. Accumulated on GPU and only synced when logged.
_CONF_STATS = os.getenv("VLLM_DSPARK_CONF_STATS", "0") == "1"
_CONF_STATS_EVERY = int(os.getenv("VLLM_DSPARK_CONF_STATS_EVERY", "500"))
_CONF_BUCKETS = 10


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

        # Per-request draft length after confidence truncation, handed to the
        # runner via take_last_valid_draft_counts(). None means "full width".
        self._valid_draft_counts: torch.Tensor | None = None
        # Calibration state: the previous step's scores and lengths, paired
        # against this step's rejection counts.
        self._prev_conf: torch.Tensor | None = None
        self._prev_valid_counts: torch.Tensor | None = None
        self._conf_hist: torch.Tensor | None = None
        self._conf_hist_accepted: torch.Tensor | None = None
        self._conf_stats_steps = 0

    # ---------------------------------------------------------- confidence

    def take_last_valid_draft_counts(self) -> torch.Tensor | None:
        """Per-request draft length for the proposal just produced.

        int32 [batch] on GPU, or None when the whole block stands. The runner
        slices the draft rows to these lengths before handing them to the
        scheduler.
        """
        counts = self._valid_draft_counts
        self._valid_draft_counts = None
        return counts

    def _truncate_from_confidence(self, confidence: torch.Tensor) -> torch.Tensor:
        """Leading run of believed slots, per request.

        Truncation has to be a prefix: the target verifies the block in order
        and the markov walk conditions slot i+1 on slot i, so a low-confidence
        slot invalidates everything behind it regardless of its own score.
        """
        believed = (confidence >= _CONF_LOGIT).to(torch.int32)
        counts = believed.cumprod(dim=1).sum(dim=1)
        return counts.clamp_(min=_CONF_MIN_DRAFT).to(torch.int32)

    def _record_confidence_stats(
        self, num_rejected_tokens_gpu: torch.Tensor | None
    ) -> None:
        """Pair last step's scores with this step's acceptance.

        `num_rejected_tokens = num_draft_tokens + 1 - valid_sampled_count`
        (spec_decode/utils.py:167), so the leading `drafted - rejected` slots
        were accepted. Everything stays on GPU; the only sync is the log.

        Restricted to batch_size == 1 on purpose: rows here are positional and
        the batch can be reordered between steps, so at np>1 a row could be
        paired with a different request's rejection count. That would corrupt
        the calibration silently, and np=1 is the workload this box is tuned
        for anyway.

        Also assumes the scheduler ran the whole proposal, which holds in
        steady-state decode but not when a draft was trimmed for some other
        reason (chunked prefill, structured output). Those steps skew the
        counts slightly; they are stats, not a correctness input.
        """
        conf, drafted = self._prev_conf, self._prev_valid_counts
        self._prev_conf = None
        self._prev_valid_counts = None
        if conf is None or drafted is None or num_rejected_tokens_gpu is None:
            return
        if conf.shape[0] != 1 or num_rejected_tokens_gpu.shape[0] != 1:
            return

        num_spec = conf.shape[1]
        if self._conf_hist is None:
            self._conf_hist = torch.zeros(
                (num_spec, _CONF_BUCKETS), dtype=torch.int32, device=conf.device
            )
            self._conf_hist_accepted = torch.zeros_like(self._conf_hist)

        drafted = drafted.to(torch.long)
        accepted = (drafted - num_rejected_tokens_gpu.to(torch.long)).clamp_(min=0)
        slot = torch.arange(num_spec, device=conf.device, dtype=torch.long)
        # A slot only counts if it was actually drafted this step.
        counted = slot < drafted
        was_accepted = counted & (slot < accepted)
        bucket = (
            (torch.sigmoid(conf[0].float()) * _CONF_BUCKETS)
            .long()
            .clamp_(0, _CONF_BUCKETS - 1)
        )
        one = torch.ones(1, dtype=torch.int32, device=conf.device)
        self._conf_hist.index_put_(
            (slot[counted], bucket[counted]), one, accumulate=True
        )
        self._conf_hist_accepted.index_put_(
            (slot[was_accepted], bucket[was_accepted]), one, accumulate=True
        )

        self._conf_stats_steps += 1
        if self._conf_stats_steps % _CONF_STATS_EVERY:
            return
        # Every rank computes the same scores from replicated inputs (the
        # markov embedding is all-reduced and the confidence weight is not
        # sharded), so accumulate everywhere but report once.
        if get_tensor_model_parallel_rank() != 0:
            return
        hist = self._conf_hist.tolist()
        ok = self._conf_hist_accepted.tolist()
        for slot_idx, (row, ok_row) in enumerate(zip(hist, ok)):
            total = sum(row)
            if not total:
                continue
            cells = " ".join(
                f"{lo / _CONF_BUCKETS:.1f}:{a}/{n}"
                for lo, (n, a) in enumerate(zip(row, ok_row))
                if n
            )
            logger.info(
                "DSpark confidence slot %d over %d steps: accepted %d/%d "
                "(%.1f%%) by bucket %s",
                slot_idx,
                self._conf_stats_steps,
                sum(ok_row),
                total,
                100.0 * sum(ok_row) / total,
                cells,
            )

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
        if _CONF_STATS:
            # First thing in the step: num_rejected_tokens_gpu grades the
            # proposal we scored on the previous call.
            self._record_confidence_stats(num_rejected_tokens_gpu)
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
        self._valid_draft_counts = None

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
        # The confidence head reads the dense pre-norm state, so ask for it in
        # the same pass rather than running hc_head a second time. When the
        # caller supplied logits (the debug path) that state is gone and we
        # skip scoring rather than recompute the vocab projection.
        want_conf = (_CONF_ENABLED or _CONF_STATS) and logits is None
        dense: torch.Tensor | None = None
        if logits is None:
            if want_conf:
                logits, dense = self.model.model.compute_logits_and_dense(
                    hidden_states
                )
            else:
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
        markov_embeds = [] if want_conf else None

        for slot in range(num_spec):
            bias, embed = self.model.model.markov_bias(prev_token)
            if want_conf:
                # The reference scores slot i against the embedding of the
                # token that seeded it, i.e. the walk's input, not its output
                # (inference/model.py:872-880).
                markov_embeds.append(embed)
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

        if want_conf:
            assert dense is not None
            confidence = self.model.model.confidence(
                dense.view(batch_size, num_spec, -1),
                torch.stack(markov_embeds, dim=1),
            )
            counts = (
                self._truncate_from_confidence(confidence)
                if _CONF_ENABLED
                else torch.full(
                    (batch_size,),
                    num_spec,
                    dtype=torch.int32,
                    device=confidence.device,
                )
            )
            if _CONF_ENABLED:
                self._valid_draft_counts = counts
            if _CONF_STATS:
                self._prev_conf = confidence
                self._prev_valid_counts = counts

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
