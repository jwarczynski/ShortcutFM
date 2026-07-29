"""Masked (absorbing-state) discrete diffusion criteria with shortcut self-consistency.

Discrete counterpart of the continuous flow-matching criteria in `shortcutfm.criteria`.
The forward process replaces target tokens with the mask token with probability t/T
(LLaDA/MDLM linear schedule); the model predicts the clean tokens from the partially
masked sequence, conditioned on t and a shortcut size d. Self-consistency trains one
step of size 2d to match the composition of two steps of size d.
"""

from collections.abc import Callable
from typing import override

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase

from shortcutfm.batch import EncoderBatch, MaskedDiffusionBatch, MaskedShortcutBatch
from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import Criterion
from shortcutfm.model.model import FlowMatchingModel as Model
from shortcutfm.shortcut_samplers import (
    LossAwareSampler,
    ScheduleSampler,
    TimeAndShortcutSampler,
)


def select_positions_to_unmask(
    currently_masked: Tensor,
    num_to_unmask: Tensor,
    logits: Tensor,
    strategy: str,
) -> Tensor:
    """Select which masked positions to reveal.

    :param currently_masked: Bool [bsz, seq_len], True where the sequence is masked
    :param num_to_unmask: Long [bsz], number of positions to reveal per sample
    :param logits: [bsz, seq_len, vocab_size] model logits used for confidence ordering
    :param strategy: "random" or "confidence"
    :return: Bool [bsz, seq_len], True at positions to reveal (subset of currently_masked)
    """
    if strategy == "confidence":
        scores = torch.softmax(logits, dim=-1).amax(dim=-1)
    elif strategy == "random":
        scores = torch.rand(currently_masked.shape, device=currently_masked.device)
    else:
        raise ValueError(f"Unknown unmask strategy: {strategy}")

    scores = scores.masked_fill(~currently_masked, float("-inf"))
    # Rank positions per sample by descending score; reveal the top num_to_unmask
    order = scores.argsort(dim=-1, descending=True)
    ranks = order.argsort(dim=-1)
    return (ranks < num_to_unmask.unsqueeze(-1)) & currently_masked


class MaskedDiffusionCriterion(Criterion):
    """Masked-diffusion cross-entropy loss and iterative unmasking inference."""

    def __init__(
        self,
        model: Model,
        diffusion_steps: int,
        tokenizer: PreTrainedTokenizerBase,
        training_cfg: TrainingConfig = None,
        default_shortcut_factory: Callable = lambda t: torch.zeros_like(t),
    ):
        super().__init__(model, diffusion_steps, training_cfg)
        self.tokenizer = tokenizer
        self.default_shortcut_factory = default_shortcut_factory
        self.mask_token_id = training_cfg.model.mask_token_id
        self.unmask_strategy = training_cfg.model.unmask_strategy
        self.use_one_over_t = training_cfg.model.ce_importance_weighting == "one_over_t"

    def corrupt(self, seqs: Tensor, t: Tensor, denoise_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Sample x_t from q(x_t | x_0): mask each target token with probability t/T.

        :param seqs: Long [bsz, seq_len] clean token ids
        :param t: Long [bsz] timesteps in [1, diffusion_steps]
        :param denoise_mask: [bsz, seq_len], 1 on real target tokens (input_ids_mask * padding_mask)
        :return: (x_t Long [bsz, seq_len], mask_indicator Bool [bsz, seq_len])
        """
        mask_prob = self.scale_t(t).unsqueeze(-1)
        mask_indicator = (torch.rand(seqs.shape, device=seqs.device) < mask_prob) & (denoise_mask == 1)

        # Force at least one masked position per sample so no sample has zero loss signal
        no_mask = (mask_indicator.sum(-1) == 0) & (denoise_mask.sum(-1) > 0)
        if no_mask.any():
            scores = torch.rand(seqs.shape, device=seqs.device) * denoise_mask
            forced = torch.zeros_like(mask_indicator)
            forced[torch.arange(seqs.size(0), device=seqs.device), scores.argmax(-1)] = True
            mask_indicator = mask_indicator | (forced & no_mask.unsqueeze(-1))

        x_t = torch.where(mask_indicator, self.mask_token_id, seqs)
        return x_t, mask_indicator

    @override
    def compute_losses(self, batch: MaskedDiffusionBatch, world_size) -> dict[str, Tensor]:
        shortcuts = self.default_shortcut_factory(batch.t)
        hidden = self.model(self.model.get_embeddings(batch.x_t), batch.t, shortcuts)
        logits = self.model.compute_logits(hidden)

        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.seqs.view(-1),
            reduction="none",
        ).view(batch.seqs.shape)

        ce = ce * batch.mask_indicator
        if self.use_one_over_t:
            ce = ce * (self.diffusion_steps / batch.t.float()).unsqueeze(-1)

        return {"masked_ce_loss": ce}

    def denoise(
        self,
        batch: EncoderBatch,
        shortcut_size: int | None = None,
        probe_every_step: bool = True,
        return_decoded: bool = False,
        return_logits: bool = False,
        step_size: int | None = None,
        guidance_scale: float | None = None,
        use_ground_truth_embeddings: bool = False,
        unmask_strategy: str | None = None,
        decode_mode: str = "schedule",
        conf_threshold: float = 0.9,
        block_size: int = 32,
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        """Iteratively unmask target tokens, conditioned on shortcut size.

        Output modes mirror `FlowMatchingCriterion.denoise`:
            - return_logits: commit-time logits per position,
              [bsz, num_steps, seq_len, vocab] if probe_every_step else [bsz, seq_len, vocab]
            - return_decoded: decoded strings (per step if probe_every_step)
            - otherwise token ids, [bsz, num_steps, seq_len] if probe_every_step else [bsz, seq_len]

        `decode_mode` selects how many / which masked positions commit each step:
            - "schedule" (default): reveal enough to reach mask ratio t_next/t, ordered
              by `unmask_strategy` (confidence or random). Original behaviour.
            - "threshold": commit every position whose top-class probability >=
              `conf_threshold` (a variable count per step; >=1 forced so it terminates).
            - "block": LLaDA-style left-to-right block decoding — eligibility restricted
              to the left-most block of `block_size` target positions that still has
              masks, then the schedule count applied within that block.
        """
        if decode_mode not in ("schedule", "threshold", "block"):
            raise ValueError(f"Unknown decode_mode: {decode_mode}")
        if shortcut_size is None and step_size is None:
            raise ValueError("Either shortcut_size or step_size must be provided")
        if (shortcut_size == 0 or shortcut_size is None) and step_size is None:
            raise ValueError("step_size must be provided when shortcut_size is 0 or None")
        if use_ground_truth_embeddings:
            raise NotImplementedError("use_ground_truth_embeddings is not supported for masked diffusion")

        self.model.eval()
        effective_step = step_size or shortcut_size
        shortcut_size = shortcut_size or 0
        # Inference default is low-confidence remasking (LLaDA); the training-config
        # strategy only governs consistency-target construction
        unmask_strategy = unmask_strategy or "confidence"

        seqs = batch.seqs
        bsz, seq_len = seqs.shape
        device = seqs.device
        vocab_size = self.model.module.word_embedding.num_embeddings
        denoise_mask = batch.input_ids_mask * batch.padding_mask

        x = torch.where(denoise_mask == 1, self.mask_token_id, seqs)
        # Track masked state explicitly — comparing x to mask_token_id would break if the
        # model ever predicts the mask token itself
        currently_masked = denoise_mask == 1
        # For block decoding: assign each target position a block id by its left-to-right
        # rank within the target region (source/pad positions get a large sentinel so they
        # never win the "left-most block with masks" selection)
        if decode_mode == "block":
            target_rank = (denoise_mask == 1).long().cumsum(dim=-1) - 1
            block_id = torch.where(denoise_mask == 1, target_rank // block_size, seq_len)
        # Source and true-padding positions carry ground-truth one-hot logits so the
        # return_logits path never exposes model re-predictions of the conditioning
        committed_logits = torch.zeros((bsz, seq_len, vocab_size), dtype=torch.float, device=device)
        committed_logits.scatter_(-1, seqs.unsqueeze(-1), 1.0)

        num_steps = len(range(self.diffusion_steps, 0, -effective_step))
        if probe_every_step:
            if return_logits:
                predictions = torch.zeros((bsz, num_steps, seq_len, vocab_size), dtype=torch.float, device=device)
            else:
                predictions = torch.zeros((bsz, num_steps, seq_len), dtype=torch.long, device=device)

        shortcuts = torch.tensor(shortcut_size, device=device).repeat(bsz)
        for step_idx, t in enumerate(range(self.diffusion_steps, 0, -effective_step)):
            t_next = max(t - effective_step, 0)
            t_tensor = torch.tensor(t, device=device).repeat(bsz)

            hidden = self.model(self.model.get_embeddings(x), t_tensor, shortcuts)
            logits = self.model.compute_logits(hidden)
            # The mask token is never a valid prediction — suppress it so revealed
            # positions always receive a real token
            logits[..., self.mask_token_id] = float("-inf")
            predicted_tokens = logits.argmax(dim=-1)

            # Positions not yet committed get this step's logits; earlier commits and
            # ground-truth source/padding positions are kept
            committed_logits = torch.where(currently_masked.unsqueeze(-1), logits, committed_logits)

            if decode_mode == "threshold":
                # Commit every still-masked position confident enough; force >=1 per
                # sample (among samples that still have masks) so the loop terminates
                confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
                to_unmask = (confidence >= conf_threshold) & currently_masked
                has_mask = currently_masked.any(dim=-1)
                needs_forced = has_mask & ~to_unmask.any(dim=-1)
                if needs_forced.any():
                    forced = select_positions_to_unmask(
                        currently_masked,
                        needs_forced.long(),
                        logits,
                        "confidence",
                    )
                    to_unmask = to_unmask | forced
            elif decode_mode == "block":
                # Restrict eligibility to the left-most block that still holds masks,
                # then apply the schedule count within that block
                masked_block = torch.where(currently_masked, block_id, seq_len)
                active_block = masked_block.amin(dim=-1, keepdim=True)
                eligible = currently_masked & (block_id == active_block)
                num_eligible = eligible.sum(-1)
                keep = torch.round(num_eligible.float() * t_next / t).long()
                to_unmask = select_positions_to_unmask(
                    eligible, num_eligible - keep, logits, unmask_strategy
                )
            else:
                # "schedule": reveal enough positions to reach mask ratio t_next / t
                num_masked = currently_masked.sum(-1)
                keep = torch.round(num_masked.float() * t_next / t).long()
                to_unmask = select_positions_to_unmask(
                    currently_masked, num_masked - keep, logits, unmask_strategy
                )
            x = torch.where(to_unmask, predicted_tokens, x)
            currently_masked = currently_masked & ~to_unmask

            if probe_every_step or step_idx == num_steps - 1:
                if return_logits:
                    step_predictions = committed_logits
                else:
                    # Preview: committed tokens plus current argmax at still-masked positions
                    step_predictions = torch.where(currently_masked, predicted_tokens, x)
                if probe_every_step:
                    # noinspection PyUnboundLocalVariable
                    predictions[:, step_idx] = step_predictions
                else:
                    predictions = step_predictions

        if return_decoded and not return_logits:
            if probe_every_step:
                flat_preds = predictions.reshape(-1, predictions.shape[-1])
                decoded = self.tokenizer.batch_decode(flat_preds, skip_special_tokens=True)
                return [decoded[i : i + num_steps] for i in range(0, len(decoded), num_steps)]
            return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        return predictions


class MaskedConsistencyCriterion(Criterion):
    """Shortcut self-consistency for masked diffusion.

    One model step of size 2d is trained to match the composition of two steps of
    size d (computed without gradient). The comparison space is configurable:
    L2 between expected embeddings (softmax(logits) @ word_embedding) or KL between
    token distributions.
    """

    def __init__(
        self,
        model: Model,
        diffusion_steps: int,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        default_shortcut_factory: Callable = lambda t: torch.zeros_like(t),
    ):
        super().__init__(model, diffusion_steps, training_cfg)
        self.reduce_fn = reduce_fn
        self.default_shortcut_factory = default_shortcut_factory
        self.mask_token_id = training_cfg.model.mask_token_id
        self.unmask_strategy = training_cfg.model.unmask_strategy
        self.loss_space = training_cfg.model.consistency_loss_space

    @override
    def losses_with_mask(self, batch: MaskedShortcutBatch, world_size) -> dict[str, Tensor]:
        # Normalize over masked positions only, not all target positions
        losses = self.compute_losses(batch, world_size)
        loss_mask = batch.mask_indicator * batch.padding_mask
        for key, value in losses.items():
            masked_per_token_loss = loss_mask * value
            per_batch_loss = masked_per_token_loss.sum(-1) / loss_mask.sum(-1).clamp(min=1)
            losses[key] = per_batch_loss
        return losses

    @override
    def compute_losses(self, batch: MaskedShortcutBatch, world_size) -> dict[str, Tensor]:
        target_logits = self._compute_shortcut_target(batch)
        prediction_logits = self._predict(batch)

        if self.loss_space == "l2_embedding":
            weight = self.model.module.word_embedding.weight
            expected_pred = torch.softmax(prediction_logits, dim=-1) @ weight
            with torch.no_grad():
                expected_target = torch.softmax(target_logits, dim=-1) @ weight.detach()
            loss = self.reduce_fn((expected_pred - expected_target) ** 2, dim=-1)
        elif self.loss_space == "kl":
            loss = F.kl_div(
                F.log_softmax(prediction_logits, dim=-1),
                torch.softmax(target_logits, dim=-1),
                reduction="none",
            ).sum(-1)
        else:
            raise ValueError(f"Unknown consistency loss space: {self.loss_space}")

        return {"consistency_loss": loss * batch.mask_indicator}

    @torch.no_grad()
    def _compute_shortcut_target(self, batch: MaskedShortcutBatch) -> Tensor:
        shortcut_input = (
            self.default_shortcut_factory(batch.t)
            if self.training_cfg.model.use_default_t_for_shortcut
            else batch.shortcut_size
        )

        hidden_1 = self.model(self.model.get_embeddings(batch.x_t), batch.t, shortcut_input)
        logits_1 = self.model.compute_logits(hidden_1)

        x_t_minus_d, _, unmasked_now = self._build_x_t_minus_d(
            batch.x_t, batch.mask_indicator, logits_1, batch.t, batch.shortcut_size
        )

        hidden_2 = self.model(
            self.model.get_embeddings(x_t_minus_d), batch.t - batch.shortcut_size, shortcut_input
        )
        logits_2 = self.model.compute_logits(hidden_2)

        # Positions committed by step 1 keep step 1's distribution; the rest take step 2's
        target_logits = torch.where(unmasked_now.unsqueeze(-1), logits_1, logits_2)
        return target_logits.detach()

    def _build_x_t_minus_d(
        self,
        x_t: Tensor,
        mask_indicator: Tensor,
        logits_1: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Construct x_{t-d} by revealing predicted tokens so the mask ratio drops to (t-d)/t.

        :return: (x_t_minus_d, still_masked, unmasked_now) — the latter two Bool [bsz, seq_len]
        """
        num_masked = mask_indicator.sum(-1)
        keep = torch.round(num_masked.float() * (t - shortcut_size).float() / t.float()).long()
        unmasked_now = select_positions_to_unmask(
            mask_indicator, num_masked - keep, logits_1, self.unmask_strategy
        )
        x_t_minus_d = torch.where(unmasked_now, logits_1.argmax(dim=-1), x_t)
        still_masked = mask_indicator & ~unmasked_now
        return x_t_minus_d, still_masked, unmasked_now

    def _predict(self, batch: MaskedShortcutBatch) -> Tensor:
        hidden = self.model(
            self.model.get_embeddings(batch.x_t), batch.t, 2 * batch.shortcut_size
        )
        return self.model.compute_logits(hidden)


class MaskedCompositeCriterion(Criterion):
    """Orchestrates masked CE and shortcut consistency losses, mirroring `CompositeCriterion`."""

    def __init__(
        self,
        masked_criterion: MaskedDiffusionCriterion,
        consistency_criterion: MaskedConsistencyCriterion,
        masked_ce_weight: float,
        consistency_weight: float,
        model: Model,
        diffusion_steps: int,
        self_consistency_ratio: float,
        sampler: ScheduleSampler,
        time_shortcut_sampler: TimeAndShortcutSampler,
        training_cfg: TrainingConfig = None,
    ):
        super().__init__(model, diffusion_steps, training_cfg=training_cfg)
        self.masked_criterion = masked_criterion
        self.consistency_criterion = consistency_criterion
        self.masked_ce_weight = masked_ce_weight
        self.consistency_weight = consistency_weight
        self.self_consistency_ratio = self_consistency_ratio
        self.sampler = sampler
        self.time_shortcut_sampler = time_shortcut_sampler

    def forward(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        return self.compute_losses(batch, world_size)

    @override
    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        self.global_step = batch.global_step
        self.masked_criterion.global_step = batch.global_step

        masked_batch, consistency_batch, weights = self._prepare_batches(batch)
        ce_sampler_weights, consistency_sampler_weights = weights

        masked_ce_loss = self.masked_criterion(masked_batch, world_size)["masked_ce_loss"]

        if consistency_batch is not None:
            consistency_loss = self.consistency_criterion(consistency_batch, world_size)["consistency_loss"]

            self.time_shortcut_sampler.update_with_local_losses(
                consistency_batch.t - 1,
                consistency_loss.detach(),
                world_size=world_size,
            )

            total_loss = (
                self.masked_ce_weight * (masked_ce_loss * ce_sampler_weights).mean()
                + self.consistency_weight * (consistency_loss * consistency_sampler_weights).mean()
            )
            result = {
                "masked_ce_loss": masked_ce_loss * self.masked_ce_weight,
                "consistency_loss": consistency_loss * self.consistency_weight,
                "timestep": torch.cat([masked_batch.t, consistency_batch.t], dim=0),
                "shortcut": consistency_batch.shortcut_size,
            }
        else:
            total_loss = self.masked_ce_weight * (masked_ce_loss * ce_sampler_weights).mean()
            result = {
                "masked_ce_loss": masked_ce_loss * self.masked_ce_weight,
                "timestep": masked_batch.t,
            }

        result["loss"] = total_loss

        if isinstance(self.sampler, LossAwareSampler):
            self.sampler.update_with_local_losses(
                masked_batch.t - 1,
                masked_ce_loss.detach(),
                world_size=world_size,
            )

        return result

    def _clip_t(self, t: Tensor) -> Tensor:
        """Clamp sampled timesteps to the configured mask-ratio bandwidth (LLaDA2.0).

        Extreme mask ratios give high gradient variance with little signal: at low t
        only 0-1 target tokens are supervised (with a huge T/t weight); applies to the
        CE branch only — the consistency branch needs the full (t, d) range.
        """
        t_min_frac = getattr(self.training_cfg.model, "t_min_frac", 0.0)
        t_max_frac = getattr(self.training_cfg.model, "t_max_frac", 1.0)
        if t_min_frac <= 0.0 and t_max_frac >= 1.0:
            return t
        t_min = max(int(t_min_frac * self.diffusion_steps), 1)
        t_max = min(int(t_max_frac * self.diffusion_steps), self.diffusion_steps)
        # Rescale linearly into [t_min, t_max] instead of clamping, which would pile
        # probability mass onto the band edges
        rescaled = t_min + (t.float() - 1) * (t_max - t_min) / max(self.diffusion_steps - 1, 1)
        return rescaled.round().long().clamp(t_min, t_max)

    def _prepare_batches(
        self, batch: EncoderBatch
    ) -> tuple[MaskedDiffusionBatch, MaskedShortcutBatch | None, tuple]:
        bsz = batch.size()

        use_consistency = (
            self.global_step >= self.training_cfg.consistency_start_step
            and self.training_cfg.self_consistency_ratio > 0
        )

        num_consistency_elems = int(self.self_consistency_ratio * bsz) if use_consistency else 0
        num_ce_elems = bsz - num_consistency_elems

        ce_seqs = batch.seqs[:num_ce_elems]
        ce_padding_mask = batch.padding_mask[:num_ce_elems]
        ce_input_ids_mask = batch.input_ids_mask[:num_ce_elems]
        t, ce_weights = self.sampler(batch_size=num_ce_elems, device=batch.seqs.device)
        t = self._clip_t(t)
        x_t, mask_indicator = self.masked_criterion.corrupt(ce_seqs, t, ce_input_ids_mask * ce_padding_mask)
        masked_batch = MaskedDiffusionBatch(
            seqs=ce_seqs,
            padding_mask=ce_padding_mask,
            input_ids_mask=ce_input_ids_mask,
            x_t=x_t,
            mask_indicator=mask_indicator,
            t=t,
            global_step=batch.global_step,
        )

        if not use_consistency:
            return masked_batch, None, (ce_weights, None)

        consistency_seqs = batch.seqs[num_ce_elems:]
        consistency_padding_mask = batch.padding_mask[num_ce_elems:]
        consistency_input_ids_mask = batch.input_ids_mask[num_ce_elems:]
        consistency_t, shortcuts, consistency_weights = self.time_shortcut_sampler(
            batch_size=num_consistency_elems,
            device=batch.seqs.device,
        )
        consistency_x_t, consistency_mask_indicator = self.masked_criterion.corrupt(
            consistency_seqs, consistency_t, consistency_input_ids_mask * consistency_padding_mask
        )
        consistency_batch = MaskedShortcutBatch(
            seqs=consistency_seqs,
            padding_mask=consistency_padding_mask,
            input_ids_mask=consistency_input_ids_mask,
            x_t=consistency_x_t,
            mask_indicator=consistency_mask_indicator,
            t=consistency_t,
            shortcut_size=shortcuts,
            global_step=batch.global_step,
        )

        return masked_batch, consistency_batch, (ce_weights, consistency_weights)

    def denoise(
        self,
        batch: EncoderBatch,
        shortcut_size: int | None = None,
        probe_every_step: bool = True,
        return_decoded: bool = False,
        return_logits: bool = False,
        step_size: int | None = None,
        guidance_scale: float | None = None,
        use_ground_truth_embeddings: bool = False,
        unmask_strategy: str | None = None,
        decode_mode: str = "schedule",
        conf_threshold: float = 0.9,
        block_size: int = 32,
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        return self.masked_criterion.denoise(
            batch,
            shortcut_size=shortcut_size,
            probe_every_step=probe_every_step,
            return_decoded=return_decoded,
            return_logits=return_logits,
            step_size=step_size,
            guidance_scale=guidance_scale,
            use_ground_truth_embeddings=use_ground_truth_embeddings,
            unmask_strategy=unmask_strategy,
            decode_mode=decode_mode,
            conf_threshold=conf_threshold,
            block_size=block_size,
        )
