from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, override

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase

from shortcutfm.batch import EncoderBatch, FlowMatchingBatch, ShortcutFMBatch
from shortcutfm.config import TrainingConfig
from shortcutfm.model.model import FlowMatchingModel as Model
from shortcutfm.shortcut_samplers import (
    LossAwareSampler,
    ScheduleSampler,
    TimeAndShortcutSampler,
)


class Criterion(Module, ABC):
    def __init__(
        self,
        model: Model,
        diffusion_steps: int,
        training_cfg: TrainingConfig = None,
    ):
        super().__init__()
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.training_cfg = training_cfg
        self.global_step = 0  # Explicitly define the global step attribute at the base class level

    def forward(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        """Compute the losses."""
        return self.losses_with_mask(batch, world_size)

    def losses_with_mask(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        """Compute the losses applying mask."""
        padding_mask = batch.padding_mask
        input_ids_mask = batch.input_ids_mask

        losses = self.compute_losses(batch, world_size)
        loss_mask = padding_mask * input_ids_mask
        for key, value in losses.items():
            masked_per_token_loss = loss_mask * value
            per_batch_loss = masked_per_token_loss.sum(-1) / loss_mask.sum(-1)
            losses[key] = per_batch_loss
        return losses

    @abstractmethod
    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        """Compute the losses."""

    def _interpolate_data_noise(self, x_start: Tensor, t: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        t = self.scale_t(t)
        t = t.view(-1, 1, 1)  # Reshape to (batch_size, 1, 1) to match x_start

        if noise is None:
            noise = torch.randn_like(x_start)

        return x_start + (noise - x_start) * t, noise

    def scale_t(self, t):
        return t.float() * (1.0 / self.diffusion_steps)


class FlowMatchingCriterion(Criterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        tokenizer: PreTrainedTokenizerBase,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
        default_shortcut_factory: Callable = lambda t: t,
    ):
        super().__init__(model, diffusion_steps, training_cfg)
        self.tokenizer = tokenizer
        self.x_t = None
        self.reduce_fn = reduce_fn
        self.loss_fn = loss_fn
        self.default_shortcut_factory = default_shortcut_factory

    def compute_losses(self, batch: FlowMatchingBatch, world_size) -> dict[str, Tensor]:
        target = self._compute_target(batch)
        output = self._predict(
            x_start=batch.x_start,
            x_t=batch.x_t,
            noise=batch.noise,
            t=batch.t,
            input_ids_mask=batch.input_ids_mask,
        )

        fm_loss = self.loss_fn(output, target)
        if self.training_cfg.normalize_flow_matching_loss:
            target_norms = torch.norm(target, dim=-1, keepdim=True)  # Shape: (batch_size, seq_len, 1)
            fm_loss = fm_loss / (target_norms + 1e-10)

        x_start_predicted = self.get_x0_from_predicition(output, batch)
        decoder_loss = self._compute_nll_loss(x_start_predicted, batch.seqs)

        return {
            "flow_matching_loss": self.reduce_fn(fm_loss, dim=-1),
            "decoder_loss": decoder_loss,
        }

    def _predict(
        self,
        *,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        """Compute the model output with optional classifier-free guidance.

        This implements Algorithm 1 from the paper for joint training with classifier-free guidance:
        - With probability cfg_probability, we discard conditioning (train unconditionally)
        - Otherwise, we train with conditioning

        During training, we're optimizing the model to predict the noise (or x0) for both
        conditional and unconditional cases, which allows us to use classifier-free guidance
        during inference.
        """
        shortcut_size = self.default_shortcut_factory(t)

        # Determine whether to train unconditionally (discard conditioning)
        # This implements: 'c ← ∅ with probability puncond'
        train_unconditionally = self.should_apply_cfg()

        if train_unconditionally:
            # For unconditional training, we use the null token embedding instead of zeros
            # Create a version of x_t where input tokens are replaced with null token embeddings
            null_token_id = self.training_cfg.model.null_token_id
            null_token_embedding = self.model.get_embeddings(torch.tensor([null_token_id], device=x_t.device))
            x_t_uncond = torch.where(input_ids_mask.unsqueeze(-1) == 0, null_token_embedding, x_t)
            # Train the model to predict the noise (or x0) for the unconditional case
            y = self.model(x_t_uncond, t, shortcut_size)
        else:
            # For conditional training, use the original x_t with the provided input_ids_mask
            # Train the model to predict the noise (or x0) for the conditional case
            y = self.model(x_t, t, shortcut_size)

        return y

    def should_apply_cfg(self):
        """Determine whether to apply unconditional training (discard conditioning) based on cfg_probability.

        Following Algorithm 1 from the paper, this implements the step:
        'c ← ∅ with probability puncond' (Randomly discard conditioning to train unconditionally)

        Returns:
            bool: True if we should discard conditioning (train unconditionally), False otherwise
        """
        # Check if CFG is enabled and we've reached the start step
        if self.training_cfg.cfg_start_step is None or self.global_step < self.training_cfg.cfg_start_step:
            return False

        # If CFG is enabled, randomly decide whether to discard conditioning based on cfg_probability
        # Draw a random number and check if it's less than cfg_probability
        # If it is, we discard conditioning (train unconditionally)
        random_value = torch.rand(1).item()
        return random_value < self.training_cfg.cfg_probability

    def _compute_nll_loss(self, hidden_last: Tensor, seqs: Tensor) -> Tensor:
        logits = self.model.compute_logits(hidden_last)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            seqs.view(-1),
            reduction="none",
        ).view(seqs.size())

    def denoise(
        self,
        batch: EncoderBatch,
        shortcut_size: int | None = None,
        probe_every_step: bool = True,
        return_decoded: bool = False,
        return_logits: bool = False,
        step_size: int | None = None,
        guidance_scale: float | None = None,
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        """Denoises batch of examples with flexible probing and output options.

        :param batch: batch of examples to denoise
        :type batch: EncoderBatch
        :param shortcut_size: shortcut size to use during denoising. If None or 0, step_size must be provided
        :type shortcut_size: Optional[int]
        :param probe_every_step: whether to probe at every step or only at the final step
        :type probe_every_step: bool
        :param return_decoded: whether to return decoded sequences or token IDs
        :type return_decoded: bool
        :param return_logits: whether to return logits instead of token IDs
        :type return_logits: bool
        :param step_size: step size to use during denoising when shortcut_size is None or 0
        :type step_size: Optional[int]

        :returns: One of the following based on parameters:
            - If return_logits=True:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len, vocab_size] with logits
                - If probe_every_step=False: Tensor[batch_size, seq_len, vocab_size] with logits
            - If return_decoded=True:
                - If probe_every_step=True: List[List[str]] where outer list is batches, inner list is steps
                - If probe_every_step=False: List[str] of decoded sequences
            - Otherwise:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len] with token IDs
                - If probe_every_step=False: Tensor[batch_size, seq_len] with token IDs
        :rtype: Union[Tensor, list[list[str]]]
        """
        if shortcut_size is None and step_size is None:
            raise ValueError("Either shortcut_size or step_size must be provided")
        if (shortcut_size == 0 or shortcut_size is None) and step_size is None:
            raise ValueError("step_size must be provided when shortcut_size is 0 or None")

        guidance_scale = guidance_scale or self.training_cfg.cfg_guidance_scale
        self.model.eval()
        # Use step_size if shortcut_size is None or 0
        # effective_step = step_size if (shortcut_size is None or shortcut_size == 0) else shortcut_size
        effective_step = step_size or shortcut_size
        shortcut_size = shortcut_size or 0

        self._reset()
        input_mask = batch.input_ids_mask.unsqueeze(-1)
        embeddings = self.model.get_embeddings(batch.seqs)
        noise = torch.randn_like(embeddings)
        self.x_t = torch.where(input_mask == 0, embeddings, noise)

        # Pre-allocate tensor for predictions if probing every step
        num_steps = len(range(self.diffusion_steps, 0, -effective_step))
        if probe_every_step:
            if return_logits:
                predictions = torch.zeros(
                    (
                        batch.seqs.shape[0],
                        num_steps,
                        batch.seqs.shape[1],
                        self.model.vocab_size,
                    ),
                    dtype=torch.float,
                    device=batch.seqs.device,
                )
            else:
                predictions = torch.zeros(
                    (batch.seqs.shape[0], num_steps, batch.seqs.shape[1]),
                    dtype=torch.long,
                    device=batch.seqs.device,
                )

        shortcuts = torch.tensor(shortcut_size, device=input_mask.device).repeat(input_mask.shape[0])
        for step_idx, t in enumerate(torch.arange(self.diffusion_steps, 0, -effective_step, device=input_mask.device)):
            t: Tensor = t.repeat(input_mask.shape[0])
            model_output = self.infere_model(self.x_t, t, shortcuts, input_mask, guidance_scale=guidance_scale)
            v_hat = self.compute_velocity(model_output, noise, input_mask)
            x0_hat = self.x_t + (effective_step / self.diffusion_steps) * v_hat
            self.x_t = x0_hat

            # Get predictions if probing every step or if this is the last step
            if probe_every_step or step_idx == num_steps - 1:
                step_predictions = self.probe(x0_hat, return_logits=return_logits)
                if probe_every_step:
                    # noinspection PyUnboundLocalVariable
                    predictions[:, step_idx, :] = step_predictions
                else:
                    predictions = step_predictions

        # Handle output format
        if return_decoded and not return_logits:
            if probe_every_step:
                # Reshape to [batch_size * num_steps, seq_len] for batch decoding
                flat_preds = predictions.reshape(-1, predictions.shape[-1])
                decoded = self.tokenizer.batch_decode(flat_preds, skip_special_tokens=True)
                # Reshape back to [batch_size, num_steps]
                return [decoded[i : i + num_steps] for i in range(0, len(decoded), num_steps)]
            else:
                return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        return predictions

    def infere_model(
        self,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_mask: Tensor,
        guidance_scale: float | None = None,
    ) -> Tensor:
        """Call the model and restore input part of the prediction with optional CFG"""
        # Check if CFG should be applied during inference
        guidance_scale = guidance_scale or self.training_cfg.cfg_guidance_scale
        if guidance_scale == 1.0 or self.training_cfg.cfg_start_step is None:
            # Standard prediction without guidance
            model_output = self.model(x_t, t, shortcut_size)
            return self._restore_input_part(model_output, x_t, input_mask)

        # Apply classifier-free guidance during inference
        # 1. Get conditional prediction (with input_mask)
        y_cond = self.model(x_t, t, shortcut_size)

        # 2. Get unconditional prediction using null token embedding
        null_token_id = self.training_cfg.model.null_token_id
        null_token_embedding = self.model.get_embeddings(torch.tensor([null_token_id], device=x_t.device))
        x_t_uncond = torch.where(input_mask == 0, null_token_embedding, x_t)
        y_uncond = self.model(x_t_uncond, t, shortcut_size)

        # 3. Apply guidance formula: y = y_uncond + guidance_scale * (y_cond - y_uncond)
        guidance_scale = self.training_cfg.cfg_guidance_scale
        model_output = y_uncond + guidance_scale * (y_cond - y_uncond)

        return self._restore_input_part(model_output, x_t, input_mask)

    @abstractmethod
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        """Recover input part of the prediction based on input_mask"""

    @abstractmethod
    def compute_velocity(
        self,
        model_output: Tensor,
        noise: Tensor,
        input_mask: Tensor,
    ) -> Tensor:
        """Computes velocity based on models output for the denoising process"""

    def probe(self, hidden_representation, return_logits: bool = False) -> Tensor:
        """Predicts sequence of tokens based on hidden_representation.

        :param hidden_representation: Hidden representation from the model
        :type hidden_representation: Tensor
        :param return_logits: Whether to return logits instead of token IDs
        :type return_logits: bool

        :return: Either logits or token IDs
        :rtype: Tensor
        """
        logits = self.model.compute_logits(hidden_representation)
        if return_logits:
            return logits
        probs = torch.softmax(logits, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        return tokens

    def _reset(self):
        """Allow subclasses to prepare for new batch of examples.

        For example, it can reset stored conditioning values.
        """
        pass

    @abstractmethod
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        """Compute the target."""

    @abstractmethod
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Tensor | None = None) -> Tensor:
        """Modify model input based on input_ids_mask. Used for self-conditioning."""

    @abstractmethod
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        """Extract x0 from the model prediction."""


class X0FlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        tokenizer: PreTrainedTokenizerBase,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
        default_shortcut_factory: Callable = lambda t: t,
    ):
        super().__init__(model, diffusion_steps, tokenizer, reduce_fn, training_cfg, loss_fn, default_shortcut_factory)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start

    @override
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Tensor | None = None) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)

    @override
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        return torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, batch.x_start, y_hat)

    @override
    def compute_velocity(self, x0_hat: Tensor, noise: Tensor, input_mask: Tensor) -> Tensor:
        v_hat = x0_hat - noise
        input_mask = input_mask.unsqueeze(-1) if input_mask.dim() == 2 else input_mask
        v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)
        return v_hat

    @override
    def _restore_input_part(self, x0_hat: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        return torch.where(input_mask == 0, x_t, x0_hat)


class VelocityFlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        tokenizer: PreTrainedTokenizerBase,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
        default_shortcut_factory: Callable = lambda t: t,
    ):
        super().__init__(model, diffusion_steps, tokenizer, reduce_fn, training_cfg, loss_fn, default_shortcut_factory)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start - batch.noise

    @override
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Tensor | None = None) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)

    @override
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        distance_to_x0 = batch.t[:, None, None] / self.diffusion_steps
        x0 = batch.x_t + y_hat * distance_to_x0
        return torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, batch.x_start, x0)

    @override
    def compute_velocity(self, v_hat: Tensor, noise: Tensor, input_mask: Tensor) -> Tensor:
        input_mask = input_mask.unsqueeze(-1) if input_mask.dim() == 2 else input_mask
        v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)
        return v_hat

    @override
    def _restore_input_part(self, v_hat: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        return torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)


class FlowMatchingCriterionDecorator(FlowMatchingCriterion, ABC):
    def __init__(
        self,
        criterion: FlowMatchingCriterion,
    ):
        super().__init__(
            criterion.model,
            criterion.diffusion_steps,
            criterion.tokenizer,
            criterion.reduce_fn,
            criterion.training_cfg,
            criterion.loss_fn,
            criterion.default_shortcut_factory,
        )
        self.criterion = criterion


class SelfConditioningFlowMatchingCriterionDecorator(FlowMatchingCriterionDecorator):
    def __init__(
        self,
        criterion: FlowMatchingCriterion,
        self_conditioning_ratio: float,
    ):
        super().__init__(criterion)
        self.self_conditioning_ratio = self_conditioning_ratio
        self.y_hat = None

    @override
    def _predict(
        self,
        *,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        """Compute the model output."""
        # prepare self-conditioning input
        x_0_hat = self._modify_model_input(input_ids_mask, x_start)

        if not self._should_apply_self_conditioning():
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion._predict(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                input_ids_mask=input_ids_mask,
            )

        # TODO: model won't be quried at t + 1 during inference, but at t + d
        # TODO: draw a shorcut value from distribution?
        t_next = torch.where(t < self.diffusion_steps, t + 1, t)
        x_t_next, _ = self._interpolate_data_noise(x_start, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            shortcut_size = self.default_shortcut_factory(t_next)
            y_hat = self.model(
                x_t_next_zero_sc,
                t_next,
                shortcut_size,
            ).detach()

        y_hat = self._modify_model_input(input_ids_mask, x_start, y_hat)
        y_hat = torch.cat((x_t, y_hat), dim=-1)
        shortcut_size = torch.zeros_like(t) if self.training_cfg.model.default_shortcut == "0" else t
        return self.model(y_hat, t, shortcut_size)

    def _should_apply_self_conditioning(self) -> Tensor:
        """Determines whether to apply self-conditioning based on the self_conditioning_ratio."""
        return torch.rand(1) < self.self_conditioning_ratio

    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Tensor | None = None) -> Tensor:
        return self.criterion._modify_model_input(input_ids_mask.unsqueeze(-1), x_start, y_hat)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return self.criterion._compute_target(batch)

    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        return self.criterion.get_x0_from_predicition(y_hat, batch)

    @override
    def infere_model(self, x_t: Tensor, t: Tensor, shortcut_size: Tensor, input_mask: Tensor) -> Tensor:
        """Adds self-conditioning part to the model's input. Call the model and resotre input part of the predicition"""
        if self.y_hat is None:
            self.y_hat = torch.zeros_like(x_t)

        x_t = torch.cat((x_t, self.y_hat), -1)
        model_output = self.model(x_t, t, shortcut_size)
        model_output = self._restore_input_part(model_output, x_t, input_mask)

        # store for self-conditioning
        self.y_hat = model_output
        return model_output

    @override
    def compute_velocity(
        self,
        model_output: Tensor,
        noise: Tensor,
        input_mask: Tensor,
    ) -> Tensor:
        """COmputes velocity from models output"""
        return self.criterion.compute_velocity(model_output, noise, input_mask)

    @override
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        _, _, dim = model_output.shape
        x_t = x_t[:, :, :dim]
        return self.criterion._restore_input_part(model_output, x_t, input_mask)

    @override
    def _reset(self):
        """Resets self-conditioning storage for new batch of data"""
        self.y_hat = None


class ConsistencyCriterion(Criterion, ABC):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
    ):
        super().__init__(model=model, diffusion_steps=diffusion_steps, training_cfg=training_cfg)
        self.reduce_fn = reduce_fn
        self.loss_fn = loss_fn

    @override
    def compute_losses(self, batch: ShortcutFMBatch, world_size) -> dict[str, Tensor]:
        target = self._compute_shortcut_target(
            shortcut_size=batch.shortcut_size,
            t=batch.t,
            x_t=batch.x_t,
            x_start=batch.x_start,
            input_ids_mask=batch.input_ids_mask,
            noise=batch.noise,
        )
        output = self._predict(
            x_start=batch.x_start,
            x_t=batch.x_t,
            noise=batch.noise,
            t=batch.t,
            shortcut_size=batch.shortcut_size,
            input_ids_mask=batch.input_ids_mask,
        )

        loss = self.loss_fn(output, target)
        return {"consistency_loss": self.reduce_fn(loss, dim=-1)}

    @torch.no_grad()
    def _compute_shortcut_target(
        self,
        *,
        shortcut_size: Tensor,
        t: Tensor,
        x_t: Tensor,
        x_start: Tensor,
        input_ids_mask: Tensor,
        noise: Tensor,
    ):
        # Check if we should use the direct target (x_start for X0, velocity for Velocity)
        if self._should_use_direct_target():
            return self._get_direct_target(x_start, noise, input_ids_mask)

        # Otherwise, use the original two-step computation
        input_ids_mask = input_ids_mask.unsqueeze(-1).expand_as(x_t)
        step1_prediction = self.model(x_t, t, shortcut_size)

        step2_input = self._prepare_2_shortcut_input(
            step1_prediction,
            x_start,
            x_t,
            t,
            shortcut_size,
            input_ids_mask,
            noise=noise,
        )
        step2_prediction = self.model(step2_input, t - shortcut_size, shortcut_size)

        target = self._modify_target(
            step1_prediction,
            step2_prediction,
            x_start,
            x_t,
            t,
            shortcut_size,
            input_ids_mask,
            step2_input,
        )
        return target.detach()

    def _should_use_direct_target(self) -> bool:
        """Determine whether to use direct target based on probability.

        Returns:
            bool: True if direct target should be used, False otherwise
        """
        if self.training_cfg is None:
            return False

        probability = self.training_cfg.shortcut_target_x_start_probability
        if probability <= 0.0:
            return False

        # Draw a random number and check if it's less than the probability
        random_value = torch.rand(1).item()
        return random_value < probability

    @abstractmethod
    def _get_direct_target(self, x_start: Tensor, noise: Tensor, input_ids_mask: Tensor) -> Tensor:
        """Get the direct target (x_start for X0, velocity for Velocity) when using probability-based targeting."""

    @abstractmethod
    def _prepare_2_shortcut_input(
        self,
        step1_prediction: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
        noise: Tensor,
    ) -> Tensor:
        """Prepares input for the second shorcut step of size d based on the first shorcut prediction"""

    @abstractmethod
    def _modify_target(
        self,
        step1_prediction: Tensor,
        step2_prediction: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
        step2_input: Tensor,
    ) -> Tensor:
        """Modifies target based on two shorcuts predicitons"""

    def _predict(
        self,
        *,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        """Compute the model output."""
        y = self.model(x_t, t, 2 * shortcut_size)
        trg = torch.where(input_ids_mask.unsqueeze(-1) == 0, 0, y)
        return trg

    @abstractmethod
    def _modify_model_input_or_output(
        self,
        input_ids_mask: Tensor,
        x_start: Tensor,
        y_hat: Tensor | None = None,
    ) -> Tensor:
        """Modify model input based on input_ids_mask. Used for self-conditioning."""


class X0ConsistencyCriterion(ConsistencyCriterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
    ):
        super().__init__(
            model,
            diffusion_steps,
            reduce_fn,
            training_cfg=training_cfg,
            loss_fn=loss_fn,
        )

    @override
    def _predict(
        self,
        *,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        """Compute the model output."""
        y = self.model(x_t, t, 2 * shortcut_size)
        return y

    @override
    def _prepare_2_shortcut_input(
        self,
        step1_prediction,
        x_start,
        x_t,
        t,
        shortcut_size,
        input_ids_mask: Tensor,
        noise: Tensor,
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        x_t = x_t[..., :embedding_dim]

        v_hat = step1_prediction - noise

        step_size = (shortcut_size / self.diffusion_steps)[:, None, None]
        step2_input = x_t + step_size * v_hat
        step2_input = torch.where(input_ids_mask == 0, x_start, step2_input)
        return step2_input

    @override
    def _get_direct_target(self, x_start: Tensor, noise: Tensor, input_ids_mask: Tensor) -> Tensor:
        """Return x_start as the direct target for X0 consistency criterion."""
        return x_start

    @override
    def _modify_target(
        self,
        step1_prediction,
        step2_prediction,
        x_start,
        x_t,
        _,
        shortcut_size,
        input_ids_mask: Tensor,
        step2_input: Tensor,
    ):
        embedding_dim = step2_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]

        target = (step1_prediction + step2_prediction) / 2
        target = torch.where(input_ids_mask == 0, x_start, target)
        return target

    @override
    def _modify_model_input_or_output(
        self,
        input_ids_mask: Tensor,
        x_start: Tensor,
        y_hat: Tensor | None = None,
    ) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)


class VelocityConsistencyCriterion(ConsistencyCriterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        reduce_fn: Callable = torch.mean,
        training_cfg: TrainingConfig = None,
        loss_fn: Callable = None,
    ):
        super().__init__(model, diffusion_steps, reduce_fn, training_cfg, loss_fn)

    @override
    def _get_direct_target(self, x_start: Tensor, noise: Tensor, input_ids_mask: Tensor) -> Tensor:
        """Return velocity (x_start - noise) as the direct target for Velocity consistency criterion."""
        input_ids_mask = input_ids_mask.unsqueeze(-1).expand_as(x_start)
        # For Velocity criterion, the direct target is the velocity (x_start - noise)
        velocity = x_start - noise
        # For input tokens, velocity should be 0 (no change needed)
        target = torch.where(input_ids_mask == 0, torch.zeros_like(velocity), velocity)
        return target.detach()

    @override
    def _prepare_2_shortcut_input(
        self,
        velocity,
        x_start,
        x_t,
        t,
        shorcut_size,
        input_ids_mask: Tensor,
        noise: Tensor,
    ):
        t = self.scale_t(t).view(-1, 1, 1)
        embedding_dim = velocity.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_t = x_t[..., :embedding_dim]
        velocity = torch.where(input_ids_mask == 0, 0, velocity)
        step_size = (shorcut_size / self.diffusion_steps)[:, None, None]
        return x_t + velocity * step_size

    @override
    def _modify_target(
        self,
        step1_prediction,
        step2_prediction,
        x_start,
        _,
        __,
        ___,
        input_ids_mask: Tensor,
        step2_input: Tensor,
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        target = (step1_prediction + step2_prediction) / 2
        target = torch.where(input_ids_mask == 0, 0, target)
        return target

    @override
    def _modify_model_input_or_output(
        self,
        input_ids_mask: Tensor,
        x_start: Tensor,
        y_hat: Tensor | None = None,
    ) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)


class ConsistencyCriterionDecorator(ConsistencyCriterion, ABC):
    def __init__(
        self,
        criterion: ConsistencyCriterion,
    ):
        super().__init__(
            criterion.model,
            criterion.diffusion_steps,
            criterion.reduce_fn,
            criterion.training_cfg,
            criterion.loss_fn,
        )
        self._criterion = criterion


class SelfConditioningConsistencyCriterionDecorator(ConsistencyCriterionDecorator):
    def __init__(
        self,
        criterion: ConsistencyCriterion,
        self_conditioning_ratio: float,
    ):
        super().__init__(criterion)
        self.self_conditioning_ratio = self_conditioning_ratio

        # Store original methods
        self._original_modify_first_step_prediction = criterion._prepare_2_shortcut_input

        # Override methods dynamically with wrapped versions
        self._criterion._prepare_2_shortcut_input = self._wrapped_prepare_2_shortcut_input

    def _wrapped_prepare_2_shortcut_input(
        self,
        step1_prediction: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        original_result = self._original_modify_first_step_prediction(
            step1_prediction,
            x_start,
            x_t,
            t,
            shortcut_size,
            input_ids_mask,
        )
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_0_hat = self._modify_model_input_or_output(input_ids_mask, x_start)
        return torch.cat((original_result, x_0_hat), dim=-1)

    @override
    @torch.no_grad()
    def _compute_shortcut_target(
        self,
        *,
        shortcut_size: Tensor,
        t: Tensor,
        x_t: Tensor,
        x_start: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        x_0_hat = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start)
        x_t = torch.cat((x_t, x_0_hat), dim=-1)
        target = self._criterion._compute_shortcut_target(
            shortcut_size=shortcut_size,
            x_start=x_start,
            x_t=x_t,
            t=t,
            input_ids_mask=input_ids_mask,
        )
        return target

    @override
    def _predict(
        self,
        x_start: Tensor,
        x_t: Tensor,
        noise: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
    ) -> Tensor:
        """Compute the model output."""
        if not self._should_apply_self_conditioning():
            x_0_hat = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start)
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            y = self._criterion._predict(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask,
            )
            return y

        # TODO: handle it better
        t_next = torch.where(
            t + shortcut_size <= self.diffusion_steps,
            t + shortcut_size,
            self.diffusion_steps,
        )
        x_t_next, noise = self._interpolate_data_noise(x_start, t_next)
        x_t_next = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, x_t_next)
        empty_self_conditioning_input = self._criterion._modify_model_input_or_output(
            input_ids_mask.unsqueeze(-1),
            x_start,
        )
        x_t_next_zero_sc = torch.cat((x_t_next, empty_self_conditioning_input), dim=-1)

        with torch.no_grad():
            y_hat = self.model(x_t_next_zero_sc, t_next, 2 * shortcut_size).detach()
        y_hat = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start, y_hat)

        x_t_sc = torch.cat((x_t, y_hat), dim=-1)
        y = self.model(x_t_sc, t, 2 * shortcut_size)

        y = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start, y)
        trg = torch.where(input_ids_mask.unsqueeze(-1) == 0, 0, y - x_t)
        return trg

    def _should_apply_self_conditioning(self) -> Tensor:
        """Determines whether to apply self-conditioning based on the self_conditioning_ratio.

        :returns:True if self-conditioning should be applied, False otherwise.
        :rtype: bool
        """
        return torch.rand(1) < self.self_conditioning_ratio

    def _modify_model_input_or_output(
        self,
        input_ids_mask: Tensor,
        x_start: Tensor,
        y_hat: Tensor | None = None,
    ) -> Tensor:
        return self._criterion._modify_model_input_or_output(input_ids_mask, x_start, y_hat)

    def _prepare_2_shortcut_input(
        self,
        step1_prediction: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
        noise: Tensor,
    ) -> Tensor:
        raise NotImplementedError("This method should not be called on decorator")

    def _modify_target(
        self,
        step1_prediction: Tensor,
        step2_prediction: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_ids_mask: Tensor,
        step2_input: Tensor,
    ) -> Tensor:
        raise NotImplementedError("This method should not be called on decorator")


class NllCriterion(Criterion):
    def __init__(
        self,
        model: Model,
        diffusion_steps,
        training_cfg: TrainingConfig = None,
    ):
        super().__init__(model, diffusion_steps, training_cfg)

    @override
    def forward(self, batch: FlowMatchingBatch, world_size) -> dict[str, Tensor]:
        output = self.model.compute_logits(batch.x_start)
        loss = (
            torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), batch.seqs.view(-1), reduction="none")
            .view(batch.seqs.size())
            .mean(-1)
        )

        return {"nll_loss": loss}

    @override
    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        raise NotImplementedError("Embedding loss should not be masked")


class IsotropyCriterion(Criterion):
    def forward(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        return self.compute_losses(batch, world_size)

    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        embedding_weights = self.model.module.word_embedding.weight
        return {"isotropy_loss": isotropy_loss(embedding_weights)}


def isotropy_loss(embeddings):
    norm_emb = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    cos_sim = torch.mm(norm_emb, norm_emb.T)
    off_diag = cos_sim - torch.eye(cos_sim.size(0), device=cos_sim.device)
    return torch.mean(off_diag**2)


class CompositeCriterion(Criterion):
    def __init__(
        self,
        flow_matching_criterion: FlowMatchingCriterion,
        consistency_criterion: ConsistencyCriterion,
        embedding_criterion: NllCriterion,
        flow_matching_weight: float,
        consistency_weight: float,
        embedding_weight: float,
        model: Model,
        diffusion_steps: int,
        self_consistency_ratio: float,
        sampler: ScheduleSampler,
        time_shortcut_sampler: TimeAndShortcutSampler,
        training_cfg: TrainingConfig = None,
    ):
        super().__init__(model, diffusion_steps, training_cfg=training_cfg)
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.flow_matching_criterion = flow_matching_criterion
        self.consistency_criterion = consistency_criterion
        self.embedding_criterion = embedding_criterion
        self.flow_matching_weight = flow_matching_weight
        self.consistency_weight = consistency_weight
        self.embedding_weight = embedding_weight
        self.criteria_weights = [
            flow_matching_weight,
            consistency_weight,
            embedding_weight,
        ]
        self.self_consistency_ratio = self_consistency_ratio
        self.sampler = sampler
        self.time_shortcut_sampler = time_shortcut_sampler

    def forward(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        return self.compute_losses(batch, world_size)

    @override
    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        self.global_step = batch.global_step
        self.flow_matching_criterion.global_step = batch.global_step

        specific_batches, weights = self._prepare_batches(batch)
        full_batch, consistency_batch = specific_batches
        fm_weights, consistency_weights = weights

        # Use full batch for flow matching and embedding losses
        flow_and_decoder_loses = self.flow_matching_criterion(full_batch, world_size)
        flow_matching_loss = flow_and_decoder_loses["flow_matching_loss"]
        decoder_loss = flow_and_decoder_loses["decoder_loss"]
        embedding_loss = self.embedding_criterion(full_batch, world_size)["nll_loss"]

        # Check if consistency loss is enabled
        if consistency_batch is not None:
            consistency_loss = self.consistency_criterion(consistency_batch, world_size)["consistency_loss"]
            losses = [
                flow_matching_loss,
                consistency_loss,
                embedding_loss,
            ]  # no decoder_loss

            # update sampler with consistency loss
            self.time_shortcut_sampler.update_with_local_losses(
                consistency_batch.t - 1,
                consistency_loss.detach(),
                world_size=world_size,
            )
            result = {
                "flow_matching_loss": flow_matching_loss * self.criteria_weights[0],
                "consistency_loss": consistency_loss * self.criteria_weights[1],
                "embedding_loss": embedding_loss * self.criteria_weights[2],
                "decoder_loss": decoder_loss,
                "timestep": full_batch.t,
                "shortcut": consistency_batch.shortcut_size,
            }
        else:
            # If consistency is not enabled, only use flow matching and embedding losses
            losses = [
                flow_matching_loss,
                embedding_loss,
            ]  # no decoder_loss or consistency_loss
            # Use only the first and third weights (flow matching and embedding)
            criteria_weights = [self.criteria_weights[0], self.criteria_weights[2]]
            result = {
                "flow_matching_loss": flow_matching_loss * self.criteria_weights[0],
                "embedding_loss": embedding_loss * self.criteria_weights[2],
                "decoder_loss": decoder_loss,
                "timestep": full_batch.t,
            }

        # weight the losses with fm_weights and consistency_weights get from loss aware samplers
        losses[0] *= fm_weights
        if consistency_batch is not None:
            losses[1] *= consistency_weights
            losses[2] *= fm_weights  # embedding loss
        else:
            losses[1] *= fm_weights  # embedding loss

        weighted_losses = [
            loss.mean() * weight
            for loss, weight in zip(
                losses,
                self.criteria_weights if consistency_batch is not None else criteria_weights,
                strict=True,
            )
        ]

        total_loss = sum(weighted_losses)
        result["loss"] = total_loss

        if isinstance(self.sampler, LossAwareSampler):
            total_loss_per_sample = flow_matching_loss
            self.sampler.update_with_local_losses(
                full_batch.t - 1,
                total_loss_per_sample.detach(),
                world_size=world_size,
            )

        return result

    def _prepare_batches(
        self,
        batch: EncoderBatch,
    ) -> tuple[tuple[FlowMatchingBatch, ShortcutFMBatch | None], tuple[Any, Any | None]]:
        """Prepare batches for composite criterion.

        Returns:
            - full_batch: FlowMatchingBatch for the entire input (used for flow matching)
            - consistency_batch: Optional ShortcutFMBatch for consistency loss (subset of full batch)
            - fm_weights: Weights from flow matching sampler
            - consistency_weights: Weights from consistency sampler (None if consistency disabled)
        """
        bsz = batch.size()

        use_consistency = (
            self.global_step >= self.training_cfg.consistency_start_step
            and self.training_cfg.self_consistency_ratio > 0
        )

        # Always prepare a full batch for flow matching
        embeddings = self.model.get_embeddings(batch.seqs)
        t, fm_weights = self.sampler(batch_size=bsz, device=batch.seqs.device)
        x_t, noise = self._interpolate_data_noise(embeddings, t)
        x_t = torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, embeddings, x_t)

        full_batch = FlowMatchingBatch(
            seqs=batch.seqs,
            padding_mask=batch.padding_mask,
            input_ids_mask=batch.input_ids_mask,
            x_start=embeddings,
            x_t=x_t,
            noise=noise,
            t=t,
            global_step=batch.global_step,
        )

        # If consistency is not enabled, return None for consistency_batch
        if not use_consistency:
            return (full_batch, None), (fm_weights, None)

        # prepare_consistency_batch (only if consistency is enabled)
        num_consistency_elems = int(self.self_consistency_ratio * bsz)

        # Use a subset of the batch for consistency
        consistency_seqs = batch.seqs[:num_consistency_elems]
        consistency_x_start = embeddings[:num_consistency_elems]
        consistency_padding_mask = batch.padding_mask[:num_consistency_elems]
        consistency_input_ids_mask = batch.input_ids_mask[:num_consistency_elems]
        consistency_t, shortcuts, consistency_weights = self.time_shortcut_sampler(
            batch_size=num_consistency_elems,
            device=batch.seqs.device,
        )
        consistency_x_t, consistency_noise = self._interpolate_data_noise(consistency_x_start, consistency_t)
        consistency_x_t = torch.where(
            consistency_input_ids_mask.unsqueeze(-1) == 0,
            consistency_x_start,
            consistency_x_t,
        )
        consistency_batch = ShortcutFMBatch(
            seqs=consistency_seqs,
            padding_mask=consistency_padding_mask,
            input_ids_mask=consistency_input_ids_mask,
            x_start=consistency_x_start,
            x_t=consistency_x_t,
            noise=consistency_noise,
            t=consistency_t,
            shortcut_size=shortcuts,
            global_step=batch.global_step,
        )

        return (full_batch, consistency_batch), (fm_weights, consistency_weights)

    def denoise(
        self,
        batch: EncoderBatch,
        shortcut_size: int | None = None,
        probe_every_step: bool = True,
        return_decoded: bool = False,
        return_logits: bool = False,
        step_size: int | None = None,
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        """Denoises batch of examples with flexible probing and output options.

        :param batch: Batch of examples to denoise
        :type batch: EncoderBatch
        :param shortcut_size: shortcut size to use during denoising. If None or 0, step_size must be provided
        :type shortcut_size: Optional[int]
        :param probe_every_step: whether to probe at every step or only at the final step
        :type probe_every_step: bool
        :param return_decoded: whether to return decoded sequences or token IDs
        :type return_decoded: bool
        :param return_logits: whether to return logits instead of token IDs
        :type return_logits: bool
        :param step_size: step size to use during denoising when shortcut_size is None or 0
        :type step_size: Optional[int]

        :returns: One of the following based on parameters:
            - If return_logits=True:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len, vocab_size] with logits
                - If probe_every_step=False: Tensor[batch_size, seq_len, vocab_size] with logits
            - If return_decoded=True:
                - If probe_every_step=True: List[List[str]] where outer list is batches, inner list is steps
                - If probe_every_step=False: List[str] of decoded sequences
            - Otherwise:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len] with token IDs
                - If probe_every_step=False: Tensor[batch_size, seq_len] with token IDs
        :rtype: Union[Tensor, list[list[str]]]
        """
        return self.flow_matching_criterion.denoise(
            batch,
            shortcut_size=shortcut_size,
            probe_every_step=probe_every_step,
            return_decoded=return_decoded,
            return_logits=return_logits,
            step_size=step_size,
        )


class FlowNllCriterion(Criterion):
    def __init__(
        self,
        flow_matching_criterion: FlowMatchingCriterion,
        nll_criterion: NllCriterion,
        model: Model,
        diffusion_steps,
        sampler: ScheduleSampler,
        training_cfg: TrainingConfig = None,
    ):
        super().__init__(model, diffusion_steps, training_cfg=training_cfg)
        self.flow_matching_criterion = flow_matching_criterion
        self.nll = nll_criterion
        self.sampler = sampler

    def forward(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        return self.compute_losses(batch, world_size)

    @override
    def compute_losses(self, batch: EncoderBatch, world_size) -> dict[str, Tensor]:
        fm_batch = self._prepare_batch(batch)

        flow_and_decoder_loses = self.flow_matching_criterion(fm_batch, world_size)
        flow_matching_loss = flow_and_decoder_loses["flow_matching_loss"]
        decoder_loss = flow_and_decoder_loses["decoder_loss"]
        embedding_loss = self.nll(fm_batch, world_size)["nll_loss"]

        losses = [flow_matching_loss.mean(), embedding_loss.mean()]  # no decoder_loss
        total_loss = sum(losses)

        return {
            "flow_matching_loss": flow_matching_loss,
            "embedding_loss": embedding_loss,
            "decoder_loss": decoder_loss,
            "loss": total_loss,
            "timestep": fm_batch.t,
        }

    def _prepare_batch(self, batch: EncoderBatch) -> FlowMatchingBatch:
        bsz = batch.size()

        embeddings = self.model.get_embeddings(batch.seqs)
        t, _ = self.sampler(batch_size=bsz, device=batch.seqs.device)
        x_t, noise = self._interpolate_data_noise(embeddings, t)
        x_t = torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, embeddings, x_t)

        return FlowMatchingBatch(
            seqs=batch.seqs,
            padding_mask=batch.padding_mask,
            input_ids_mask=batch.input_ids_mask,
            x_start=embeddings,
            x_t=x_t,
            noise=noise,
            t=t,
            global_step=batch.global_step,
        )

    def denoise(
        self,
        batch: EncoderBatch,
        shortcut_size: int | None = None,
        probe_every_step: bool = True,
        return_decoded: bool = False,
        return_logits: bool = False,
        step_size: int | None = None,
    ) -> np.ndarray[str, np.dtype[str]]:
        """Denoises batch of examples with flexible probing and output options.

        :param batch: batch of examples to denoise
        :type batch: EncoderBatch
        :param shortcut_size: shortcut size to use during denoising. If None or 0, step_size must be provided
        :type shortcut_size: Optional[int]
        :param probe_every_step: whether to probe at every step or only at the final step
        :type probe_every_step: bool
        :param return_decoded: whether to return decoded sequences or token IDs
        :type return_decoded: bool
        :param return_logits: whether to return logits instead of token IDs
        :type return_logits: bool
        :param step_size: step size to use during denoising when shortcut_size is None or 0
        :type step_size: Optional[int]

        :returns: One of the following based on parameters:
            - If return_logits=True:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len, vocab_size] with logits
                - If probe_every_step=False: Tensor[batch_size, seq_len, vocab_size] with logits
            - If return_decoded=True:
                - If probe_every_step=True: List[List[str]] where outer list is batches, inner list is steps
                - If probe_every_step=False: List[str] of decoded sequences
            - Otherwise:
                - If probe_every_step=True: Tensor[batch_size, num_steps, seq_len] with token IDs
                - If probe_every_step=False: Tensor[batch_size, seq_len] with token IDs
        :rtype: Union[Tensor, list[list[str]]]
        """
        return self.flow_matching_criterion.denoise(
            batch,
            shortcut_size=shortcut_size,
            probe_every_step=probe_every_step,
            return_decoded=return_decoded,
            return_logits=return_logits,
            step_size=step_size,
        )
