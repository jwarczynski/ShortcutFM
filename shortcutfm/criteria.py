from abc import ABC, abstractmethod
from itertools import zip_longest
from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerBase
from typing_extensions import Optional, override

from shortcutfm.batch import EncoderBatch, FlowMatchingBatch, ShortcutFMBatch
from shortcutfm.model.model import FlowMatchingModel as Model
from shortcutfm.shortcut_samplers import ScheduleSampler, TimeAndShortcutSampler


class Criterion(Module, ABC):
    def __init__(self, model: Model, diffusion_steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.diffusion_steps = diffusion_steps

    def forward(self, batch: EncoderBatch) -> dict[str, Tensor]:
        """ Compute the losses. """
        return self.losses_with_mask(batch)

    def losses_with_mask(self, batch: EncoderBatch) -> dict[str, Tensor]:
        """ Compute the losses applying mask. """
        padding_mask = batch.padding_mask
        input_ids_mask = batch.input_ids_mask

        losses = self.compute_losses(batch)
        loss_mask = padding_mask * input_ids_mask
        for key, value in losses.items():
            masked_per_token_loss = loss_mask * value
            per_batch_loss = masked_per_token_loss.sum(-1) / loss_mask.sum(-1)
            losses[key] = per_batch_loss
        return losses

    @abstractmethod
    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        """ Compute the losses. """

    def _interpolate_data_noise(
            self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
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
    ):
        super().__init__(model, diffusion_steps)
        self.tokenizer = tokenizer
        self.x_t = None
        self.reduce_fn = reduce_fn

    def compute_losses(self, batch: FlowMatchingBatch) -> dict[str, Tensor]:
        target = self._compute_target(batch)
        output = self._predict(
            x_start=batch.x_start,
            x_t=batch.x_t,
            noise=batch.noise,
            t=batch.t,
            input_ids_mask=batch.input_ids_mask,
        )

        fm_loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        x_start_predicted = self.get_x0_from_predicition(output, batch)
        decoder_loss = self._compute_nll_loss(x_start_predicted, batch.seqs)

        return {
            "flow_matching_loss": self.reduce_fn(fm_loss, dim=-1),
            "decoder_loss": decoder_loss
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
        """ Compute the model output. """
        y = self.model(x_t, t, torch.zeros_like(t))
        return y

    def _compute_nll_loss(self, hidden_last: Tensor, seqs: Tensor) -> Tensor:
        logits = self.model.compute_logits(hidden_last)
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            seqs.view(-1),
            reduction="none"
        ).view(seqs.size())

    def denoise(
            self,
            batch: EncoderBatch,
            shortcut_size: Optional[int] = None,
            probe_every_step: bool = True,
            return_decoded: bool = False,
            return_logits: bool = False,
            step_size: Optional[int] = None
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        """
        Denoises batch of examples with flexible probing and output options.

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

        # Use step_size if shortcut_size is None or 0
        effective_step = step_size if (shortcut_size is None or shortcut_size == 0) else shortcut_size
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
                    (batch.seqs.shape[0], num_steps, batch.seqs.shape[1], self.model.vocab_size),
                    dtype=torch.float,
                    device=batch.seqs.device
                )
            else:
                predictions = torch.zeros(
                    (batch.seqs.shape[0], num_steps, batch.seqs.shape[1]),
                    dtype=torch.long,
                    device=batch.seqs.device
                )

        shortcuts = torch.tensor(shortcut_size, device=input_mask.device).repeat(input_mask.shape[0])
        for step_idx, t in enumerate(torch.arange(self.diffusion_steps, 0, -effective_step, device=input_mask.device)):
            t: Tensor = t.repeat(input_mask.shape[0])
            model_output = self.infere_model(
                self.x_t,
                t,
                shortcuts,
                input_mask
            )
            v_hat = self.compute_velocity(
                self.x_t,
                model_output,
                t,
                shortcuts,
                input_mask
            )
            x0_hat = self.x_t + (effective_step / self.diffusion_steps) * v_hat
            self.x_t = x0_hat

            # Get predictions if probing every step or if this is the last step
            if probe_every_step or step_idx == num_steps - 1:
                step_predictions = self.probe(x0_hat, return_logits=return_logits)
                if probe_every_step:
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
                return [decoded[i:i + num_steps] for i in range(0, len(decoded), num_steps)]
            else:
                return self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        return predictions

    def infere_model(self, x_t: Tensor, t: Tensor, shortcut_size: Tensor, input_mask: Tensor) -> Tensor:
        """Call the model and resotre input part of the pediction"""
        model_output = self.model(x_t, t, shortcut_size)
        return self._restore_input_part(model_output, x_t, input_mask)

    @abstractmethod
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        """recover input part of the prediction based on input_mask"""

    @abstractmethod
    def compute_velocity(
            self,
            x_t,
            model_output: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_mask: Tensor
    ) -> Tensor:
        """computes velocity based on models output for the denoising process"""

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
        """
        Allow subclasses to prepare for new batch of examples.

        For example, it can reset stored conditioning values.
        """
        pass

    @abstractmethod
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        """ Compute the target. """

    @abstractmethod
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        """ Modify model input based on input_ids_mask. Used for self-conditioning. """

    @abstractmethod
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        """ Extract x0 from the model prediction. """


class X0FlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
            self,
            model: Model,
            diffusion_steps,
            tokenizer: PreTrainedTokenizerBase,
            reduce_fn: Callable = torch.mean,
    ):
        super().__init__(model, diffusion_steps, tokenizer, reduce_fn)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start

    @override
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)

    @override
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        return torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, batch.x_start, y_hat)

    @override
    def compute_velocity(
            self,
            x_t,
            x0_hat: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_mask: Tensor
    ) -> Tensor:
        v_hat = x0_hat - x_t
        assert torch.all(v_hat[input_mask.expand_as(v_hat) == 0] == 0), "v_hat is not zero where input_mask is zero"
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
    ):
        super().__init__(model, diffusion_steps, tokenizer, reduce_fn)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start - batch.noise

    @override
    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)

    @override
    def get_x0_from_predicition(self, y_hat: Tensor, batch: FlowMatchingBatch) -> Tensor:
        x0 = batch.x_t + y_hat
        return torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, batch.x_start, x0)

    @override
    def compute_velocity(
            self,
            x_t,
            v_hat: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_mask: Tensor
    ) -> Tensor:
        v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)
        return v_hat

    @override
    def _restore_input_part(self, v_hat: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        return torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)


class FlowMatchinCriterionDecorator(FlowMatchingCriterion, ABC):
    def __init__(
            self,
            criterion: FlowMatchingCriterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps, criterion.tokenizer, criterion.reduce_fn)
        self.criterion = criterion


class SelfConditioningFlowMatchingCriterionDecorator(FlowMatchinCriterionDecorator):

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
        """ Compute the model output. """
        # prepare self-conditioning input
        x_0_hat = self._modify_model_input(input_ids_mask, x_start)

        if not self._should_apply_self_conditioning():
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion._predict(
                x_start=x_start, x_t=x_t, noise=noise, t=t, input_ids_mask=input_ids_mask
            )

        # TODO: model won't be quried at t + 1 during inference, but at t + d
        # TODO: draw a shorcut value from distribution?
        t_next = torch.where(t < self.diffusion_steps, t + 1, t)
        x_t_next, _ = self._interpolate_data_noise(x_start, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            y_hat = self.model(
                x_t_next_zero_sc,
                t_next,
                torch.zeros_like(t_next, device=t_next.device)
            ).detach()

        y_hat = self._modify_model_input(input_ids_mask, x_start, y_hat)
        y_hat = torch.cat((x_t, y_hat), dim=-1)
        return self.model(y_hat, t, torch.zeros_like(t))

    def _should_apply_self_conditioning(self) -> Tensor:
        """ Determines whether to apply self-conditioning based on the self_conditioning_ratio. """
        return torch.rand(1) < self.self_conditioning_ratio

    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
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
            x_t,
            model_output: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_mask: Tensor
    ) -> Tensor:
        """COmputes velocity from models output"""
        return self.criterion.compute_velocity(x_t, model_output, t, shortcut_size, input_mask)

    @override
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        _, _, dim = model_output.shape
        x_t = x_t[:, :, :dim]
        return self.criterion._restore_input_part(model_output, x_t, input_mask)

    @override
    def _reset(self):
        """Resets self-conditioning storage for new batch of data"""
        self.y_hat = None


class ConsistencyCrterion(Criterion, ABC):
    def __init__(
            self,
            model: Model,
            diffusion_steps,
            reduce_fn: Callable = torch.mean,
    ):
        super().__init__(model, diffusion_steps)
        self.reduce_fn = reduce_fn

    @override
    def compute_losses(self, batch: ShortcutFMBatch) -> dict[str, Tensor]:
        target = self._compute_shortcut_target(
            shortcut_size=batch.shortcut_size,
            t=batch.t,
            x_t=batch.x_t,
            x_start=batch.x_start,
            input_ids_mask=batch.input_ids_mask,
        )
        output = self._predict(
            x_start=batch.x_start,
            x_t=batch.x_t,
            noise=batch.noise,
            t=batch.t,
            shortcut_size=batch.shortcut_size,
            input_ids_mask=batch.input_ids_mask,
        )

        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        return {
            "consistency_loss": self.reduce_fn(loss, dim=-1)
        }

    @torch.no_grad()
    def _compute_shortcut_target(
            self,
            *,
            shortcut_size: Tensor,
            t: Tensor,
            x_t: Tensor,
            x_start: Tensor,
            input_ids_mask: Tensor,
    ):
        input_ids_mask = input_ids_mask.unsqueeze(-1).expand_as(x_t)
        step1_prediction = self.model(x_t, t, shortcut_size)

        step2_input = self._prepare_2_shortcut_input(step1_prediction, x_start, x_t, t, shortcut_size, input_ids_mask)
        step2_prediction = self.model(step2_input, t - shortcut_size, shortcut_size)

        target = self._modify_target(step1_prediction, step2_prediction, x_start, x_t, t, shortcut_size, input_ids_mask)
        return target.detach()

    @abstractmethod
    def _prepare_2_shortcut_input(
            self,
            step1_prediction: Tensor,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """ Prepares input for the second shorcut step of size d based on the first shorcut prediction"""

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
    ) -> Tensor:
        """ Modifies target based on two shorcuts predicitons """

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
        """ Compute the model output. """

        y = self.model(x_t, t, 2 * shortcut_size)
        y = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y)
        return y

    @abstractmethod
    def _modify_model_input_or_output(
            self,
            input_ids_mask: Tensor,
            x_start: Tensor,
            y_hat: Optional[Tensor] = None
    ) -> Tensor:
        """ Modify model input based on input_ids_mask. Used for self-conditioning. """


class X0ConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            diffusion_steps,
            reduce_fn: Callable = torch.mean,
    ):
        super().__init__(model, diffusion_steps, reduce_fn)

    @override
    def _prepare_2_shortcut_input(
            self, step1_prediction, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        x_t = x_t[..., :embedding_dim]
        step2_input =  x_t + (shorcut_size / self.diffusion_steps)[:, None, None] * step1_prediction
        step2_input = torch.where(input_ids_mask == 0, x_start, step2_input)
        return step2_input

    @override
    def _modify_target(
            self, _, step2_prediction, x_start, x_t, __, ___, input_ids_mask: Tensor
    ):
        embedding_dim = step2_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        target = torch.where(input_ids_mask == 0, x_start, step2_prediction)
        return target

    @override
    def _modify_model_input_or_output(
            self,
            input_ids_mask: Tensor,
            x_start: Tensor,
            y_hat: Optional[Tensor] = None
    ) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)


class VelocityConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            diffusion_steps,
            reduce_fn: Callable = torch.mean,
    ):
        super().__init__(model, diffusion_steps, reduce_fn)

    @override
    def _prepare_2_shortcut_input(
            self, velocity, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        t = self.scale_t(t).view(-1, 1, 1)
        embedding_dim = velocity.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_t = x_t[..., :embedding_dim]
        velocity = torch.where(input_ids_mask == 0, 0, velocity)
        return x_t + velocity * t

    @override
    def _modify_target(
            self, step1_prediction, step2_prediction, x_start, _, __, ___, input_ids_mask: Tensor
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        target =  (step1_prediction + step2_prediction) / 2
        target = torch.where(input_ids_mask == 0, 0, target)
        return target

    @override
    def _modify_model_input_or_output(
            self,
            input_ids_mask: Tensor,
            x_start: Tensor,
            y_hat: Optional[Tensor] = None
    ) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)


class ConsistencyCriterionDecorator(ConsistencyCrterion, ABC):
    def __init__(
            self,
            criterion: ConsistencyCrterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps, criterion.reduce_fn)
        self._criterion = criterion


class SelfConditioningConsistencyCriterionDecorator(ConsistencyCriterionDecorator):

    def __init__(
            self,
            criterion: ConsistencyCrterion,
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
            step1_prediction, x_start, x_t, t, shortcut_size, input_ids_mask
        )
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_0_hat = self._modify_model_input_or_output(input_ids_mask, x_start, step1_prediction)
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
            input_ids_mask=input_ids_mask
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
        """ Compute the model output. """
        if not self._should_apply_self_conditioning():
            x_0_hat = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start)
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            y = self._criterion._predict(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask
            )
            return y

        # TODO: handle it better
        t_next = torch.where(t + shortcut_size <= self.diffusion_steps, t + shortcut_size, t)
        x_t_next, noise = self._interpolate_data_noise(x_start, t_next)
        x_t_next = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, x_t_next)
        empty_self_conditioning_input = self._criterion._modify_model_input_or_output(
            input_ids_mask.unsqueeze(-1),
            x_start
        )
        x_t_next_zero_sc = torch.cat((x_t_next, empty_self_conditioning_input), dim=-1)

        with torch.no_grad():
            y_hat = self.model(x_t_next_zero_sc, t_next, shortcut_size).detach()
        y_hat = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start, y_hat)

        x_t_sc = torch.cat((x_t, y_hat), dim=-1)
        y = self.model(x_t_sc, t, 2 * shortcut_size)

        y = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start, y)
        return y

    def _should_apply_self_conditioning(self) -> Tensor:
        """
        Determines whether to apply self-conditioning based on the self_conditioning_ratio.

        :returns:True if self-conditioning should be applied, False otherwise.
        :rtype: bool
        """
        return torch.rand(1) < self.self_conditioning_ratio

    def _modify_model_input_or_output(
            self,
            input_ids_mask: Tensor,
            x_start: Tensor,
            y_hat: Optional[Tensor] = None
    ) -> Tensor:
        return self._criterion._modify_model_input_or_output(input_ids_mask, x_start, y_hat)

    def _prepare_2_shortcut_input(
            self, step1_prediction: Tensor,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
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
    ) -> Tensor:
        raise NotImplementedError("This method should not be called on decorator")


class NllCriterion(Criterion):
    def __init__(
            self,
            model: Model,
            diffusion_steps,
    ):
        super().__init__(model, diffusion_steps)

    @override
    def compute_losses(self, batch: FlowMatchingBatch) -> dict[str, Tensor]:
        output = self.model.compute_logits(batch.x_start)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            batch.seqs.view(-1),
            reduction="none"
        ).view(batch.seqs.size())

        return {
            "nll_loss": loss
        }

class IsotropyCriterion(Criterion):

    def forward(self, *args, **kwargs):
        return self.compute_losses(*args, **kwargs)

    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        embedding_weights = self.model.module.word_embedding.weight
        return {
            "isotropy_loss": isotropy_loss(embedding_weights)
        }


def isotropy_loss(embeddings):
    norm_emb = embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    cos_sim = torch.mm(norm_emb, norm_emb.T)
    off_diag = cos_sim - torch.eye(cos_sim.size(0), device=cos_sim.device)
    return torch.mean(off_diag ** 2)


class CompositeCriterion(Criterion):

    def __init__(
            self,
            criteria: tuple[Criterion, ...],
            criteria_weights: tuple[float, ...],
            model: Model,
            diffusion_steps: int,
            self_consistency_ratio: float,
            sampler: TimeAndShortcutSampler,
    ):
        assert len(criteria) == len(criteria_weights), \
            (f"criteria and criteria_weights must have the same length but got"
             f" {len(criteria)} and {len(criteria_weights)}")

        super().__init__(model, diffusion_steps)
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.criteria = criteria
        self.criteria_weights = criteria_weights
        self.self_consistency_ratio = self_consistency_ratio
        self.sampler = sampler

    def forward(self, batch: EncoderBatch) -> dict[str, Tensor]:
        return self.compute_losses(batch)

    @override
    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        specific_batches = self._prepare_batches(batch)
        flow_marching_batch, consistency_batch, full_batch = specific_batches

        flow_and_decoder_loses = self.criteria[0](flow_marching_batch)
        flow_matching_loss = flow_and_decoder_loses["flow_matching_loss"]
        decoder_loss = flow_and_decoder_loses["decoder_loss"]
        consistency_loss = self.criteria[1](consistency_batch)["consistency_loss"]
        embedding_loss = self.criteria[2](full_batch)["nll_loss"]
        isotropy_loss = self.criteria[3](batch)["isotropy_loss"]

        losses = [flow_matching_loss.mean(), consistency_loss.mean(), embedding_loss.mean(), isotropy_loss]  # no decoder_loss
        weighted_losses = [
            loss * (weight or 1) for loss, weight in zip_longest(losses, self.criteria_weights or [], fillvalue=1)
        ]
        total_loss = sum(weighted_losses)

        return {
            "flow_matching_loss": weighted_losses[0],
            "consistency_loss": weighted_losses[1],
            "embedding_loss": weighted_losses[2],
            "decoder_loss": decoder_loss,
            "isotropy_loss": weighted_losses[3],
            "loss": total_loss,
            "timestep": full_batch.t,
            "shortcut": consistency_batch.shortcut_size,
        }

    def _prepare_batches(self, batch: EncoderBatch) -> tuple[FlowMatchingBatch, ShortcutFMBatch, FlowMatchingBatch]:
        bsz = batch.size()

        embeddings = self.model.get_embeddings(batch.seqs)
        t, shortcuts = self.sampler(batch_size=bsz, device=batch.seqs.device)
        x_t, noise = self._interpolate_data_noise(embeddings, t)
        x_t = torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, embeddings, x_t)

        num_consistency_elems = int(self.self_consistency_ratio * bsz)
        num_flow_matching_elems = bsz - num_consistency_elems

        full_batch = ShortcutFMBatch(
            batch.seqs,
            batch.padding_mask,
            batch.input_ids_mask,
            embeddings,
            x_t,
            noise,
            t,
            shortcuts
        )
        flow_matching_batch, consistency_batch = full_batch.split(num_flow_matching_elems)

        return (
            FlowMatchingBatch.from_shortcut_fm_batch(flow_matching_batch),
            consistency_batch,
            FlowMatchingBatch.from_shortcut_fm_batch(full_batch)
        )

    def denoise(
            self,
            batch: EncoderBatch,
            shortcut_size: Optional[int] = None,
            probe_every_step: bool = True,
            return_decoded: bool = False,
            return_logits: bool = False,
            step_size: Optional[int] = None
    ) -> np.ndarray[str, np.dtype[str]] | Tensor:
        """
        Denoises batch of examples with flexible probing and output options.

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
        # TODO: fix this terrible implementation (criteria[0])
        return self.criteria[0].denoise(
            batch,
            shortcut_size=shortcut_size,
            probe_every_step=probe_every_step,
            return_decoded=return_decoded,
            return_logits=return_logits,
            step_size=step_size
        )


class FlowNllCriterion(Criterion):
    def __init__(
            self,
            flow_matching_criterion: FlowMatchingCriterion,
            nll_criterion: NllCriterion,
            model: Model,
            diffusion_steps,
            sampler: ScheduleSampler,
    ):
        super().__init__(model, diffusion_steps)
        self.flow_matching_criterion = flow_matching_criterion
        self.nll = nll_criterion
        self.sampler = sampler
        print("initialized FlowNllCriterion")

    def forward(self, batch: EncoderBatch) -> dict[str, Tensor]:
        return self.compute_losses(batch)

    @override
    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        fm_batch = self._prepare_batch(batch)

        flow_and_decoder_loses = self.flow_matching_criterion(fm_batch)
        flow_matching_loss = flow_and_decoder_loses["flow_matching_loss"]
        decoder_loss = flow_and_decoder_loses["decoder_loss"]
        # embedding_loss = self.nll(fm_batch)["nll_loss"]

        # losses = [flow_matching_loss.mean(), embedding_loss.mean()]  # no decoder_loss
        losses = [flow_matching_loss.mean()]
        total_loss = sum(losses)

        return {
            "flow_matching_loss": flow_matching_loss,
            # "embedding_loss": embedding_loss,
            "decoder_loss": decoder_loss,
            "loss": total_loss,
            "timestep": fm_batch.t
        }

    def _prepare_batch(self, batch: EncoderBatch) -> FlowMatchingBatch:
        bsz = batch.size()

        embeddings = self.model.get_embeddings(batch.seqs)
        t, _ = self.sampler(batch_size=bsz, device=batch.seqs.device)
        x_t, noise = self._interpolate_data_noise(embeddings, t)
        x_t = torch.where(batch.input_ids_mask.unsqueeze(-1) == 0, embeddings, x_t)

        return FlowMatchingBatch(
            batch.seqs,
            batch.padding_mask,
            batch.input_ids_mask,
            embeddings,
            x_t,
            noise,
            t
        )

    def denoise(
            self,
            batch: EncoderBatch,
            shortcut_size: Optional[int] = None,
            probe_every_step: bool = True,
            return_decoded: bool = False,
            return_logits: bool = False,
            step_size: Optional[int] = None
    ) -> np.ndarray[str, np.dtype[str]]:
        """
        Denoises batch of examples with flexible probing and output options.

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
            step_size=step_size
        )
