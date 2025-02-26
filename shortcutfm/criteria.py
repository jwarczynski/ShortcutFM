from abc import ABC, abstractmethod
from itertools import zip_longest

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import Optional, override

from shortcutfm.batch import EncoderBatch, FlowMatchingBatch, ShortcutFMBatch
from shortcutfm.model.model import FlowMatchingModel as Model
from shortcutfm.shortcut_samplers import TimeAndShorcutStampler


class Criterion(Module, ABC):
    def __init__(self, model: Model, diffusion_steps: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.diffusion_steps = diffusion_steps

    def __call__(self, batch: EncoderBatch) -> dict[str, Tensor]:
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
            losses[key] = per_batch_loss.mean()
        return losses

    @abstractmethod
    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        """ Compute the losses. """

    def _interpolate_data_noise(
            self, x_start: Tensor, t: Tensor, noise: Optional[Tensor]=None
    ) -> tuple[Tensor, Tensor]:
        t = self.scale_t(t)
        t = t.view(-1, 1, 1)  # Reshape to (batch_size, 1, 1) to match x_start

        if noise is None:
            noise = torch.rand(x_start.size(), device=x_start.device)

        return x_start + (noise - x_start) * t, noise

    def scale_t(self, t):
        return t.float() * (1.0 / self.diffusion_steps)


class FlowMatchingCriterion(Criterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    def compute_losses(self, batch: FlowMatchingBatch) -> dict[str, Tensor]:
        target = self._compute_target(batch)
        output = self._predict(
            x_start=batch.x_start,
            x_t=batch.x_t,
            noise=batch.noise,
            t=batch.t,
            input_ids_mask=batch.input_ids_mask,
        )

        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        return {
            "flow_matching_loss": loss.sum(-1)
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
        y = self.model(x_t, t, torch.tensor(0, device=x_t.device))
        return y

    @abstractmethod
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        """ Compute the target. """

    @abstractmethod
    def _modify_model_input(self, input_ids_mask:Tensor, x_start: Tensor, y_hat: Optional[Tensor]=None) -> Tensor:
        """ Modify model input based on input_ids_mask. Used for self-conditioning. """


class X0FlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start

    @override
    def _modify_model_input(self, input_ids_mask:Tensor, x_start: Tensor, y_hat: Optional[Tensor]=None) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)


class VelocityFlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return batch.x_start - batch.noise

    @override
    def _modify_model_input(self, input_ids_mask:Tensor, x_start: Tensor, y_hat: Optional[Tensor]=None) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)


class FlowMatchinCriterionDecorator(FlowMatchingCriterion, ABC):
    def __init__(
            self,
            criterion: FlowMatchingCriterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps)
        self.criterion = criterion


class SelfConditioningFlowMatchingCriterionDecorator(FlowMatchinCriterionDecorator):

    def __init__(
            self,
            criterion: FlowMatchingCriterion,
            self_conditioning_ratio: float,
    ):
        super().__init__(criterion)
        self.self_conditioning_ratio = self_conditioning_ratio

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
        """ Compute the target and the model output. """
        x_0_hat = self._modify_model_input(input_ids_mask, x_start)

        if not self._should_apply_self_conditioning():
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion._predict(
                x_start=x_start, x_t=x_t, noise=noise, t=t, input_ids_mask=input_ids_mask
            )

        # TODO: model won't be quried at t + 1, only at t + d during inference
        # TODO: draw a shorcut value from distribution?
        t_next = t + 1
        x_t_next, _ = self._interpolate_data_noise(x_start, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            y_hat = self.model(x_t_next_zero_sc, self.scale_t(t_next), 0).detach()

        y_hat = self._modify_model_input(input_ids_mask, x_start, y_hat)
        y_hat = torch.cat((x_t, y_hat), dim=-1)
        return self.model(y_hat, t, 0)

    def _should_apply_self_conditioning(self) -> Tensor:
        """ Determines whether to apply self-conditioning based on the self_conditioning_ratio. """
        return torch.rand(1) < self.self_conditioning_ratio

    def _modify_model_input(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        return self.criterion._modify_model_input(input_ids_mask.unsqueeze(-1), x_start, y_hat)

    @override
    def _compute_target(self, batch: FlowMatchingBatch) -> Tensor:
        return self.criterion._compute_target(batch)


class ConsistencyCrterion(Criterion, ABC):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

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
        step2_prediction = self.model(step2_input, t + shortcut_size, shortcut_size)

        target = self._modify_target(step1_prediction, step2_prediction, x_start, x_t, t, shortcut_size, input_ids_mask)
        return target.detach()

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

        # TODO: pass loss_fn as argument
        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        return {
            "consistency_loss": loss.sum(-1)
        }

    @abstractmethod
    def _modify_model_input_or_output(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        """ Modify model input based on input_ids_mask. Used for self-conditioning. """


class X0ConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    @override
    def _prepare_2_shortcut_input(
            self, step1_prediction, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        step2_input = torch.where(input_ids_mask == 0, x_start, step1_prediction)
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
    def _modify_model_input_or_output(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        if y_hat is None:
            return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
        return torch.where(input_ids_mask == 0, x_start, y_hat).to(x_start.device)


class VelocityConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    @override
    def _prepare_2_shortcut_input(
            self, velocity, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        t = self.scale_t(t).view(-1, 1, 1)
        velocity = torch.where(input_ids_mask == 0, 0, velocity)
        return x_t + velocity * t

    @override
    def _modify_target(
            self, step1_prediction, step2_prediction, x_start, _, __, ___, input_ids_mask: Tensor
    ):
        step2_prediction = torch.where(input_ids_mask == 0, step1_prediction, step2_prediction)
        return (step1_prediction + step2_prediction) / 2

    @override
    def _modify_model_input_or_output(self, input_ids_mask: Tensor, x_start: Tensor, y_hat: Optional[Tensor] = None) -> Tensor:
        if y_hat is None:
            return torch.zeros_like(x_start).to(x_start.device)
        return torch.where(input_ids_mask == 0, 0, y_hat).to(x_start.device)


class ConsistencyCriterionDecorator(ConsistencyCrterion, ABC):
    def __init__(
            self,
            criterion: ConsistencyCrterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps)
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
        x_t = x_t[..., :embedding_dim]
        x_0_hat = self._criterion._modify_model_input_or_output(input_ids_mask, x_t)
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
            # TODO: replace with self._predict
            y = self._criterion._predict(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask
            )
            return y

        # TODO: check if t + shortcut_size is not > diffusion_steps
        t_next = t + shortcut_size
        x_t_next, noise = self._interpolate_data_noise(x_start, t_next)
        x_t_next = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, x_t_next)
        empty_self_conditioning_input = self._criterion._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start)
        x_t_next_zero_sc = torch.cat((x_t_next, empty_self_conditioning_input), dim=-1)

        y_hat = self.model(x_t_next_zero_sc, t, 2 * shortcut_size).detach()
        y_hat = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y_hat)

        x_t_sc = torch.cat((x_t, y_hat), dim=-1)
        y = self.model(x_t_sc, t, 2 * shortcut_size)

        y = self._modify_model_input_or_output(input_ids_mask.unsqueeze(-1), x_start, y)
        return y

    def _should_apply_self_conditioning(self):
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
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

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


class CompositeCriterion(Criterion):

    def __init__(
            self,
            criteria: tuple[Criterion, ...],
            criteria_weights: tuple[float, ...],
            model: Model,
            diffusion_steps: int,
            self_consistency_ratio: float,
            time_scheduler: Callable[[int], tuple[Tensor, Tensor]],
            shortcut_sampler: Callable[[int], Tensor],
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
        self.time_scheduler = time_scheduler
        self.shortcut_sampler = shortcut_sampler

    def __call__(self, batch: EncoderBatch) -> dict[str, Tensor]:
        return self.compute_losses(batch)

    @override
    def compute_losses(self, batch: EncoderBatch) -> dict[str, Tensor]:
        specific_batches = self._prepare_batches(batch)
        flow_marching_batch, consistency_batch, full_batch = specific_batches

        flow_and_decoder_loses = self.criteria[0](flow_marching_batch)
        flow_matching_loss = flow_and_decoder_loses["flow_matching_loss"]
        decoder_loss = flow_and_decoder_loses["flow_matching_loss"]
        consistency_loss = self.criteria[1](consistency_batch)["consistency_loss"]
        embedding_loss = self.criteria[2](full_batch)["nll_loss"]

        losses = [flow_matching_loss, consistency_loss, embedding_loss] # no decoder_loss
        weighted_losses = [
            loss * (weight or 1) for loss, weight in zip_longest(losses, self.criteria_weights or [], fillvalue=1)
        ]
        total_loss = sum(weighted_losses)

        return {
            "flow_matching_loss": flow_matching_loss,
            "consistency_loss": consistency_loss,
            "embedding_loss": embedding_loss,
            "decoder_loss": decoder_loss,
            "loss": total_loss
        }

    def _prepare_batches(self, batch: EncoderBatch) -> tuple[FlowMatchingBatch, ShortcutFMBatch, FlowMatchingBatch]:
        embeddings = self.model.aplly_embeddings(batch.seqs)
        t, weights = self.time_scheduler(batch.size())
        x_t, noise = self._interpolate_data_noise(embeddings, t)

        bsz = batch.size()
        num_consistency_elems = int(self.self_consistency_ratio * bsz)
        num_flow_matching_elems = bsz - num_consistency_elems

        full_batch = FlowMatchingBatch(
            batch.seqs,
            batch.padding_mask,
            batch.input_ids_mask,
            embeddings,
            x_t,
            noise,
            t,
        )
        flow_matching_batch, consistency_batch = full_batch.split(num_flow_matching_elems)

        shortcuts = self.shortcut_sampler(consistency_batch.size())
        consistency_batch = ShortcutFMBatch.from_flow_matching_batch(
            consistency_batch,
            shortcuts,
        )

        return (
            flow_matching_batch,
            consistency_batch,
            full_batch
        )
