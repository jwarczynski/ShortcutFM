from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module
from typing_extensions import override


class Model(ABC):
    def __init__(
            self,
            module: Module
    ):
        self.module = module

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Forward pass of the model. """


class Criterion(ABC):
    def __init__(
            self,
            model: Model,
            diffusion_steps: int,
    ):
        self.model = model
        self.diffusion_steps = diffusion_steps

    def __call__(
            self,
            *,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and model output. """
        return self.get_target_and_output(
            x_start=x_start,
            x_t=x_t,
            noise=noise,
            t=t,
            shortcut_size=shortcut_size,
            input_ids_mask=input_ids_mask,
        )

    @abstractmethod
    def get_target_and_output(
            self,
            *,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and model output. """

    def interpolate_data_noise(self, x_start: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        t = self.scale_t(t)
        t = t.view(-1, 1, 1)  # Reshape to (batch_size, 1, 1) to match x_start

        return x_start + (noise - x_start) * t

    def scale_t(self, t):
        return t.float() * (1.0 / self.diffusion_steps)


class FlowMatchingCriterion(Criterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    def get_target_and_output(
            self,
            *,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        y = self.model(x_t, t, torch.tensor(0, device=x_t.device))
        return x_start, y


# TODO: maybe remove or implement computing target
class X0FlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    def get_target_and_output(
            self,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ):
        """ Compute the loss. """
        return super().get_target_and_output(
            x_start=x_start,
            x_t=x_t,
            noise=noise,
            t=t,
            shortcut_size=shortcut_size,
            input_ids_mask=input_ids_mask,
        )


# TODO: maybe remove or implement computing target
class VelocityFlowMatchingCriterion(FlowMatchingCriterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    @override
    def get_target_and_output(
            self,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        target, output = super().get_target_and_output(
            x_start=x_start,
            x_t=x_t,
            noise=noise,
            t=t,
            shortcut_size=shortcut_size,
            input_ids_mask=input_ids_mask,
        )
        # TODO: consider: target = x_start - noise
        return target, output


class FlowMatchinCriterionDecorator(FlowMatchingCriterion, ABC):
    def __init__(
            self,
            criterion: FlowMatchingCriterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps)
        self.criterion = criterion

    @abstractmethod
    def __call__(
            self,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortuct_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """


class SelfConditioningFlowMatchingCriterionDecorator(FlowMatchinCriterionDecorator):
    def __init__(
            self,
            criterion: FlowMatchingCriterion,
            self_conditioning_ratio: float,
    ):
        super().__init__(criterion)
        self.self_conditioning_ratio = self_conditioning_ratio

    def __call__(
            self,
            *,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortuct_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        x_0_hat = _get_empty_self_conditioning_input(x_start, input_ids_mask.unsqueeze(-1))

        if torch.rand(1) > self.self_conditioning_ratio:
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion(
                x_start=x_start, x_t=x_t, noise=noise, t=t, shortcut_size=shortuct_size, input_ids_mask=input_ids_mask
            )

        t_next = t + 1
        x_t_next = self.interpolate_data_noise(x_start, noise, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            y_hat = self.model(x_t_next_zero_sc, self.scale_t(t_next), 0).detach()

        y_hat = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y_hat)
        y_hat = torch.cat((x_t, y_hat), dim=-1)
        return x_start, self.model(y_hat, t, 0)


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
            self, step1_prediction: Tensor,
            step2_prediction: Tensor,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """ Modifies target based on two shorcuts predicitons """

    def compute_shortcut_target(
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

    def get_target_and_output(
            self,
            *,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        with torch.no_grad():
            target = self.compute_shortcut_target(
                shortcut_size=shortcut_size,
                t=t,
                x_t=x_t,
                x_start=x_start,
                input_ids_mask=input_ids_mask
            )

        y = self.model(x_t, t, 2 * shortcut_size)
        y = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y)
        return target, y


class X0ConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    def _prepare_2_shortcut_input(
            self, step1_prediction, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        embedding_dim = step1_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        step2_input = torch.where(input_ids_mask == 0, x_start, step1_prediction)
        return step2_input

    def _modify_target(
            self, _, step2_prediction, x_start, x_t, __, ___, input_ids_mask: Tensor
    ):
        embedding_dim = step2_prediction.size(-1)
        input_ids_mask = input_ids_mask[..., :embedding_dim]
        x_start = x_start[..., :embedding_dim]
        target = torch.where(input_ids_mask == 0, x_start, step2_prediction)
        return target


class VelocityConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

    def _prepare_2_shortcut_input(
            self, velocity, x_start, x_t, t, shorcut_size, input_ids_mask: Tensor
    ):
        t = self.scale_t(t).view(-1, 1, 1)
        velocity = torch.where(input_ids_mask == 0, 0, velocity)
        return x_t + velocity * t

    def _modify_target(
            self, step1_prediction, step2_prediction, x_start, _, __, ___, input_ids_mask: Tensor
    ):
        step2_prediction = torch.where(input_ids_mask == 0, step1_prediction, step2_prediction)
        return (step1_prediction + step2_prediction) / 2

    def get_target_and_output(
            self,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            noise: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        # TODO: target = x_start - noise
        return super().get_target_and_output(
            x_start=x_start,
            x_t=x_t,
            noise=noise,
            t=t,
            shortcut_size=shortcut_size,
            input_ids_mask=input_ids_mask,
        )


class ConsistencyCriterionDecorator(ConsistencyCrterion, ABC):
    def __init__(
            self,
            criterion: ConsistencyCrterion,
    ):
        super().__init__(criterion.model, criterion.diffusion_steps)
        self.criterion = criterion


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
        self.criterion._prepare_2_shortcut_input = self._wrapped_prepare_2_shortcut_input

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
        x_0_hat = _get_empty_self_conditioning_input(x_t, input_ids_mask)
        return torch.cat((original_result, x_0_hat), dim=-1)

    def get_target_and_output(
            self,
            x_start: Tensor,
            x_t: Tensor,
            noise: Tensor,
            t: Tensor,
            shortcut_size: Tensor,
            input_ids_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """ Compute the target and the model output. """
        x_0_hat = _get_empty_self_conditioning_input(x_start, input_ids_mask.unsqueeze(-1))

        if not self._should_apply_self_conditioning():
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask
            )

        # TODO: check if t + shortcut_size is not > diffusion_steps
        t_next = t + shortcut_size
        x_t_next = self.interpolate_data_noise(x_start, noise, t_next)
        x_t_next = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, x_t_next)
        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)

        with torch.no_grad():
            x_t_target = torch.cat((x_t, x_0_hat), dim=-1)
            target = self.criterion.compute_shortcut_target(
                shortcut_size=shortcut_size,
                x_start=x_start,
                x_t=x_t_target,
                t=t,
                input_ids_mask=input_ids_mask
            )

        y_hat = self.model(x_t_next_zero_sc, t, 2 * shortcut_size).detach()
        y_hat = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y_hat)

        x_t_sc = torch.cat((x_t, y_hat), dim=-1)
        y = self.model(x_t_sc, t, 2 * shortcut_size)
        y = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y)

        return target, y

    def _should_apply_self_conditioning(self):
        """
        Determines whether to apply self-conditioning based on the self_conditioning_ratio.

        :returns:True if self-conditioning should be applied, False otherwise.
        :rtype: bool
        """
        return torch.rand(1) < self.self_conditioning_ratio

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


def _get_empty_self_conditioning_input(x_start: Tensor, input_ids_mask: Tensor) -> Tensor:
    """
    returns empty self conditioning input

    :param x_start: target_embedding
    :type x_start: torch.Tensor
    :param input_ids_mask: mask delineating input sequence form target sequence
    :type input_ids_mask: torch.Tensor
    """
    return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)
