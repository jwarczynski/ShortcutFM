from abc import ABC, abstractmethod
from typing import Callable

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

    def __call__(self, *args, **kwargs):
        """ Compute the target and model output. """
        return self.get_target_and_output(*args, **kwargs)

    @abstractmethod
    def get_target_and_output(self, *args, **kwargs):
        """ Compute the target and model output. """

    def get_x_t(self, x_start: Tensor, noise: Tensor, t: Tensor) -> Tensor:
        t = self.scale_t(t)
        t = t.view(-1, 1, 1)  # Reshape to (batch_size, 1, 1) to match x_start

        return  x_start + (noise - x_start) * t

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
            shortuct_size: Tensor,
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
            shortuct_size: Tensor,
            input_ids_mask: Tensor,
    ):
        """ Compute the loss. """
        return super().get_target_and_output(
            x_start=x_start,
            x_t=x_t,
            noise=noise,
            t=t,
            shortuct_size=shortuct_size,
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
            shortuct_size=shortcut_size,
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
        x_0_hat = _get_empty_self_conditioning_input(x_start, input_ids_mask)

        if torch.rand(1) > self.self_conditioning_ratio:
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion(x_start, x_t, noise, t, shortuct_size, input_ids_mask)

        t_next = t + 1
        x_t_next = self.get_x_t(x_start, noise, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            y_hat = self.model(x_t_next_zero_sc, self.scale_t(t_next), 0).detach()

        # TODO: for velocity it might be better to set 0 everywhere instead x_start
        y_hat = torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, y_hat)
        y_hat = torch.cat((x_t, y_hat), dim=-1)
        return x_start, self.model(y_hat, t, 0)


# TODO: for velocity it might be better to set 0 everywhere instead x_start
def _get_empty_self_conditioning_input(x_start: Tensor, input_ids_mask: Tensor) -> Tensor:
    """
    returns empty self conditioning input

    :param x_start: target_embedding
    :type x_start: torch.Tensor
    :param input_ids_mask: mask delineating input sequence form target sequence
    :type input_ids_mask: torch.Tensor
    """
    return torch.where(input_ids_mask.unsqueeze(-1) == 0, x_start, 0).to(x_start.device)


class ConsistencyCrterion(Criterion, ABC):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)
        self.first_step_predicition_hooks = []
        self.target_hooks = []


    def add_first_step_predicition_hook(
            self,
            hook: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    ):
        self.first_step_predicition_hooks.append(hook)

    def add_target_hook(
            self,
            hook: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    ):
        self.target_hooks.append(hook)

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
            target = self._compute_shortcut_target(
                shorcut_size=shortcut_size,
                t=t,
                x_t=x_t,
                x_start=x_start,
                input_ids_mask=input_ids_mask
            )

        y = self.model(x_t, t, 2 * shortcut_size)
        return target, y

    def _compute_shortcut_target(
            self,
            *,
            shorcut_size:Tensor,
            t:Tensor,
            x_t:Tensor,
            x_start:Tensor,
            input_ids_mask:Tensor,
    ):
        input_ids_mask = input_ids_mask.unsqueeze(-1)
        step1_prediction = self.model(x_t, t, shorcut_size)

        step2_input = step1_prediction
        for hook in self.first_step_predicition_hooks:
            step2_input = hook(step1_prediction, x_t, t, shorcut_size, input_ids_mask)

        step2_prediction = self.model(step2_input, t + shorcut_size, shorcut_size)
        target = step2_prediction
        for hook in self.target_hooks:
            target = hook(step1_prediction, step2_prediction, x_t, t, shorcut_size, input_ids_mask)

        return target.detach()


class X0ConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

        def get_input_for2sohrtcut(
                step1_prediction, x_t, t, shorcut_size, input_ids_mask: Tensor
        ):
            step2_input = torch.where(input_ids_mask == 0, x_t, step1_prediction)
            return step2_input

        def compute_x0_consitency_target(
                _, step2_prediction, x_t, __, ___, input_ids_mask:Tensor
        ):
            target = torch.where(input_ids_mask == 0, x_t, step2_prediction)
            return target

        self.add_first_step_predicition_hook(get_input_for2sohrtcut)
        self.add_target_hook(compute_x0_consitency_target)


class VelocityConsistencyCrterion(ConsistencyCrterion):
    def __init__(
            self,
            model: Model,
            difusion_steps,
    ):
        super().__init__(model, difusion_steps)

        def get_input_for2sohrtcut(
                velocity, x_t, t, shorcut_size, input_ids_mask: Tensor
        ):
            t = self.scale_t(t).view(-1, 1, 1)
            velocity = torch.where(input_ids_mask == 0, 0, velocity)
            return x_t + velocity * t

        def compute_velocity_consitency_target(
                step1_prediction, step2_prediction, _, __, ___, input_ids_mask: Tensor
        ):
            step2_prediction = torch.where(input_ids_mask == 0, step1_prediction, step2_prediction)
            return (step1_prediction + step2_prediction) / 2

        self.add_first_step_predicition_hook(get_input_for2sohrtcut)
        self.add_target_hook(compute_velocity_consitency_target)

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
        super().__init__(criterion.model)
        self.criterion = criterion

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


class SelfConditioningConsistencyCriterionDecorator(ConsistencyCriterionDecorator):
    def __init__(
            self,
            criterion: ConsistencyCrterion,
            self_conditioning_ratio: float,
    ):
        super().__init__(criterion)
        self.self_conditioning_ratio = self_conditioning_ratio
        self.criterion.add_first_step_predicition_hook(add_empty_self_conditioning_input)

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
        x_0_hat = _get_empty_self_conditioning_input(x_start, input_ids_mask)

        if torch.rand(1) > self.self_conditioning_ratio:
            x_t = torch.cat((x_t, x_0_hat), dim=-1)
            return self.criterion(
                x_start=x_start,
                x_t=x_t,
                noise=noise,
                t=t,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask
            )

        t_next = t + 1
        x_t_next = self.get_x_t(x_start, noise, t_next)

        x_t_next_zero_sc = torch.cat((x_t_next, x_0_hat), dim=-1)
        with torch.no_grad():
            target, y_hat = self.criterion(
                x_start=x_start,
                x_t=x_t_next_zero_sc,
                noise=noise,
                t=t_next,
                shortcut_size=shortcut_size,
                input_ids_mask=input_ids_mask
            )
            y_hat = y_hat.detach()

        x_t_sc = torch.cat((x_t, y_hat), dim=-1)
        y = self.model(x_t_sc, t, 2 * shortcut_size)
        return target, y



def add_empty_self_conditioning_input(x, x_t, t, shorcut_size, input_ids_mask):
    x_0_hat = _get_empty_self_conditioning_input(x, input_ids_mask)
    return torch.cat((x, x_0_hat), dim=-1)