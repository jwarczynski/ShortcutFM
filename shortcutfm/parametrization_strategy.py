from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module

from shortcutfm.utils.nn import mean_with_mask


class ParametrizationStartegy(ABC):
    def __init__(
            self,
            *,
            self_conditioning_ratio: float,
            consistency_objective_ratio: float,
    ) -> None:
        self.self_conditioning_ratio = self_conditioning_ratio
        self.consistency_objective_ratio = consistency_objective_ratio

    @abstractmethod
    def consistency_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """
        computes loss for consistency objective:

        ∥sθ(xt, t, 2d) − starget∥^2
        where starget = sθ(xt, t, shoruct_size)/2 + sθ(x′  t+shoruct_size, t, shoruct_size)/2 and x′  t+shoruct_size = xt + sθ(xt, t, shoruct_size)shoruct_size.

        :param model: model (sθ)
        :type model: nn.Module
        :param x_start: target_embedding
        :type x_start: torch.Tensor
        :param x_t: noisy input at time `t`
        :type x_t: torch.Tensor
        :param t: time at which the prediction is made
        :type t: torch.Tensor
        :param d: shortcut size
        :type d: torch.Tensor
        :param input_ids: input and target tokens conacatenated into one tensor
        :type input_ids: torch.Tensor
        :param input_ids_mask: mask delineating input sequence form target sequence
        :type input_ids_mask: torch.Tensor
        :returns: loss for consistency objective
        :rtype: torch.Tenosor
        """

    @abstractmethod
    def target_prediction_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """
        computes loss for target prediction objective:

        ∥sθ(xt, t, 0) − x_target∥^2

        :param model: model (sθ)
        :type model: nn.Module
        :param x_start: target_embedding
        :type x_start: torch.Tensor
        :param x_t: noisy input at time `t`
        :type x_t: torch.Tensor
        :param t: time at which the prediction is made
        :type t: torch.Tensor
        :param d: shortcut size
        :type d: torch.Tensor
        :param input_ids: input and target tokens conacatenated into one tensor
        :type input_ids: torch.Tensor
        :param input_ids_mask: mask delineating input sequence form target sequence
        :type input_ids_mask: torch.Tensor
        :returns: loss for target prediction objective
        :rtype: torch.Tenosor
        """

    def get_empty_self_conditioning_input(self, x_start: Tensor, input_ids_mask: Tensor) -> Tensor:
        """
        returns empty self conditioning input

        :param x_start: target_embedding
        :type x_start: torch.Tensor
        :param input_ids_mask: mask delineating input sequence form target sequence
        :type input_ids_mask: torch.Tensor
        """
        return torch.where(input_ids_mask == 0, x_start, 0).to(x_start.device)


class VelocityPredictionStartegy(ParametrizationStartegy):

    def consistency_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    def target_prediction_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """
        computes loss for velocity prediction objective:
        ∥sθ(xt, t, 0) − (x0 - noise)∥^2

        :param model: model (sθ)
        :type model: nn.Module
        :param x_start: target_embedding
        :type x_start: torch.Tensor
        :param x_t: noisy input at time `t`
        :type x_t: torch.Tensor
        :param t: time at which the prediction is made
        :type t: torch.Tensor
        :param d: shortcut size
        :type d: torch.Tensor
        :param input_ids: input and target tokens conacatenated into one tensor
        :type input_ids: torch.Tensor
        :param input_ids_mask: mask delineating input sequence form target sequence
        :type input_ids_mask: torch.Tensor
        :returns: loss for velocity prediction objective
        :rtype: torch.Tenosor
        """
        raise NotImplementedError


class X0PredictionStrategy(ParametrizationStartegy):

    def target_prediction_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        """
        computes loss for x0 prediction objective:

        ∥sθ(xt, t, 0) − x0∥^2

        :param model: model (sθ)
        :type model: nn.Module
        :param x_start: target_embedding
        :type x_start: torch.Tensor
        :param x_t: noisy input at time `t`
        :type x_t: torch.Tensor
        :param t: time at which the prediction is made
        :type t: torch.Tensor
        :param d: shortcut size
        :type d: torch.Tensor
        :param input_ids: input and target tokens conacatenated into one tensor
        :type input_ids: torch.Tensor
        :param input_ids_mask: mask delineating input sequence form target sequence
        :type input_ids_mask: torch.Tensor
        :returns: loss for x0 prediction objective
        :rtype: torch.Tenosor
        """

        if self.self_conditioning_ratio > 0.0:
            x_0_hat = self.get_empty_self_conditioning_input(x_start, input_ids_mask)
            x_t_zero_sc = torch.cat((x_t, x_0_hat), dim=-1)
            y_hat = model(x_t_zero_sc, t, 0)
            y_hat = torch.where(input_ids_mask == 0, x_start, y_hat)

            if torch.rand(1)[0] < self.self_conditioning_ratio:
                y_hat = y_hat.detach()
                y_hat = torch.cat((x_t, y_hat), dim=-1)
                y = model(y_hat, t, 0)
                return mean_with_mask((y - x_0_hat) ** 2, input_ids_mask)

            y = torch.where(input_ids_mask == 0, x_start, x_t)
            return mean_with_mask((y_hat - y) ** 2, input_ids_mask)

        y = model(x_t, t, 0)
        return mean_with_mask((y - x_start) ** 2, input_ids_mask)

    def consistency_objectvie(
            self,
            model: Module,
            x_start: Tensor,
            x_t: Tensor,
            t: Tensor,
            d: Tensor,
            input_ids: Tensor,
            input_ids_mask: Tensor,
    ) -> Tensor:
        if self.self_conditioning_ratio > 0.0:
            x_0_hat = self.get_empty_self_conditioning_input(x_start, input_ids_mask)
            x_t_zero_sc = torch.cat((x_t, x_0_hat), dim=-1)

            with torch.no_grad():
                target = self._compute_shortcut_target_self_conditioning(model, d, t, x_0_hat, x_t_zero_sc)

            y_hat = model(x_t_zero_sc, t, 2 * d)
            y_hat = torch.where(input_ids_mask == 0, x_start, y_hat)

            if torch.rand(1)[0] < self.self_conditioning_ratio:
                y_hat = y_hat.detach()
                y_hat = torch.cat((x_t, y_hat), dim=-1)
                y = model(y_hat, t, 2 * d)

                return mean_with_mask((y - target) ** 2, input_ids_mask)

            return mean_with_mask((y_hat - target) ** 2, input_ids_mask)

        with torch.no_grad():
            target = self._compute_shortcut_target(model, d, t, x_t)

        y = model(x_t, t, 2 * d)
        return mean_with_mask((y - target) ** 2, input_ids_mask)

    def _compute_shortcut_target_self_conditioning(self, model, shorcut_size, t, x_0_hat, x_t_sc):
        first_step_prediction = model(x_t_sc, t, shorcut_size)
        first_step_prediction_zero_sc = torch.cat((first_step_prediction, x_0_hat), dim=-1)
        second_step_prediction = model(first_step_prediction_zero_sc, t + shorcut_size, shorcut_size)
        target = second_step_prediction.detach()
        return target

    def _compute_shortcut_target(self, model, shorcut_size, t, x_t):
        first_step_prediction = model(x_t, t, shorcut_size)
        second_step_prediction = model(first_step_prediction, t + shorcut_size, shorcut_size)
        target = second_step_prediction.detach()
        return target
