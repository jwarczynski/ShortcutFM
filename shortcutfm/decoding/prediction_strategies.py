from abc import ABC, abstractmethod
from typing import override

import numpy as np
import torch
from numpy import dtype, ndarray
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from shortcutfm.batch import EncoderBatch
from shortcutfm.model.model import FlowMatchingModel


class PredictionStrategy(ABC):
    """Base class for denoisng process"""

    # TODO: tokenizer can also be MyTokenizer, but for now it does not support batch_decoding
    def __init__(
        self,
        model: FlowMatchingModel,
        diffusion_steps: int,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.model = model
        self.diffusion_steps = diffusion_steps
        self.tokenizer = tokenizer

        self.x_t = None

    def __call__(self, batch: EncoderBatch, shortcut_size: int) -> ndarray[str, dtype[str]]:
        return self.denoise(batch, shortcut_size)

    def denoise(self, batch: EncoderBatch, shortcut_size: int) -> ndarray[str, dtype[str]]:
        """Denoises batch of exapmles

        :param batch: batch of exapmles to denoise
        :type batch: EncoderBatch
        :param shortcut_size: shorcut size to use during denoising
        :type shortcut_size: int

        :returns: np.array of strings for each exmaple and timestep. Each row corresponds to single example
        :rtype: ndarray
        """
        self._reset()
        input_mask = batch.input_ids_mask.unsqueeze(-1)
        embeddings = self.model.get_embeddings(batch.seqs)
        noise = torch.randn_like(embeddings)
        self.x_t = torch.where(input_mask == 0, embeddings, noise)
        predicted_seqs = self.denoise_loop(shortcut_size, input_mask)

        return np.array(predicted_seqs).T

    def denoise_loop(self, shortcut_size: int, input_mask) -> list[list[str]]:
        """Denosing loop for a given shorcut size"""
        predicted_seqs: list[list[str]] = []

        shortcuts = torch.tensor(shortcut_size, device=input_mask.device).repeat(input_mask.shape[0])
        for t in torch.arange(self.diffusion_steps, 0, -shortcut_size, device=input_mask.device):
            t = t.repeat(input_mask.shape[0])
            model_output = self.infere_model(self.x_t, t, shortcuts, input_mask)
            v_hat = self.predict_velocity(self.x_t, model_output, t, shortcuts, input_mask)
            x0_hat = self.x_t + (shortcuts / self.diffusion_steps)[:, None, None] * v_hat

            self.x_t = x0_hat
            predicted_seqs.append(self.probe(x0_hat))

        return predicted_seqs

    def infere_model(self, x_t: Tensor, t: Tensor, shortcut_size: Tensor, input_mask: Tensor) -> Tensor:
        """Call the model and resotre input part of the pediction"""
        model_output = self.model(x_t, t, shortcut_size)
        return self._restore_input_part(model_output, x_t, input_mask)

    @abstractmethod
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        """Recover input part of the prediction based on input_mask"""

    @abstractmethod
    def predict_velocity(
        self,
        x_t,
        model_output: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_mask: Tensor,
    ) -> Tensor:
        """Computes velocity based on models output"""

    def probe(self, hidden_representation) -> list[str]:
        """Predicts sequence of tokens based on hidden_representation"""
        logtis = self.model.compute_logits(hidden_representation)
        probs = torch.softmax(logtis, dim=-1)
        tokens = torch.argmax(probs, dim=-1)
        seqs = self.tokenizer.batch_decode(tokens)
        return seqs

    def _reset(self):
        """Allow subclasses to prepare for new batch of examples.

        For example, it can reset stored conditioning values.
        """
        return


class X0PredictionStrategy(PredictionStrategy):
    @override
    def predict_velocity(self, x_t, x0_hat: Tensor, t: Tensor, shortcut_size: Tensor, input_mask: Tensor) -> Tensor:
        v_hat = x0_hat - x_t
        assert torch.all(v_hat[input_mask.expand_as(v_hat) == 0] == 0), "v_hat is not zero where input_mask is zero"
        return v_hat

    @override
    def _restore_input_part(self, x0_hat: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        return torch.where(input_mask == 0, x_t, x0_hat)


class VelocityPredcitionStrategy(PredictionStrategy):
    @override
    def predict_velocity(self, x_t, v_hat: Tensor, t: Tensor, shortcut_size: Tensor, input_mask: Tensor) -> Tensor:
        v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)
        return v_hat

    @override
    def _restore_input_part(self, v_hat: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        return torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)


class SelfConditioningPredictionDecorator(PredictionStrategy):
    """Decorator for creating self-conditioning input for the model"""

    def __init__(
        self,
        prediction_strategy: PredictionStrategy,
        model: FlowMatchingModel,
        diffusion_steps: int,
        tokenizer,
    ) -> None:
        super().__init__(model, diffusion_steps, tokenizer)
        self.prediction_strategy = prediction_strategy
        self.y_hat = None

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
    def predict_velocity(
        self,
        x_t,
        model_output: Tensor,
        t: Tensor,
        shortcut_size: Tensor,
        input_mask: Tensor,
    ) -> Tensor:
        """COmputes velocity from models output"""
        return self.prediction_strategy.predict_velocity(x_t, model_output, t, shortcut_size, input_mask)

    @override
    def _restore_input_part(self, model_output: Tensor, x_t: Tensor, input_mask: Tensor) -> Tensor:
        _, _, dim = model_output.shape
        x_t = x_t[:, :, :dim]
        return self.prediction_strategy._restore_input_part(model_output, x_t, input_mask)

    @override
    def _reset(self):
        """Resets self-conditioning storage for new batch of data"""
        self.y_hat = None
