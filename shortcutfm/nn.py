"""
Various utilities for neural networks.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from shortcutfm.config import TrainingConfig


class MyleLR(LRScheduler):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, start_lr: float = 0.0, last_epoch: int = -1):
        if num_warmup_steps <= 0:
            raise ValueError("`num_warmup_steps` must be greater than 0.")

        self.num_warmup_steps = num_warmup_steps
        self.start_lr = start_lr

        # Ensure 'initial_lr' is set for each param group
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = param_group['lr']

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)  # Prevent division by zero
        base_lrs = [group['initial_lr'] for group in self.optimizer.param_groups]

        if step < self.num_warmup_steps:
            factor = step / self.num_warmup_steps  # Linear warmup
        else:
            factor = (self.num_warmup_steps / step) ** 0.5  # Inverse sqrt decay

        return [self.start_lr + (base_lr - self.start_lr) * factor for base_lr in base_lrs]


class VMFLoss(ABC):
    """
    Abstract base class for von Mises-Fisher negative log-likelihood loss.
    Implements common logic for approximate log(C_m(kappa)) to avoid underflow issues.

    The implementation is based on the probabilistic loss proposed in:
    Kumar et al., "Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs," arXiv, 2019,
    http://arxiv.org/abs/1812.04616

    Args:
        config (TrainingConfig): Configuration object with:
            - model.hidden_size (int): Dimension m of the embeddings.
            - loss.mvf_loss_config.lambda_1 (float, optional): Regularization parameter for NormPenalizedVMFLoss.
            - loss.mvf_loss_config.lambda_2 (float, optional): Regularization parameter for DotProductScaledVMFLoss.
            - loss.mvf_loss_config.cosine_threshold (float, optional): Threshold for cosine penalty in CosinePenalizedVMFLoss.
            - loss.mvf_loss_config.cosine_penalty_scale (float, optional): Scale for cosine penalty in CosinePenalizedVMFLoss.
    """

    def __init__(self, config: TrainingConfig):
        self.embedding_dim = config.model.hidden_size
        self.v = self.embedding_dim / 2.0 - 1  # v = m/2 - 1

    def _compute_neg_log_cm(self, kappa):
        """
        Compute approximate -log(C_m(kappa)) to avoid underflow.

        Args:
            kappa (torch.Tensor): Norm ||hat{e}||, shape (batch_size, seq_len).

        Returns:
            torch.Tensor: -log(C_m(kappa)), shape (batch_size, seq_len).
        """
        # Approximate log(C_m(kappa)):
        # log(C_m(kappa)) >= sqrt((v + 1)^2 + kappa^2) - (v - 1) * log(v - 1 + sqrt((v + 1)^2 + kappa^2))
        sqrt_term = torch.sqrt((self.v + 1) ** 2 + kappa ** 2)  # Shape: (batch_size, seq_len)
        log_cm_approx = sqrt_term - (self.v - 1) * torch.log(self.v - 1 + sqrt_term + 1e-10)  # Epsilon for stability
        return -log_cm_approx

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the vMF loss.

        Args:
            output (torch.Tensor): Model output (hat{e}), shape (batch_size, seq_len, embedding_dim).
            target (torch.Tensor): Target embedding (e(w)), shape (batch_size, seq_len, embedding_dim), unnormalized.

        Returns:
            torch.Tensor: Per embedding loss, shape (batch_size, seq_len, embedding_dim).
        """
        # Ensure target is unit-norm
        target = target / (torch.norm(target, dim=-1, keepdim=True) + 1e-10)

        loss = self._compute_loss(output, target)  # Shape: (batch_size, seq_len)

        # Expand loss to match output shape
        return loss.unsqueeze(-1).expand_as(output)  # Shape: (batch_size, seq_len, embedding_dim)

    @abstractmethod
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the vMF loss.

        Args:
            output (torch.Tensor): Model output (hat{e}), shape (batch_size, seq_len, embedding_dim).
            target (torch.Tensor): Target embedding (e(w)), shape (batch_size, seq_len, embedding_dim), unit-norm.

        Returns:
            torch.Tensor: Per-sample loss, shape (batch_size, seq_len).
        """
        pass


class NormPenalizedVMFLoss(VMFLoss):
    """
    vMF loss with norm regularization: -log(C_m(||hat{e}||)) - hat{e}^T e(w) + lambda_1 ||hat{e}||.

    The implementation follows the regularization approach in:
    Kumar et al., "Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs," arXiv, 2019,
    http://arxiv.org/abs/1812.04616

    Args:
        config (TrainingConfig): Configuration object with:
            - model.hidden_size (int): Dimension m of the embeddings.
            - loss.mvf_loss_config.lambda_1 (float): Regularization parameter.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.lambda_1 = config.loss.mvf_loss_config.lambda_1

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kappa = torch.norm(output, dim=-1, p=2)  # Shape: (batch_size, seq_len)

        # Compute -log(C_m(kappa))
        neg_log_cm = self._compute_neg_log_cm(kappa)  # Shape: (batch_size, seq_len)

        # Compute hat{e}^T e(w)
        dot_product = torch.sum(output * target, dim=-1)  # Shape: (batch_size, seq_len)

        # Loss: -log(C_m(||hat{e}||)) - hat{e}^T e(w) + lambda_1 ||hat{e}||
        loss = neg_log_cm - dot_product + self.lambda_1 * kappa
        return loss


class DotProductScaledVMFLoss(VMFLoss):
    """
    vMF loss with scaled dot product: -log(C_m(||hat{e}||)) - lambda_2 * hat{e}^T e(w).

    The implementation follows the regularization approach in:
    Kumar et al., "Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs," arXiv, 2019,
    http://arxiv.org/abs/1812.04616

    Args:
        config (TrainingConfig): Configuration object with:
            - model.hidden_size (int): Dimension m of the embeddings.
            - loss.mvf_loss_config.lambda_2 (float): Regularization parameter.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.lambda_2 = config.loss.mvf_loss_config.lambda_2

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kappa = torch.norm(output, dim=-1, p=2)  # Shape: (batch_size, seq_len)

        # Compute -log(C_m(kappa))
        neg_log_cm = self._compute_neg_log_cm(kappa)  # Shape: (batch_size, seq_len)

        # Compute hat{e}^T e(w)
        dot_product = torch.sum(output * target, dim=-1)  # Shape: (batch_size, seq_len)

        # Loss: -log(C_m(||hat{e}||)) - lambda_2 * hat{e}^T e(w)
        loss = neg_log_cm - self.lambda_2 * dot_product
        return loss


class CosinePenalizedVMFLoss(VMFLoss):
    """
    vMF loss with cosine similarity penalty: -log(C_m(||hat{e}||)) + log(1 + ||hat{e}||) * (cosine_threshold - cos(theta)).

    The implementation is adapted from an alternative vMF loss formulation, taken from official repostory to:
    Kumar et al., "Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs," arXiv, 2019,
    http://arxiv.org/abs/1812.04616

    Args:
        config (TrainingConfig): Configuration object with:
            - model.hidden_size (int): Dimension m of the embeddings.
            - loss.mvf_loss_config.cosine_threshold (float): Threshold for cosine similarity penalty (default: 0.2).
            - loss.mvf_loss_config.cosine_penalty_scale (float): Scaling factor for cosine penalty (default: 1.0).
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.cosine_threshold = config.loss.mvf_loss_config.cosine_threshold
        self.cosine_penalty_scale = config.loss.mvf_loss_config.cosine_penalty_scale

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kappa = torch.norm(output, dim=-1, p=2)  # Shape: (batch_size, seq_len)

        # Compute -log(C_m(kappa))
        neg_log_cm = self._compute_neg_log_cm(kappa)  # Shape: (batch_size, seq_len)

        # Normalize output and target for cosine similarity
        output_norm = output / (torch.norm(output, dim=-1, keepdim=True) + 1e-10)
        target_norm = target  # Already normalized in __call__

        # Compute cos(theta) = hat{e}_norm^T e(w)_norm
        cos_theta = torch.sum(output_norm * target_norm, dim=-1)  # Shape: (batch_size, seq_len)

        # Loss: -log(C_m(||hat{e}||)) + scale * log(1 + ||hat{e}||) * (cosine_threshold - cos(theta))
        loss = neg_log_cm + self.cosine_penalty_scale * torch.log1p(kappa) * (self.cosine_threshold - cos_theta)
        return loss


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
