"""
Various utilities for neural networks.
"""

import math

import torch as th
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


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
