"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn

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


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def mean_with_mask(tensor, mask):
    assert tensor.shape == mask.shape
    mean_value = (tensor*mask).sum(
        dim=list(range(1, len(tensor.shape)))) / mask.sum(dim=list(range(1, len(tensor.shape)))
    )
    return mean_value

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


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
