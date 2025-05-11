from abc import ABC, abstractmethod

import numpy as np
import torch


class TimeAndShortcutSampler:
    """Base class for time and shortcut samplers, handling shared attributes and validation."""

    diffusion_steps: int
    min_shortcut_size: int
    max_shortcut_size: int

    def __init__(self, diffusion_steps: int, min_shortcut_size: int):
        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size
        self.max_shortcut_size = diffusion_steps // 2  # Max shortcut ensures two steps stay within bounds

    def __call__(self, batch_size, device):
        return self.sample(batch_size, device)

    def sample(self, batch_size, device):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the sample method.")


class ShortcutSampler:
    """Sampler for shortcut sizes, used by ShortcutFirstTimeAndShortcutSampler."""

    def __init__(self, diffusion_steps, min_shortcut_size):
        self.shortcut_values = torch.arange(min_shortcut_size, diffusion_steps // 2 + 1, dtype=torch.long)

    def sample(self, batch_size, device):
        """Sample shortcut sizes uniformly."""
        indices = torch.randint(0, len(self.shortcut_values), (batch_size,), device=device)
        return self.shortcut_values.to(device)[indices]


class ShortcutFirstTimeAndShortcutSampler(TimeAndShortcutSampler):
    """Sampler that first samples shortcuts uniformly, then selects valid timesteps."""

    def __init__(self, diffusion_steps, min_shortcut_size):
        super().__init__(diffusion_steps, min_shortcut_size)
        self.shortcut_sampler = ShortcutSampler(diffusion_steps, min_shortcut_size)

    def sample(self, batch_size, device):
        """Sample shortcuts first, then timesteps based on valid ranges."""
        shortcut_values = self.shortcut_sampler.sample(batch_size, device)
        max_steps = self.diffusion_steps // shortcut_values

        # Sample timesteps ensuring at least 2 steps can be taken without going below 0
        indices = torch.cat([torch.randint(2, max_step + 1, (1,), device=device) for max_step in max_steps])
        timesteps = indices * shortcut_values
        return timesteps.to(torch.long), shortcut_values.to(torch.long)


class TimestepFirstTimeAndShortcutSampler(TimeAndShortcutSampler):
    """Sampler that first samples timesteps uniformly, then selects valid shortcuts."""

    def sample(self, batch_size, device):
        """Sample timesteps uniformly, then determine valid shortcuts."""
        # Sample timesteps from [2*min_shortcut_size, diffusion_steps] to ensure valid shortcuts
        min_timestep = 2 * self.min_shortcut_size
        timesteps = torch.randint(
            min_timestep, self.diffusion_steps + 1, (batch_size,), device=device, dtype=torch.long
        )

        # Sample valid shortcuts (from min_shortcut_size to t/2 for each timestep t)
        shortcut_values = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            t = timesteps[i]
            max_shortcut = t // 2
            if max_shortcut < self.min_shortcut_size:
                shortcut_values[i] = self.min_shortcut_size
            else:
                valid_shortcuts = torch.arange(self.min_shortcut_size, max_shortcut + 1, device=device)
                shortcut_values[i] = valid_shortcuts[torch.randint(0, len(valid_shortcuts), (1,), device=device)]

        return timesteps, shortcut_values


class ScheduleSampler(ABC):
    """A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    def __call__(self, batch_size, device):
        return self.sample(batch_size, device)

    @abstractmethod
    def weights(self):
        """Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices + 1, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion_steps):
        self.diffusion_steps = diffusion_steps
        self._weights = np.ones([diffusion_steps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses, world_size):
        """Update the reweighting using losses from a model in PyTorch Lightning.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        :param world_size: the world_size
        """
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather batch sizes from all ranks
            batch_sizes = [torch.tensor([0], dtype=torch.int32, device=local_ts.device) for _ in range(world_size)]
            torch.distributed.all_gather(
                batch_sizes,
                torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
            )
            batch_sizes = [x.item() for x in batch_sizes]
            max_bs = max(batch_sizes)

            # Gather timesteps and losses
            timestep_batches = [torch.zeros(max_bs, dtype=local_ts.dtype, device=local_ts.device) for _ in batch_sizes]
            loss_batches = [
                torch.zeros(max_bs, dtype=local_losses.dtype, device=local_losses.device) for _ in batch_sizes
            ]
            torch.distributed.all_gather(timestep_batches, local_ts)
            torch.distributed.all_gather(loss_batches, local_losses)

            # Extract valid data from padded tensors
            timesteps = [x.item() for y, bs in zip(timestep_batches, batch_sizes, strict=False) for x in y[:bs]]
            losses = [x.item() for y, bs in zip(loss_batches, batch_sizes, strict=False) for x in y[:bs]]
        else:
            # Single-device case: no gathering needed
            timesteps = local_ts.tolist()
            losses = local_losses.tolist()

        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion_steps, history_per_term=10, uniform_prob=0.001):
        self.diffusion_steps = diffusion_steps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([diffusion_steps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([diffusion_steps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses, strict=False):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
