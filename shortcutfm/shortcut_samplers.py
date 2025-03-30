from abc import ABC, abstractmethod

import numpy as np
import torch


class ShortcutSampler:
    def __init__(self, diffusion_steps, min_shortcut_size):
        assert diffusion_steps % min_shortcut_size == 0, "diffusion_steps must be divisible by min_shortcut_size"
        assert diffusion_steps >= min_shortcut_size, "diffusion_steps must be greater than min_shortcut_size"
        assert diffusion_steps % 2 == 0, "diffusion_steps must be even"
        assert min_shortcut_size % 2 == 0, "min_shortcut_size must be even"

        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size
        self.max_shortcut_size = diffusion_steps // 2 # becasue during training we will query at the twice the shortcut size
        self.shortcut_values = 2 ** torch.arange(
            1, int(np.log2(self.max_shortcut_size)) + 1
        ).to(torch.long)

    def __call__(self, batch_size, device):
        return self.sample(batch_size, device)

    def sample(self, batch_size, device):
        """
        Sample shortcut sizes from powers of 2, ensuring they fit within training constraints.
        """
        indices = torch.randint(0, len(self.shortcut_values), (batch_size,), device=device)
        shorcut_values = self.shortcut_values.to(device)
        return shorcut_values[indices].to(device)


class TimeAndShortcutSampler:
    def __init__(self, shortcut_sampler, diffusion_steps):
        assert diffusion_steps % 2 == 0, "diffusion_steps must be even"

        self.diffusion_steps = diffusion_steps
        self.shortcut_sampler = shortcut_sampler

    def __call__(self, batch_size, device):
        return self.sample(batch_size, device)

    def sample(self, batch_size, device):
        """
        Sample time steps based on shortcut values, ensuring alignment with inference behavior.
        """
        shortcut_values = self.shortcut_sampler.sample(batch_size, device)
        max_steps = self.diffusion_steps // shortcut_values

        # Ensure each batch element gets a proper sample
        indices = torch.cat(
            [
                # 2 because for consistency target we need to be able to make to steps of size shortcut and do not go below timestep 0
                # so smallest timestep must be at least 2 times the shortcut size
                torch.randint(2, max_step + 1, (1,), device=device) for max_step in max_steps
            ]
        )

        timesteps = indices * shortcut_values
        return timesteps.to(torch.long), shortcut_values.to(torch.long)


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
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
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

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
        """
        Update the reweighting using losses from a model in PyTorch Lightning.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        :param world_size: the world_size
        """
        if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            # Gather batch sizes from all ranks
            batch_sizes = [
                torch.tensor([0], dtype=torch.int32, device=local_ts.device)
                for _ in range(world_size)
            ]
            torch.distributed.all_gather(
                batch_sizes,
                torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
            )
            batch_sizes = [x.item() for x in batch_sizes]
            max_bs = max(batch_sizes)

            # Gather timesteps and losses
            timestep_batches = [torch.zeros(max_bs, device=local_ts.device) for _ in batch_sizes]
            loss_batches = [torch.zeros(max_bs, device=local_losses.device) for _ in batch_sizes]
            torch.distributed.all_gather(timestep_batches, local_ts)
            torch.distributed.all_gather(loss_batches, local_losses)

            # Extract valid data from padded tensors
            timesteps = [
                x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
            ]
            losses = [
                x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]
            ]
        else:
            # Single-device case: no gathering needed
            timesteps = local_ts.tolist()
            losses = local_losses.tolist()

        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

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
        self._loss_history = np.zeros(
            [diffusion_steps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion_steps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
