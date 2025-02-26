import torch
import numpy as np

class ShortcutSampler:
    def __init__(self, diffusion_steps, min_shortcut_size):
        assert diffusion_steps % min_shortcut_size == 0, "diffusion_steps must be divisible by min_shortcut_size"
        assert diffusion_steps >= min_shortcut_size, "diffusion_steps must be greater than min_shortcut_size"
        assert diffusion_steps % 2 == 0, "diffusion_steps must be even"
        assert min_shortcut_size % 2 == 0, "min_shortcut_size must be even"

        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size
        self.max_shortcut_size = diffusion_steps // 2
        self.shortcut_values = 2 ** torch.arange(
            int(np.log2(min_shortcut_size)), int(np.log2(self.max_shortcut_size)) + 1
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


class TimeAndShorcutStampler:
    def __init__(self, shortcut_sampler, diffusion_steps, min_shortcut_size):
        assert diffusion_steps % min_shortcut_size == 0, "diffusion_steps must be divisible by min_shortcut_size"
        assert diffusion_steps >= min_shortcut_size, "diffusion_steps must be greater than min_shortcut_size"
        assert diffusion_steps % 2 == 0, "diffusion_steps must be even"
        assert min_shortcut_size % 2 == 0, "min_shortcut_size must be even"

        self.diffusion_steps = diffusion_steps
        self.min_shortcut_size = min_shortcut_size
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
        indices = torch.cat([
            torch.randint(1, max_step + 1, (1,), device=device) for max_step in max_steps
        ])

        timesteps = indices * shortcut_values
        return timesteps.to(torch.long), shortcut_values.to(torch.long)
