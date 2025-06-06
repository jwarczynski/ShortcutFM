import json
from pathlib import Path

import lightning as pl
import numpy as np
import torch


# Custom Gradient Monitor Callback for Lightning
class GradientMonitor(pl.Callback):
    def __init__(self, log_every_n_steps=100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % self.log_every_n_steps == 0:
            # Log gradient norms
            grad_norm_dict = {}
            total_norm = 0
            for name, p in pl_module.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm_dict[f"grad_norm/{name}"] = param_norm
                    total_norm += param_norm.item() ** 2

            # Log total norm
            total_norm = total_norm**0.5
            trainer.logger.log_metrics(
                {"grad_norm/total": total_norm, **grad_norm_dict},
                step=trainer.global_step,
            )


# Custom EMA Callback for Lightning
class EMACallback(pl.Callback):
    def __init__(self, decay=0.999, update_interval=1, ema_eval=True):
        super().__init__()
        self.decay = decay
        self.update_interval = update_interval
        self.shadow_params = {}
        self.ema_eval = ema_eval  # Whether to use EMA weights for validation
        self.original_params = None

    def on_train_start(self, trainer, pl_module):
        # Initialize shadow parameters if they don't exist
        if not self.shadow_params:
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.detach().clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only update at specified intervals
        if (trainer.global_step + 1) % self.update_interval == 0:
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    if name in self.shadow_params:
                        self.shadow_params[name] = self.shadow_params[name].to(param.device)
                        self.shadow_params[name] = self.shadow_params[name] * self.decay + param.detach() * (
                            1 - self.decay
                        )
                    else:
                        self.shadow_params[name] = param.detach().clone()

    def on_validation_start(self, trainer, pl_module):
        if self.ema_eval:
            # Save original parameters
            self.original_params = {}
            for name, param in pl_module.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    self.original_params[name] = param.detach().clone()
                    # Apply EMA weights for validation
                    param.data.copy_(self.shadow_params[name].to(param.device))

    def on_validation_end(self, trainer, pl_module):
        if self.ema_eval and self.original_params is not None:
            # Restore original parameters after validation
            for name, param in pl_module.named_parameters():
                if name in self.original_params:
                    param.data.copy_(self.original_params[name])
            self.original_params = None

    def on_test_start(self, trainer, pl_module):
        # Same as validation - use EMA weights for testing
        self.on_validation_start(trainer, pl_module)

    def on_test_end(self, trainer, pl_module):
        # Restore original weights after testing
        self.on_validation_end(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        # Same as validation - use EMA weights for testing
        self.on_validation_start(trainer, pl_module)

    def on_predict_end(self, trainer, pl_module):
        # Restore original weights after testing
        self.on_validation_end(trainer, pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Save EMA weights in checkpoint"""
        return {"shadow_params": self.shadow_params}

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        """Load EMA weights from checkpoint"""
        if "shadow_params" in checkpoint:
            self.shadow_params = checkpoint["shadow_params"]

    def state_dict(self):
        """Returns the state dict of the EMA Callback for manual checkpointing"""
        return {
            "shadow_params": self.shadow_params,
            "decay": self.decay,
            "update_interval": self.update_interval,
        }

    def load_state_dict(self, state_dict):
        """Loads the state from a state dict"""
        self.shadow_params = state_dict.get("shadow_params", {})
        self.decay = state_dict.get("decay", self.decay)
        self.update_interval = state_dict.get("update_interval", self.update_interval)


class SaveTestOutputsCallback(pl.Callback):
    """Callback to save test outputs with their corresponding inputs."""

    def __init__(
        self,
        save_path: Path,
        diff_steps: int,
        shortcut_size: int,
        start_example_idx: int = 1,
    ):
        """Initialize callback.

        :param save_path: path to save outputs
        :type save_path: Path
        :param diff_steps: total number of diffusion steps
        :type diff_steps: int
        :param shortcut_size: size of step between time steps
        :type shortcut_size: int
        :param start_example_idx: starting index for examples
        :type start_example_idx: int
        """
        super().__init__()
        self.save_path = save_path
        self.diff_steps = diff_steps
        self.shortcut_size = shortcut_size
        self.start_example_idx = start_example_idx

        # Calculate time steps using shortcut_size as step size
        self.time_steps = np.arange(0, diff_steps, shortcut_size, dtype=int)

        # Store outputs for each rank
        self.inputs = []
        self.predictions = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Store input sequences and predictions from each batch."""
        input_ids, predictions = outputs
        # Convert tensors to CPU and store
        self.inputs.append(input_ids.detach().cpu())
        self.predictions.append(predictions.detach().cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        """Save inputs and predictions in a format suitable for later processing."""
        # Create separate files for inputs and predictions
        inputs_file = self.save_path / f"inputs_rank{trainer.global_rank}.pt"
        preds_file = self.save_path / f"predictions_rank{trainer.global_rank}.pt"
        metadata_file = self.save_path / f"metadata_rank{trainer.global_rank}.json"

        # Concatenate all batches
        all_inputs = torch.cat(self.inputs, dim=0)
        all_predictions = torch.cat(self.predictions, dim=0)

        # Save tensors
        torch.save(all_inputs, inputs_file)
        torch.save(all_predictions, preds_file)

        # Save metadata
        metadata = {
            "num_examples": len(all_inputs),
            "time_steps": self.time_steps.tolist(),
            "input_shape": list(all_inputs.shape),
            "predictions_shape": list(all_predictions.shape),
            "start_example_idx": self.start_example_idx,
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved {len(all_inputs)} test examples:")
        print(f"   - Input sequences: {inputs_file}")
        print(f"   - Predictions: {preds_file}")
        print(f"   - Metadata: {metadata_file}")

        # Clear stored outputs after saving
        self.inputs = []
        self.predictions = []
