import lightning as pl
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
            total_norm = total_norm ** 0.5
            trainer.logger.log_metrics(
                {"grad_norm/total": total_norm, **grad_norm_dict},
                step=trainer.global_step
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
                                1 - self.decay)
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
        return {"shadow_params": self.shadow_params, "decay": self.decay, "update_interval": self.update_interval}

    def load_state_dict(self, state_dict):
        """Loads the state from a state dict"""
        self.shadow_params = state_dict.get("shadow_params", {})
        self.decay = state_dict.get("decay", self.decay)
        self.update_interval = state_dict.get("update_interval", self.update_interval)


def save_model_with_ema(model, ema_callback, filepath):
    """
    Save a model checkpoint including EMA weights

    Args:
        model: The PyTorch Lightning module
        ema_callback: The EMACallback instance
        filepath: Path where to save the checkpoint
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "ema_state_dict": ema_callback.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint with EMA weights saved to {filepath}")


def load_model_with_ema(model_class, filepath, ema_weights=False):
    """
    Load a model from checkpoint, optionally with EMA weights

    Args:
        model_class: The model class to instantiate
        filepath: Path to the checkpoint file
        ema_weights: Whether to use EMA weights instead of regular weights

    Returns:
        model: The loaded model
    """
    checkpoint = torch.load(filepath, map_location='cpu')

    # Create model instance (this assumes your model can be instantiated with default args)
    # Adjust as needed for your model initialization
    model = model_class()

    if ema_weights and "ema_state_dict" in checkpoint:
        # Load EMA weights into the model
        ema_state = checkpoint["ema_state_dict"]
        if "shadow_params" in ema_state:
            # Apply EMA weights directly to model
            model.load_state_dict(ema_state["shadow_params"])
            print(f"Model loaded with EMA weights from {filepath}")
        else:
            model.load_state_dict(checkpoint["state_dict"])
            print(f"EMA state dict found but no shadow params. Using regular weights.")
    else:
        # Load regular model weights
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded with regular weights from {filepath}")

    # If you want to also restore the EMA callback state
    ema_callback = EMACallback()
    if "ema_state_dict" in checkpoint:
        ema_callback.load_state_dict(checkpoint["ema_state_dict"])

    return model, ema_callback
