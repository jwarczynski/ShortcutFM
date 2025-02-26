from typing import Any, Optional, Sequence, Union

import torch
import torch.distributed as dist
import wandb
from composer import Callback, Logger
from composer.core import State
from composer.loggers import WandBLogger
from composer.models import ComposerModel
from composer.utils import dist as composer_dist
from torch import Tensor
from torchmetrics import Metric

from shortcutfm.batch import EncoderBatch
from shortcutfm.criteria import Criterion


class TrainUnit(ComposerModel):

    def __init__(self, criterion: Criterion) -> None:
        super().__init__()
        self.criterion = criterion
        self.device = None

    def forward(self, batch: EncoderBatch) -> Any:
        if self.device is None:
            self.device = next(self.criterion.model.module.parameters()).device

        batch.seqs.to(self.device)
        batch.padding_mask.to(self.device)
        batch.input_ids_mask.to(self.device)
        return self.criterion(batch)

    def loss(self, outputs: dict[str, Tensor], batch: EncoderBatch, *args, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        return outputs["loss"]

    def eval_forward(self, batch: EncoderBatch, outputs: Optional[Any] = None, ) -> Any:
        return outputs["loss"]

    def get_metrics(self, is_train=False) -> dict[str, Metric]:
        ...

    def update_metric(self, batch, outputs, metric) -> None:
        ...


class MetricTracker(Callback):
    def __init__(self, total_elements=0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: handle resuming from a checkpoint
        self.total_elements = total_elements  # Accumulated count

    def after_train_batch(self, state, logger):
        numel = state.batch.numel()  # Number of elements in batch
        self.total_elements += numel

        # Extract losses from model outputs
        outputs = state.outputs
        if isinstance(outputs, dict):  # Ensure it's a dict
            loss_logs = {
                f"loss/{key}": value.item() for key, value in outputs.items()
                if isinstance(value, Tensor) and key != "loss"  # total loss automatically logged
            }
        else:
            loss_logs = {}

        # Log batch size, total elements, and losses
        logger.log_metrics(
            {
                "batch_size": state.batch.size(), "num_elements": numel, "total_elements": self.total_elements,
                **loss_logs
            }
        )


class LogGradientsAndNormCallback(Callback):
    def after_train_batch(self, state: State, logger: Logger):
        if not isinstance(logger, WandBLogger):
            return

        model = state.model
        gradients = {}
        total_norm = torch.tensor(0.0, device=state.device)  # Store on the correct device

        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())

                # Compute local gradient L2 norm
                param_norm = param.grad.norm(2)
                total_norm += param_norm ** 2  # Sum of squares

        # Aggregate across all devices
        if composer_dist.get_world_size() > 1:
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM)  # Sum norms across all processes

        total_norm = torch.sqrt(total_norm).item()  # Final L2 norm

        # Only rank 0 logs to W&B to avoid redundant logs
        if composer_dist.get_global_rank() == 0:
            gradients["gradient_norm"] = total_norm
            wandb.log(gradients, step=state.timestamp.batch.value)
