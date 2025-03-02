from typing import Optional

import lightning as pl
from numpy import dtype, ndarray
from torch import Tensor
from torch.optim import AdamW

from shortcutfm.batch import EncoderBatch
from shortcutfm.criteria import Criterion
from shortcutfm.decoding.prediction_strategies import PredictionStrategy
from shortcutfm.train.optim import SchedulerFactory
from shortcutfm.config import SchedulerConfig


class TrainModule(pl.LightningModule):

    def __init__(
            self,
            criterion: Criterion,
            optimizer_config: SchedulerConfig,
            prediction_strategy: Optional[PredictionStrategy] = None,
            prediction_shorcut_size: int = 1,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.optimizer_config = optimizer_config
        self.prediction_strategy = prediction_strategy
        self.prediction_shorcut_size = prediction_shorcut_size
        self.save_hyperparameters(ignore=['criterion', 'prediction_strategy'])

    def forward(self, batch: EncoderBatch) -> dict[str, Tensor]:
        return self.criterion(batch)

    def training_step(self, batch: EncoderBatch, batch_idx: int) -> Tensor:
        outputs = self(batch)
        self.log_dict(
            {f"train/{k}": v for k, v in outputs.items()},
            on_step=True, on_epoch=False, prog_bar=True
        )
        # TODO: shuld we weight the loss by the number of training tokens?
        return outputs["loss"]

    def validation_step(self, batch: EncoderBatch, batch_idx: int) -> Tensor:
        outputs = self(batch)
        self.log_dict(
            {f"val/{k}": v for k, v in outputs.items()},
            on_step=False, on_epoch=True, prog_bar=True
        )
        # TODO: log the outputs
        return outputs["loss"]

    def test_step(self, batch: EncoderBatch, batch_idx: int) -> tuple[Tensor, Tensor]:
        """Run test step and return both input sequences and model predictions.
        
        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - input_ids: Input token sequences [batch_size, seq_len]
                - predictions: Model predictions [batch_size, num_steps, seq_len]
        """
        predictions = self.criterion.denoise(batch, self.prediction_shorcut_size)
        return batch.seqs, predictions

    def _predict_step(self, batch: EncoderBatch, batch_idx: int) -> ndarray[str, dtype[str]]:
        if not self.prediction_strategy:
            raise "Predicition Strategu Not provided. Cannot perform densoing"

        return self.prediction_strategy(batch, self.shortcut_size)

    def set_prediction_shorcut_size(self, shortcut_size: int) -> None:
        self.prediction_shorcut_size = shortcut_size

    def configure_optimizers(self):
        optimizer = AdamW(
            self.criterion.model.module.parameters(),
            lr=self.optimizer_config.lr,
            weight_decay=self.optimizer_config.weight_decay
        )

        scheduler = {
            'scheduler': self._get_lr_scheduler(optimizer),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    def _get_lr_scheduler(self, optimizer):
        return SchedulerFactory.get_scheduler(
            name=self.optimizer_config.type,
            optimizer=optimizer,
            config=self.optimizer_config
        )
