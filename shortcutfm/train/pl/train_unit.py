from itertools import islice
from typing import Literal, Optional

import lightning as pl
import numpy as np
import torch
from numpy import dtype, ndarray
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from transformers import PreTrainedTokenizer
import wandb

from shortcutfm.batch import EncoderBatch
from shortcutfm.config import SchedulerConfig
from shortcutfm.criteria import CompositeCriterion
from shortcutfm.decoding.prediction_strategies import PredictionStrategy
from shortcutfm.train.optim import SchedulerFactory


class TrainModule(pl.LightningModule):

    def __init__(
            self,
            criterion: CompositeCriterion,
            optimizer_config: SchedulerConfig,
            tokenizer: Optional[PreTrainedTokenizer] = None,
            prediction_strategy: Optional[PredictionStrategy] = None,
            prediction_shortcut_size: int = 64,
            denoising_step_size: int = 32,
            num_val_batches_to_log: int = 2,
            num_timestep_bins: int = 4,  # Number of bins for timestep logging
            log_train_predictions_every_n_epochs: int = 100,  # Number of epochs between train prediction logging
            log_train_predictions_from_n_epochs: int = 1000,  # Number of epochs between train prediction logging
    ) -> None:
        super().__init__()
        self.model = criterion
        self.optimizer_config = optimizer_config

        self.tokenizer = tokenizer
        self.prediction_strategy = prediction_strategy
        self.prediction_shortcut_size = prediction_shortcut_size
        self.denoising_step_size = denoising_step_size
        self.num_val_batches_to_log = num_val_batches_to_log
        self.log_train_predictions_every_n_epochs = log_train_predictions_every_n_epochs
        self.log_train_predictions_from_n_epochs = log_train_predictions_from_n_epochs
        self.predictions = []  # For validation predictions
        self.train_predictions = []  # For training predictions
        self.last_train_batch = None  # Store last batch for full denoising

        # Setup timestep bins using linear spacing
        # max_timestep = self.criterion.diffusion_steps  # Maximum timestep value
        # Create linearly spaced bin edges
        # self.timestep_bins = np.linspace(0, max_timestep, num_timestep_bins + 1, dtype=int)

        # Initialize dictionaries to store losses for each component and bin
        self.timestep_losses = {}  # Will be populated with loss components in training_step

        # Initialize lists to store exact timesteps and shortcuts for histogram logging
        self.timesteps_for_histogram = []
        self.shortcuts_for_histogram = []

        self.save_hyperparameters(ignore=['criterion', 'prediction_strategy', 'tokenizer'])

    def forward(self, batch: EncoderBatch) -> dict[str, Tensor]:
        embeddings = self.model.get_embeddings(batch.seqs)
        output = self.model(embeddings, torch.zeros(batch.seqs.shape[0], device=embeddings.device))
        fm_loss = torch.nn.functional.mse_loss(output, embeddings, reduction="none")
        return {"loss": fm_loss}

    def training_step(self, batch: EncoderBatch, batch_idx: int) -> Tensor:
        outputs = self(batch)

        # Store last batch of the epoch for full denoising
        if batch_idx == 0:  # Use first batch for consistency
            self.last_train_batch = batch

        # Log only loss-related metrics
        loss_metrics = {
            f"train/{k}": v.mean() for k, v in outputs.items()
            if "loss" in k.lower()
        }
        self.log_dict(
            loss_metrics,
            on_step=True, on_epoch=False, prog_bar=True
        )

        # Store exact timesteps and shortcuts for histogram logging
        if "timestep" in outputs:
            self.timesteps_for_histogram.extend(outputs["timestep"].detach().cpu().numpy().tolist())
        if "shortcut" in outputs:
            self.shortcuts_for_histogram.extend(outputs["shortcut"].detach().cpu().numpy().tolist())

        # Process and store timestep losses
        # self._process_timestep_losses(outputs)

        return outputs["loss"].mean()

    def _process_timestep_losses(self, outputs: dict[str, Tensor]) -> None:
        """Process and store losses for each timestep bin.

        This method processes the loss components for each timestep,
        bins them according to timestep values, and stores them for later logging.

        :param outputs: Dictionary containing model outputs including losses and timesteps
        :type outputs: dict[str, Tensor]
        """
        if "timestep" not in outputs:
            return

        timesteps = outputs["timestep"]  # Shape: [batch_size]

        # Process each loss term except total loss
        for key, value in outputs.items():
            if "loss" in key.lower() and key not in ["loss", "embedding_loss", "isotropy_loss"]:
                # Initialize storage for this loss component if not exists
                if key not in self.timestep_losses:
                    self.timestep_losses[key] = [[] for _ in range(len(self.timestep_bins) - 1)]

                losses = value.detach()  # Shape: [batch_size] or scalar

                # If loss is scalar, expand it to match batch size
                if losses.ndim == 0:
                    losses = losses.expand(timesteps.shape[0])

                # Process each timestep-loss pair in the batch
                for timestep, loss in zip(timesteps, losses):
                    bin_idx = self._get_timestep_bin(timestep.item())
                    if 0 <= bin_idx < len(self.timestep_bins) - 1:
                        self.timestep_losses[key][bin_idx].append(loss.item())

    def _get_timestep_bin(self, timestep: int) -> int:
        """Get the bin index for a given timestep using linear bins."""
        bin_size = self.criterion.diffusion_steps // (len(self.timestep_bins) - 1)
        return min(timestep // bin_size, len(self.timestep_bins) - 2)

    def on_train_epoch_end(self) -> None:
        """Log average losses for each timestep bin and full denoising predictions for one batch."""
        # self._log_timestep_bin_losses()
        # self._log_sampling_histograms()
        # self._process_train_batch_predictions()
        # self.log_anisotropy()
        pass

    def _log_timestep_bin_losses(self) -> None:
        """Log average losses for each timestep bin and clear the loss storage.

        This method processes the accumulated losses for each timestep bin,
        computes their averages, logs them, and then clears the storage for the next epoch.
        """
        # Log timestep bin losses
        for loss_name, bins in self.timestep_losses.items():
            for bin_idx, losses in enumerate(bins):
                if losses:  # If we have losses for this bin
                    avg_loss = np.mean(losses)
                    bin_start = self.timestep_bins[bin_idx]
                    bin_end = self.timestep_bins[bin_idx + 1]

                    # Log average loss for this timestep bin and component
                    metric_name = f"train/{loss_name}_t{bin_start:04d}_t{bin_end:04d}"
                    self.log(
                        metric_name,
                        avg_loss,
                        on_step=False,
                        on_epoch=True
                    )

        # Clear losses for next epoch
        self.timestep_losses = {
            key: [[] for _ in range(len(self.timestep_bins) - 1)]
            for key in self.timestep_losses.keys()
        }

    def _log_sampling_histograms(self) -> None:
        """Log histograms of timesteps and shortcuts sampled during training.
        
        This method creates and logs histograms to track the distribution of:
        - Timesteps that were sampled during training
        - Shortcut sizes that were used
        It also logs the count of samples for each.
        """
        if self.timesteps_for_histogram:
            self.logger.experiment.log({
                "train/timesteps_histogram": wandb.Histogram(self.timesteps_for_histogram),
                "train/timesteps_count": len(self.timesteps_for_histogram)
            })
            self.timesteps_for_histogram = []  # Clear for next epoch
            
        if self.shortcuts_for_histogram:
            self.logger.experiment.log({
                "train/shortcuts_histogram": wandb.Histogram(self.shortcuts_for_histogram),
                "train/shortcuts_count": len(self.shortcuts_for_histogram)
            })
            self.shortcuts_for_histogram = []  # Clear for next epoch

    def _process_train_batch_predictions(self) -> None:
        """Process and log predictions for the saved training batch.

        This method performs full denoising on the last saved training batch,
        computes cross entropy loss, and logs the predictions along with their
        source and reference texts.
        """
        log_text = (
                self.trainer.current_epoch % self.log_train_predictions_every_n_epochs == 0 and
                self.trainer.current_epoch >= self.log_train_predictions_from_n_epochs
        )
        if self.last_train_batch is not None:
            self._process_batch_predictions(
                batch=self.last_train_batch,
                predictions_list=self.train_predictions,
                stage="train",
                create_entries=log_text,
            )

            # Log the predictions table
            if self.train_predictions and hasattr(self.logger, "log_table"):
                columns = ["epoch", "sample_idx", "source", "reference", "predicted", "cross_entropy"]
                self.logger.log_table(
                    "train/predictions",
                    columns=columns,
                    data=self.train_predictions
                )

    def _process_batch_predictions(
            self,
            batch: EncoderBatch,
            predictions_list: list,
            batch_idx: Optional[int] = None,
            stage: Literal["val", "train"] = None,
            create_entries: bool = True,
    ) -> float:
        """Process a batch for predictions and store results.

        This method handles the full denoising process, computes cross entropy loss,
        and stores the predictions for later logging.

        :param batch: The batch to process
        :type batch: EncoderBatch
        :param predictions_list: List to store predictions in
        :type predictions_list: list
        :param batch_idx: Optional batch index to include in predictions
        :type batch_idx: Optional[int]
        :param stage: Stage of training (train/val) for logging
        :type stage: Literal["val, train"]
        :return: The cross entropy loss for this batch
        :rtype: float
        """
        with torch.no_grad():
            # Get logits from denoising process
            predictions: Tensor = self.criterion.denoise(
                batch=batch,
                shortcut_size=self.prediction_shortcut_size,
                probe_every_step=False,  # Only get final predictions
                return_logits=True,  # Get logits for cross entropy
                step_size=self.denoising_step_size,
            )

            # Compute masked cross entropy loss
            ce_loss = self._compute_masked_cross_entropy(predictions, batch)

            if create_entries:
                # Get token IDs for decoding
                predicted_tokens = predictions.argmax(dim=-1)

                # Extract text parts
                source_text, reference_text, predicted_text = self._extract_text_parts(
                    batch, predicted_tokens
                )

                # Create and store prediction entries
                prediction_entries = self._create_prediction_entries(
                    source_text=source_text,
                    reference_text=reference_text,
                    predicted_text=predicted_text,
                    ce_losses=ce_loss,
                    batch_idx=batch_idx
                )
                predictions_list.extend(prediction_entries)

            # Log cross entropy if stage is provided
            if stage:
                self.log(f"{stage}/full_denoising_ce", ce_loss.mean().item(), on_step=False, on_epoch=True)

            return ce_loss.mean().item()

    def _compute_masked_cross_entropy(
            self,
            predictions: Tensor,
            batch: EncoderBatch
    ) -> Tensor:
        """Compute masked cross entropy loss for predictions.

        :param predictions: Model predictions [batch_size, seq_len, vocab_size]
        :type predictions: Tensor
        :param batch: The batch containing target sequences and masks
        :type batch: EncoderBatch
        :return: Masked cross entropy loss [batch_size, seq_len]
        :rtype: Tensor
        """
        # Compute cross entropy between sequences
        ce_loss = F.cross_entropy(
            predictions.view(-1, predictions.size(-1)),
            batch.seqs.view(-1),
            reduction="none"
        ).view(batch.seqs.shape)

        # Apply masks and normalize
        loss_mask = batch.input_ids_mask * batch.padding_mask
        ce_loss = (ce_loss * loss_mask).sum(-1) / loss_mask.sum(-1)

        return ce_loss

    def _extract_text_parts(
            self,
            batch: EncoderBatch,
            predicted_tokens: Tensor
    ) -> tuple[list[str], list[str], list[str]]:
        """Extract source, reference and predicted text parts.

        :param batch: The batch containing sequences and masks
        :type batch: EncoderBatch
        :param predicted_tokens: Predicted token IDs [batch_size, seq_len]
        :type predicted_tokens: Tensor
        :return: Tuple of (source_texts, reference_texts, predicted_texts)
        :rtype: tuple[list[str], list[str], list[str]]
        """
        # Split sequences into source and reference using input_mask
        source_tokens = batch.seqs.clone()
        reference_tokens = batch.seqs.clone()

        # Zero out reference/source parts based on input_mask
        source_tokens[batch.input_ids_mask == 1] = self.tokenizer.pad_token_id
        reference_tokens[batch.input_ids_mask == 0] = self.tokenizer.pad_token_id

        # Decode each part separately
        source_text = self.tokenizer.batch_decode(source_tokens, skip_special_tokens=True)
        reference_text = self.tokenizer.batch_decode(reference_tokens, skip_special_tokens=True)
        predicted_text = self.tokenizer.batch_decode(predicted_tokens, skip_special_tokens=False)

        return source_text, reference_text, predicted_text

    def _create_prediction_entries(
            self,
            source_text: list[str],
            reference_text: list[str],
            predicted_text: list[str],
            ce_losses: Tensor,
            batch_idx: Optional[int] = None,
            max_entries: int = 8,
    ) -> list[list]:
        """Create prediction entries for logging.

        :param source_text: List of source text parts
        :type source_text: list[str]
        :param reference_text: List of reference text parts
        :type reference_text: list[str]
        :param predicted_text: List of predicted text parts
        :type predicted_text: list[str]
        :param ce_loss: Cross entropy loss value
        :type ce_loss: float
        :param batch_idx: Optional batch index to include in entries
        :type batch_idx: Optional[int]
        :return: List of prediction entries
        :rtype: list[list]
        """
        entries = []
        for i, (src, ref, pred, ce_loss) in enumerate(
                islice(zip(source_text, reference_text, predicted_text, ce_losses), max_entries)
        ):
            prediction_entry = [
                self.current_epoch,
                src.strip(),  # Remove any padding artifacts
                ref.strip(),  # Remove any padding artifacts
                pred,
                ce_loss.item()
            ]

            # Insert batch_idx if provided
            if batch_idx is not None:
                prediction_entry.insert(1, batch_idx)
                prediction_entry.insert(2, i)
            else:
                prediction_entry.insert(1, i)

            entries.append(prediction_entry)

        return entries

    def log_anisotropy(self):
        """Log the anisotropy of the model embeddings."""
        if hasattr(self.criterion.model.module, "word_embedding"):
            embedding_weights = self.criterion.model.module.word_embedding.weight
            anisotropy = self._calculate_anisotropy(embedding_weights)
            self.log("train/anisotropy", anisotropy, on_step=False, on_epoch=True)

    @torch.no_grad()
    def _calculate_anisotropy(self, model_emb):
        model_emb = model_emb / torch.norm(model_emb, dim=-1, keepdim=True)
        cos_similarity = torch.mm(model_emb, model_emb.T).sum() - model_emb.size(0)
        anisotropy = cos_similarity / model_emb.size(0) / (model_emb.size(0) - 1)
        return anisotropy

    def validation_step(self, batch: EncoderBatch, batch_idx: int) -> Tensor:
        outputs = self(batch)
        self.log_dict(
            {
                f"val/{k}": v.mean() for k, v in outputs.items()
                if "loss" in k.lower()
            },
            on_step=False, on_epoch=True, prog_bar=True
        )

        # Perform full denoising and log text for a few validation batches
        # if (
        #         batch_idx < self.num_val_batches_to_log and
        #         self.trainer.current_epoch >= self.log_train_predictions_from_n_epochs
        # ):
        #     self._process_validation_predictions(batch, batch_idx)

        return outputs["loss"]

    def _process_validation_predictions(self, batch: EncoderBatch, batch_idx: int) -> float:
        """Process a batch for validation predictions and store results.

        This method handles the full denoising process, computes cross entropy loss,
        and stores the predictions for later logging.

        :param batch: The validation batch to process
        :type batch: EncoderBatch
        :param batch_idx: The index of the current batch
        :type batch_idx: int
        :return: The cross entropy loss for this batch
        :rtype: float
        """
        return self._process_batch_predictions(
            batch=batch,
            predictions_list=self.predictions,
            batch_idx=batch_idx,
            stage="val"
        )

    # def on_validation_end(self) -> None:
    #     """Log all predictions from the epoch to the table."""
    #     if self.predictions and hasattr(self.logger, "log_table"):
    #         columns = ["epoch", "batch", "sample_idx", "source", "reference", "predicted", "cross_entropy"]
    #         self.logger.log_table(
    #             "val/predictions",
    #             columns=columns,
    #             data=self.predictions
    #         )

    def test_step(self, batch: EncoderBatch, batch_idx: int) -> tuple[Tensor, Tensor]:
        """Run test step and return both input sequences and model predictions.
        
        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - input_ids: Input token sequences [batch_size, seq_len]
                - predictions: Model predictions [batch_size, num_steps, seq_len]
        """
        predictions = self.criterion.denoise(batch, self.prediction_shortcut_size)
        return batch.seqs, predictions

    def _predict_step(self, batch: EncoderBatch, batch_idx: int) -> ndarray[str, dtype[str]]:
        if not self.prediction_strategy:
            raise "Predicition Strategu Not provided. Cannot perform densoing"

        return self.prediction_strategy(batch, self.shortcut_size)

    def set_prediction_shortcut_size(self, shortcut_size: int) -> None:
        self.prediction_shortcut_size = shortcut_size

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.module.parameters(),
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
