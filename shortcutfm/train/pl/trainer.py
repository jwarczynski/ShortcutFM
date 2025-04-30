import logging
from functools import partial
from pathlib import Path

import lightning as pl
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import Dataset
from shortcutfm.batch import collate
from shortcutfm.config import TrainingConfig
from shortcutfm.model.model import FlowMatchingModel
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import GradientMonitor
from shortcutfm.train.pl.train_unit import TrainModule
from shortcutfm.train.pl.trainer_factory import create_criterion, get_ema_callback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_dataloaders(cfg: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from config.

    :param cfg: Training configuration
    :type cfg: TrainingConfig
    :return: Train and validation dataloaders
    :rtype: tuple[DataLoader, DataLoader]
    """
    logger.info("Loading dataset...")
    train_ds = Dataset.load_from_disk(cfg.training_data_path)
    train_text_ds = TextDataset(train_ds)
    logger.info(f"Train dataset contains {len(train_ds)} samples.")

    val_ds = Dataset.load_from_disk(cfg.validation_data_path)
    val_text_ds = TextDataset(val_ds)
    logger.info(f"Validation dataset contains {len(val_ds)} samples.")

    configured_collate = partial(
        collate,
        mark_first_padding=cfg.padding_strategy.mark_first_padding,
        mark_second_padding=cfg.padding_strategy.mark_second_padding,
    )

    train_dataloader = DataLoader(
        train_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=configured_collate,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=configured_collate,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader


def create_wandb_logger(cfg: TrainingConfig, model: FlowMatchingModel) -> WandbLogger | None:
    """Create and configure WandB logger if enabled and not in dry run mode.

    :param cfg: Training configuration
    :type cfg: TrainingConfig
    :param model: Model to watch in WandB
    :type model: FlowMatchingModel
    :return: Configured WandB logger or None if disabled or in dry run mode
    :rtype: Optional[WandbLogger]
    """
    if not cfg.dry_run and cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            id=cfg.wandb.run_id,
            resume=cfg.wandb.resume,
        )
        wandb_logger.watch(model.module, log="all")
        wandb_logger.log_hyperparams(cfg.model_dump())
        return wandb_logger
    return None


def setup_checkpoint_directory_and_save_config(cfg: TrainingConfig, wandb_logger: WandbLogger | None) -> Path:
    """Set up checkpoint directory and save training config.

    :param cfg: Training configuration
    :type cfg: TrainingConfig
    :param wandb_logger: Optional WandB logger for run ID
    :type wandb_logger: Optional[WandbLogger]
    :return: Path to checkpoint directory
    :rtype: Path
    """
    # Configure checkpoint directory with wandb run ID if available
    checkpoint_dir = cfg.checkpoint.save_folder
    if wandb_logger is not None:
        checkpoint_dir = Path(checkpoint_dir) / f"run_{wandb_logger.experiment.id}"

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save training config in the checkpoint directory
    config_path = checkpoint_dir / "training_config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg.model_dump(), f)
    logger.info(f"Saved training config to {config_path}")

    return checkpoint_dir


def get_lightning_trainer(cfg: TrainingConfig):
    """Initializes the Lightning trainer using parsed config.

    :param cfg: Training configuration
    :type cfg: TrainingConfig
    :return: Tuple containing trainer, training unit, and dataloaders
    :rtype: tuple[pl.Trainer, TrainModule, DataLoader, DataLoader]
    """
    # Create Lightning module
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config_name)
    criterion = create_criterion(cfg, tokenizer=tokenizer)
    train_unit = TrainModule(
        criterion,
        cfg.optimizer.scheduler,
        tokenizer=tokenizer,
        prediction_shortcut_size=cfg.prediction_shortcut_size,
        denoising_step_size=cfg.denoising_step_size,
        num_val_batches_to_log=cfg.num_val_batches_to_log,
        num_timestep_bins=cfg.num_timestep_bins,
        log_train_predictions_every_n_epochs=cfg.log_train_predictions_every_n_epochs,
        log_train_predictions_from_n_epochs=cfg.log_train_predictions_from_n_epochs,
    )

    train_dataloader, val_dataloader = create_dataloaders(cfg)

    wandb_logger = create_wandb_logger(cfg, train_unit.criterion.model)

    checkpoint_dir = setup_checkpoint_directory_and_save_config(cfg, wandb_logger)

    callbacks = [
        ModelSummary(),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            save_top_k=cfg.checkpoint.save_top_k,
            every_n_train_steps=cfg.checkpoint.save_interval,
            filename="{epoch}-{step}",
            save_last=cfg.checkpoint.save_last,
            monitor=cfg.checkpoint.monitor,
            mode=cfg.checkpoint.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
        GradientMonitor(),
    ]

    if ema_callback := get_ema_callback(cfg, cfg.checkpoint.path):
        callbacks.append(ema_callback)

    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        precision=32,
        gradient_clip_val=cfg.gradient_clipping,
        gradient_clip_algorithm="norm",
        deterministic=cfg.deterministic,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.num_gpus,
        log_every_n_steps=cfg.log_interval,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        val_check_interval=cfg.val_interval,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        overfit_batches=int(cfg.overfit_batches) if cfg.overfit_batches >= 1 else cfg.overfit_batches,
    )

    return trainer, train_unit, train_dataloader, val_dataloader
