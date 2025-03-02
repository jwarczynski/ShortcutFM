import logging
from pathlib import Path

import lightning as pl
import torch
from datasets import Dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shortcutfm.batch import collate
from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import (
    CompositeCriterion,
    NllCriterion,
    SelfConditioningConsistencyCriterionDecorator, SelfConditioningFlowMatchingCriterionDecorator,
    X0ConsistencyCrterion,
    X0FlowMatchingCriterion,
    FLowNllCriterion,
)
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.shortcut_samplers import ShortcutSampler, TimeAndShorcutStampler, UniformSampler
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import EMACallback, GradientMonitor
from shortcutfm.train.pl.train_unit import TrainModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_lightning_trainer(cfg: TrainingConfig):
    """Initializes the Lightning trainer using parsed config."""

    # Initialize Model
    model = TransformerNetModelFactory(cfg.model).build()
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.module.parameters()) // 1_000_000}M")

    logger.info("Loading dataset...")
    train_ds = Dataset.load_from_disk(cfg.training_data_path)
    train_text_ds = TextDataset(train_ds)
    logger.info(f"Train dataset contains {len(train_ds)} samples.")

    val_ds = Dataset.load_from_disk(cfg.validation_data_path)
    val_text_ds = TextDataset(val_ds)
    logger.info(f"Validation dataset contains {len(val_ds)} samples.")

    # Define Criterions
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config_name)
    
    # Create base flow matching criterion
    flow_matching_criterion = X0FlowMatchingCriterion(
        model, 
        diffusion_steps=cfg.model.diffusion_steps,
        tokenizer=tokenizer
    )
    nll_criterion = NllCriterion(model, cfg.model.diffusion_steps)
    
    # Apply self-conditioning decorator if sc_rate > 0
    if cfg.model.sc_rate > 0:
        flow_matching_criterion = SelfConditioningFlowMatchingCriterionDecorator(
            flow_matching_criterion, 
            self_conditioning_ratio=cfg.model.sc_rate
        )

    # Create either CompositeCriterion or FlowNllCriterion based on self_consistency_ratio
    if cfg.self_consistency_ratio > 0:
        # Build criteria list for CompositeCriterion
        criteria = [flow_matching_criterion]
        weights = [cfg.flow_matching_loss_weight]

        # Add consistency criterion with optional self-conditioning decorator
        consistency_criterion = X0ConsistencyCrterion(model, cfg.model.diffusion_steps)
        if cfg.model.sc_rate > 0:
            consistency_criterion = SelfConditioningConsistencyCriterionDecorator(
                consistency_criterion, 
                self_conditioning_ratio=cfg.model.sc_rate
            )
        criteria.append(consistency_criterion)
        weights.append(cfg.consistency_loss_weight)

        # Add NLL criterion
        criteria.append(nll_criterion)
        weights.append(cfg.nll_loss_weight)

        # Create shortcut sampler
        shortcut_sampler = ShortcutSampler(
            diffusion_steps=cfg.model.diffusion_steps, 
            min_shortcut_size=cfg.model.min_shortcut_size
        )
        time_and_shortcut_sampler = TimeAndShorcutStampler(
            shortcut_sampler,
            cfg.model.diffusion_steps,
            cfg.model.min_shortcut_size
        )

        criterion = CompositeCriterion(
            criteria=tuple(criteria),
            criteria_weights=tuple(weights),
            model=model,
            diffusion_steps=cfg.model.diffusion_steps,
            self_consistency_ratio=cfg.self_consistency_ratio,
            sampler=time_and_shortcut_sampler,
        )
    else:
        sampler = UniformSampler(cfg.model.diffusion_steps)
        criterion = FLowNllCriterion(
            flow_matching_criterion=flow_matching_criterion,
            nll_criterion=nll_criterion,
            model=model,
            diffusion_steps=cfg.model.diffusion_steps,
            sampler=sampler
        )

    # Create Lightning module
    pl_model = TrainModule(criterion, cfg.optimizer.scheduler)

    # Setup data
    train_dataloader = DataLoader(
        train_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        shuffle=False
    )

    # Configure WandB Logger for Lightning
    wandb_logger = None
    if not cfg.dry_run and cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            id=cfg.wandb.run_id,
            resume=cfg.wandb.resume,
        )
        wandb_logger.watch(model.module, log="all")
        wandb_logger.log_hyperparams(cfg)

    # Configure checkpoint directory with wandb run ID if available
    checkpoint_dir = cfg.checkpoint.save_folder
    if wandb_logger is not None:
        checkpoint_dir = Path(checkpoint_dir) / f"run_{wandb_logger.experiment.id}"

    # Configure Callbacks
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
        LearningRateMonitor(logging_interval='step'),
        GradientMonitor(),
    ]

    # Add EMA callback if configured
    if cfg.ema is not None:
        callbacks.append(
            EMACallback(
                decay=cfg.ema.smoothing,
                update_interval=cfg.ema.update_interval,
            )
        )

    # Create Lightning Trainer
    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.gradient_clipping,
        gradient_clip_algorithm="norm",
        deterministic=cfg.deterministic,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=cfg.log_interval,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        val_check_interval=cfg.val_interval,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    return trainer, pl_model, train_dataloader, val_dataloader
