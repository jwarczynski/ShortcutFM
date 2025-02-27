import logging

import lightning as pl
import torch
from datasets import Dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, DistributedSampler

from shortcutfm.batch import collate
from shortcutfm.criteria import (
    CompositeCriterion,
    NllCriterion,
    SelfConditioningConsistencyCriterionDecorator, SelfConditioningFlowMatchingCriterionDecorator,
    X0ConsistencyCrterion,
    X0FlowMatchingCriterion,
)
from shortcutfm.model.config import TransformerNetModelConfig
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.shortcut_samplers import ShortcutSampler, TimeAndShorcutStampler
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import EMACallback, GradientMonitor
from shortcutfm.train.pl.train_unit import TrainModule

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_lightning_trainer(cfg):
    """Initializes the Lightning trainer using parsed config."""

    # Initialize Model
    transformer_model_config = TransformerNetModelConfig(**cfg.model_config)
    model = TransformerNetModelFactory(transformer_model_config).build()
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.module.parameters()) // 1_000_000}M")
    logger.info(f"Model: {model}")

    logger.info("Loading dataset...")

    train_ds = Dataset.load_from_disk(cfg.data_config.dataset_path)
    train_text_ds = TextDataset(train_ds)
    logger.info(f"Train dataset contains {len(train_ds)} samples.")

    val_ds = Dataset.load_from_disk(cfg.data_config.validation_data_path)
    val_text_ds = TextDataset(val_ds)
    logger.info(f"Validation dataset contains {len(val_ds)} samples.")

    # Define Criterions
    flow_matching_criterion = X0FlowMatchingCriterion(model, diffusion_steps=cfg.model_config.diffusion_steps)
    self_conditioning_flow_matching_criterion = SelfConditioningFlowMatchingCriterionDecorator(
        flow_matching_criterion, self_conditioning_ratio=cfg.model_config.sc_rate
    )

    consistency_criterion = X0ConsistencyCrterion(model, cfg.model_config.diffusion_steps)
    self_conditioning_consistency_criterion = SelfConditioningConsistencyCriterionDecorator(
        consistency_criterion, self_conditioning_ratio=cfg.model_config.sc_rate
    )

    nll_criterion = NllCriterion(model, cfg.model_config.diffusion_steps)

    shortcut_sampler = ShortcutSampler(
        diffusion_steps=cfg.model_config.diffusion_steps, min_shortcut_size=cfg.model_config.min_shortcut_size
    )
    time_and_shortcut_sampler = TimeAndShorcutStampler(
        shortcut_sampler,
        cfg.model_config.diffusion_steps,
        cfg.model_config.min_shortcut_size
    )

    criterion = CompositeCriterion(
        criteria=(self_conditioning_flow_matching_criterion, self_conditioning_consistency_criterion, nll_criterion,),
        criteria_weights=(1, 1, 1),
        model=model,
        diffusion_steps=cfg.model_config.diffusion_steps,
        self_consistency_ratio=cfg.training_config.self_consistency_ratio,
        sampler=time_and_shortcut_sampler,
    )

    # Setup optimizer configuration for the Lightning module
    optimizer_config = {
        'lr': cfg.training_config.lr,
        'weight_decay': cfg.training_config.weight_decay,
        'warmup_steps': cfg.training_config.warmup_steps,
        'start_lr': cfg.training_config.start_lr
    }

    # Create Lightning module
    pl_model = TrainModule(criterion, optimizer_config)

    # Setup data
    train_dataloader = DataLoader(
        train_text_ds,
        batch_size=cfg.data_config.batch_size,
        collate_fn=collate,
        shuffle=not torch.distributed.is_initialized(),
    )

    val_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.data_config.batch_size,
        collate_fn=collate,
        shuffle=not torch.distributed.is_initialized(),
    )

    # Configure WandB Logger for Lightning
    wandb_logger = None
    if not cfg.dry_run:
        wandb_logger = WandbLogger(
            project=cfg.training_config.project_name,
            name=cfg.training_config.run_name,
            id=None,
            resume="allow" if cfg.training_config.resume_wandb else None,
        )
        wandb_logger.watch(model.module, log="all")

    # Configure Callbacks
    callbacks = [
        ModelSummary(),
        ModelCheckpoint(
            dirpath=cfg.training_config.save_folder,
            save_top_k=cfg.training_config.num_checkpoints_to_keep,
            every_n_train_steps=cfg.training_config.save_interval,
            filename="{epoch}-{step}",
            save_last=True,
            monitor="train/loss",
        ),
        LearningRateMonitor(logging_interval='step'),
        GradientMonitor(),  # Custom callback for monitoring gradients - needs implementation
        EMACallback(  # Custom callback for EMA - needs implementation
            decay=cfg.training_config.ema.smoothing,
            update_interval=cfg.training_config.ema.update_interval,
        ),
    ]

    # Create Lightning Trainer
    trainer = pl.Trainer(
        max_steps=cfg.training_config.max_steps,
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training_config.gradient_clipping,
        gradient_clip_algorithm="norm",
        deterministic=cfg.training_config.deterministic,
        precision="16-mixed" if cfg.training_config.get("fp16", False) else "32",
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=cfg.training_config.log_interval,
        val_check_interval=cfg.training_config.val_interval,
        accumulate_grad_batches=cfg.training_config.accumulate_grad_batches,
    )

    return trainer, pl_model, train_dataloader, val_dataloader
