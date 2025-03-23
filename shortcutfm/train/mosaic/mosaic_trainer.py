import logging

import torch
from composer import DataSpec, Trainer
from composer.algorithms import GradientClipping
from composer.algorithms.ema.ema import EMA
from composer.callbacks import CheckpointSaver, LRMonitor, OptimizerMonitor
from composer.loggers import WandBLogger
from composer.utils import dist
from datasets import Dataset
from torch.utils.data import DataLoader

from shortcutfm.batch import collate
from shortcutfm.criteria import (
    CompositeCriterion,
    NllCriterion, SelfConditioningConsistencyCriterionDecorator, SelfConditioningFlowMatchingCriterionDecorator,
    X0ConsistencyCriterion, X0FlowMatchingCriterion,
)
from shortcutfm.model.config import TransformerNetModelConfig
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.shortcut_samplers import ShortcutSampler
from shortcutfm.step_sample import ShortcutAwareSampler
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.mosaic.mosaic_train_unit import LogGradientsAndNormCallback, MetricTracker, TrainUnit

from shortcutfm.utils.nn import MyleLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_composer_trainer(cfg):
    """Initializes the trainer using parsed config."""

    # Initialize Model
    transformer_model_config = TransformerNetModelConfig(**cfg.model_config)
    model = TransformerNetModelFactory(transformer_model_config).build()
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.module.parameters()) // 1_000_000}M")
    logger.info(f"Model: {model.module}")

    logger.info("Loading dataset...")
    ds = Dataset.load_from_disk(cfg.data_config.dataset_path)
    text_ds = TextDataset(ds)
    logger.info(f"Dataset contains {len(ds)} samples.")

    sampler = dist.get_sampler(text_ds, shuffle=True)
    dataloader = DataLoader(
        text_ds,
        batch_size=cfg.data_config.batch_size,
        collate_fn=collate,
        sampler=sampler,
    )

    # Define Criterions
    flow_matching_criterion = X0FlowMatchingCriterion(model, diffusion_steps=cfg.model_config.diffusion_steps)
    self_conditioning_flow_matching_cirterion = SelfConditioningFlowMatchingCriterionDecorator(
        flow_matching_criterion, self_conditioning_ratio=cfg.model_config.sc_rate
    )
    consistency_criterion = X0ConsistencyCriterion(model, cfg.model_config.diffusion_steps)
    slef_conditioning_consistency_criterion = SelfConditioningConsistencyCriterionDecorator(
        consistency_criterion, self_conditioning_ratio=cfg.model_config.sc_rate
    )
    nll_criterion = NllCriterion(model, cfg.model_config.diffusion_steps)

    criterion = CompositeCriterion(
        criteria=(self_conditioning_flow_matching_cirterion, slef_conditioning_consistency_criterion, nll_criterion,),
        criteria_weights=(1, 1, 1),
        model=model,
        diffusion_steps=cfg.model_config.diffusion_steps,
        self_consistency_ratio=cfg.training_config.self_consistency_ratio,
        time_scheduler=ShortcutAwareSampler(cfg.model_config.diffusion_steps, min_shortcut_size=2 ** 6),
        shortcut_sampler=ShortcutSampler(cfg.model_config.diffusion_steps, min_shorcut_size=2 ** 6),
    )

    # Instantiate Callbacks
    metrics_tracker = MetricTracker()
    gradients_tracker_callback = LogGradientsAndNormCallback()
    checkpoint_callback = CheckpointSaver(
        folder=cfg.training_config.save_folder,
        save_interval=f"{cfg.training_config.save_interval}ba",
        num_checkpoints_to_keep=cfg.training_config.num_checkpoints_to_keep,
        overwrite=cfg.training_config.checkpoint.overwrite,
    )

    callbacks = [
        checkpoint_callback,
        metrics_tracker,
        # gradients_tracker_callback,
        LRMonitor(),
        OptimizerMonitor()
    ]

    # Initialize WandB Logger
    wandb_logger = WandBLogger(
        project=cfg.training_config.project_name,
        name=cfg.training_config.run_name,
        rank_zero_only=True,
        init_kwargs={"id": None, "resume": cfg.training_config.resume_wandb}
    )

    # Data Specification
    data_spec = DataSpec(
        dataloader=dataloader,
        num_samples=len(text_ds) // cfg.data_config.batch_size,
        get_num_tokens_in_batch=lambda batch: batch.numel(),
        get_num_samples_in_batch=lambda batch: batch.size(),
        split_batch=lambda batch, idx: batch.split(idx),
    )

    # Define Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.module.parameters(),
        lr=cfg.training_config.lr,
        weight_decay=cfg.training_config.weight_decay
    )
    scheduler = MyleLR(optimizer, cfg.training_config.warmup_steps, cfg.training_config.start_lr)

    # Define Training Algorithms
    algo = [
        EMA(
            smoothing=cfg.training_config.ema.smoothing,
            half_life=cfg.training_config.ema.half_life,
            update_interval=f"{cfg.training_config.ema.update_interval}ba",
        ),
        GradientClipping(clipping_type="norm", clipping_threshold=cfg.training_config.gradient_clipping),
    ]

    train_unit = TrainUnit(criterion)
    for name, param in model.module.named_parameters():
        print(name, param.requires_grad)

    trainer = Trainer(
        model=train_unit,
        train_dataloader=data_spec,
        max_duration=f"{cfg.training_config.max_steps}ba",
        loggers=[wandb_logger],
        callbacks=callbacks,
        optimizers=optimizer,
        schedulers=scheduler,
        step_schedulers_every_batch=True,
        algorithms=algo,
        deterministic_mode=cfg.training_config.deterministic,
        seed=cfg.training_config.seed,
        device="gpu" if torch.cuda.is_available() else "cpu",
    )

    return trainer
