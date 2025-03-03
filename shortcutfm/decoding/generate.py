import logging
import sys
from pathlib import Path
from typing import Optional

import lightning as pl
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shortcutfm.batch import collate
from shortcutfm.config import GenerationConfig, TrainingConfig
from shortcutfm.criteria import (
    X0FlowMatchingCriterion, CompositeCriterion, SelfConditioningFlowMatchingCriterionDecorator,
    NllCriterion, X0ConsistencyCrterion, FlowNllCriterion, SelfConditioningConsistencyCriterionDecorator
)
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.model.model import FlowMatchingModel
from shortcutfm.shortcut_samplers import UniformSampler, ShortcutSampler, TimeAndShorcutStampler
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import EMACallback, SaveTestOutputsCallback
from shortcutfm.train.pl.train_unit import TrainModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: str, args_list: list[str]) -> GenerationConfig:
    """Parse and validate generation config from YAML file"""
    if not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(yaml_cfg, om.from_cli(args_list))
    merged_dict = OmegaConf.to_container(merged_cfg, resolve=True)

    # Use model_validate to create config which will trigger field validators
    cfg = GenerationConfig.model_validate(merged_dict)
    return cfg


def create_criterion(training_cfg: TrainingConfig) -> CompositeCriterion | FlowNllCriterion:
    """Create model, tokenizer and criterion based on training config.
    
    Returns:
        tuple: (criterion, model) tuple containing the criterion and flow matching model
    """
    # Initialize model
    model = TransformerNetModelFactory(training_cfg.model).build()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_cfg.model.config_name)

    # Create base flow matching criterion
    flow_matching_criterion = X0FlowMatchingCriterion(
        model,
        diffusion_steps=training_cfg.model.diffusion_steps,
        tokenizer=tokenizer
    )

    # Apply self-conditioning decorator if sc_rate > 0
    if training_cfg.model.sc_rate > 0:
        flow_matching_criterion = SelfConditioningFlowMatchingCriterionDecorator(
            flow_matching_criterion,
            self_conditioning_ratio=training_cfg.model.sc_rate
        )

    # Create either CompositeCriterion or FlowNllCriterion based on self_consistency_ratio
    if training_cfg.self_consistency_ratio > 0:
        criterion = _create_composite_criterion(
            flow_matching_criterion,
            model,
            training_cfg
        )
    else:
        criterion = _create_flow_nll_criterion(
            flow_matching_criterion,
            model,
            training_cfg
        )

    return criterion


def _create_composite_criterion(
        flow_matching_criterion: X0FlowMatchingCriterion,
        model: FlowMatchingModel,
        training_cfg: TrainingConfig
) -> CompositeCriterion:
    """Create composite criterion with consistency and NLL components."""
    # Build criteria list for CompositeCriterion
    criteria = [flow_matching_criterion]
    weights = [training_cfg.flow_matching_loss_weight]

    # Add consistency criterion with optional self-conditioning decorator
    consistency_criterion = X0ConsistencyCrterion(model, training_cfg.model.diffusion_steps)
    if training_cfg.model.sc_rate > 0:
        consistency_criterion = SelfConditioningConsistencyCriterionDecorator(
            consistency_criterion,
            self_conditioning_ratio=training_cfg.model.sc_rate
        )
    criteria.append(consistency_criterion)
    weights.append(training_cfg.consistency_loss_weight)

    # Add NLL criterion
    nll_criterion = NllCriterion(model, training_cfg.model.diffusion_steps)
    criteria.append(nll_criterion)
    weights.append(training_cfg.nll_loss_weight)

    # Create shortcut sampler
    shortcut_sampler = ShortcutSampler(
        diffusion_steps=training_cfg.model.diffusion_steps,
        min_shortcut_size=training_cfg.model.min_shortcut_size
    )
    time_and_shortcut_sampler = TimeAndShorcutStampler(
        shortcut_sampler,
        training_cfg.model.diffusion_steps,
        training_cfg.model.min_shortcut_size
    )

    return CompositeCriterion(
        criteria=tuple(criteria),
        criteria_weights=tuple(weights),
        model=model,
        diffusion_steps=training_cfg.model.diffusion_steps,
        self_consistency_ratio=training_cfg.self_consistency_ratio,
        sampler=time_and_shortcut_sampler,
    )


def _create_flow_nll_criterion(
        flow_matching_criterion: X0FlowMatchingCriterion,
        model: FlowMatchingModel,
        training_cfg: TrainingConfig
) -> FlowNllCriterion:
    """Create flow NLL criterion."""
    nll_criterion = NllCriterion(model, training_cfg.model.diffusion_steps)
    sampler = UniformSampler(training_cfg.model.diffusion_steps)
    return FlowNllCriterion(
        flow_matching_criterion=flow_matching_criterion,
        nll_criterion=nll_criterion,
        model=model,
        diffusion_steps=training_cfg.model.diffusion_steps,
        sampler=sampler
    )


def load_unit_from_checkpoint(
        criterion: CompositeCriterion | FlowNllCriterion,
        gen_cfg: GenerationConfig
) -> TrainModule:
    """Load and configure training unit from checkpoint."""
    unit = TrainModule.load_from_checkpoint(
        str(gen_cfg.checkpoint_path),
        criterion=criterion,
        scheduler_config=gen_cfg.training_config.optimizer.scheduler
    )
    unit.set_prediction_shortcut_size(gen_cfg.generation_shortcut_size)
    return unit


def create_test_dataloader(gen_cfg: GenerationConfig) -> DataLoader:
    """Create test dataloader from config."""
    test_ds = Dataset.load_from_disk(gen_cfg.test_data_path)
    test_text_ds = TextDataset(test_ds)

    return DataLoader(
        test_text_ds,
        batch_size=gen_cfg.batch_size,
        collate_fn=collate,
        shuffle=False,
    )


def get_ema_callback(gen_cfg: GenerationConfig) -> Optional[EMACallback]:
    """Create and configure EMA callback if needed.
    
    Returns:
        Optional[EMACallback]: Configured EMA callback or None if not needed/available
    """
    if not (gen_cfg.training_config.ema is not None and gen_cfg.use_ema_weights):
        return None

    ema_callback = EMACallback(
        decay=gen_cfg.training_config.ema.smoothing,
        update_interval=gen_cfg.training_config.ema.update_interval,
        ema_eval=True
    )

    checkpoint = torch.load(gen_cfg.checkpoint_path, map_location="cpu", weights_only=False)
    if "EMACallback" not in checkpoint["callbacks"]:
        logger.warning("EMA weights requested but not found in checkpoint")
        return None

    ema_callback_state = checkpoint["callbacks"]["EMACallback"]
    ema_callback.load_state_dict(ema_callback_state)
    logger.info("Using EMA weights for generation")
    return ema_callback


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m shortcutfm.decoding.generate <config_path> <cli_args>")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Parse and validate generation config
    gen_cfg = parse_generation_config(yaml_path, args_list)
    logger.info("Generation Configuration:\n" + om.to_yaml(gen_cfg.model_dump()))
    logger.info("Training Configuration:\n" + om.to_yaml(gen_cfg.training_config.model_dump()))

    pl.seed_everything(gen_cfg.seed)

    callbacks = []
    if ema_callback := get_ema_callback(gen_cfg):
        callbacks.append(ema_callback)

    save_outputs_callback = SaveTestOutputsCallback(
        save_path=Path(gen_cfg.output_folder),
        diff_steps=gen_cfg.training_config.model.diffusion_steps,
        shortcut_size=gen_cfg.generation_shortcut_size,
        start_example_idx=1
    )
    callbacks.append(save_outputs_callback)

    criterion = create_criterion(gen_cfg.training_config)
    unit = load_unit_from_checkpoint(criterion, gen_cfg)
    test_dataloader = create_test_dataloader(gen_cfg)

    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_test_batches=gen_cfg.limit_test_batches,
    )
    trainer.test(unit, dataloaders=test_dataloader)
