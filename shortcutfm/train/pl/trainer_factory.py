import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import (
    CompositeCriterion,
    FlowNllCriterion,
    NllCriterion,
    SelfConditioningConsistencyCriterionDecorator,
    SelfConditioningFlowMatchingCriterionDecorator,
    X0ConsistencyCrterion,
    X0FlowMatchingCriterion,
)
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.model.model import FlowMatchingModel
from shortcutfm.shortcut_samplers import ShortcutSampler, TimeAndShortcutSampler, UniformSampler
from shortcutfm.train.pl.callbacks import EMACallback
from shortcutfm.train.pl.train_unit import TrainModule

logger = logging.getLogger(__name__)


def create_criterion(training_cfg: TrainingConfig, tokenizer=None) -> CompositeCriterion | FlowNllCriterion:
    """Create model, tokenizer and criterion based on training config.

    :param training_cfg: Training configuration containing model and training parameters
    :type training_cfg: TrainingConfig
    :return: Composite criterion with flow matching and optional consistency components
    :rtype: CompositeCriterion | FlowNllCriterion
    """
    # Initialize model
    model = TransformerNetModelFactory(training_cfg.model).build()

    # Initialize tokenizer
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(training_cfg.model.config_name)

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
    """Create composite criterion with consistency and NLL components.

    :param flow_matching_criterion: Base flow matching criterion
    :type flow_matching_criterion: X0FlowMatchingCriterion
    :param model: Flow matching model instance
    :type model: FlowMatchingModel
    :param training_cfg: Training configuration containing loss weights and model parameters
    :type training_cfg: TrainingConfig
    :return: Composite criterion combining flow matching, consistency and NLL losses
    :rtype: CompositeCriterion
    """
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
    time_and_shortcut_sampler = TimeAndShortcutSampler(
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
    """Create flow NLL criterion.

    :param flow_matching_criterion: Base flow matching criterion
    :type flow_matching_criterion: X0FlowMatchingCriterion
    :param model: Flow matching model instance
    :type model: FlowMatchingModel
    :param training_cfg: Training configuration containing model parameters
    :type training_cfg: TrainingConfig
    :return: Flow NLL criterion combining flow matching and NLL losses
    :rtype: FlowNllCriterion
    """
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
        checkpoint_path: Path | str,
        training_config: TrainingConfig
) -> TrainModule:
    """Load and configure training unit from checkpoint.

    :param criterion: Criterion instance to use for training
    :type criterion: CompositeCriterion | FlowNllCriterion
    :param checkpoint_path: Path to the checkpoint file
    :type checkpoint_path: Path | str
    :param training_config: Training configuration containing optimizer settings
    :type training_config: TrainingConfig
    :return: Configured training unit loaded from checkpoint
    :rtype: TrainModule
    """
    unit = TrainModule.load_from_checkpoint(
        str(checkpoint_path),
        criterion=criterion,
        scheduler_config=training_config.optimizer.scheduler
    )
    return unit


def get_ema_callback(
        training_config: TrainingConfig,
        checkpoint_path: Optional[str | Path] = None,
        strict: bool = True
) -> Optional[EMACallback]:
    """Create and configure EMA callback if needed.

    :param training_config: Training configuration containing EMA settings
    :type training_config: TrainingConfig
    :param checkpoint_path: Optional path to checkpoint for loading EMA state
    :type checkpoint_path: Optional[str | Path]
    :param strict: If True, raises an error when EMA state cannot be loaded from checkpoint
    :type strict: bool
    :return: Configured EMA callback or None if not needed/available
    :rtype: Optional[EMACallback]
    :raises: ValueError when strict=True and EMA state cannot be loaded from checkpoint
    """
    if not training_config.ema:
        logger.info("No EMA config. EMA disabled.")
        return None

    ema_callback = EMACallback(
        decay=training_config.ema.smoothing,
        update_interval=training_config.ema.update_interval,
        ema_eval=True
    )

    if checkpoint_path is None:
        logger.info("Checkpoint path not provided. Initializing fresh EMA callback.")
        return ema_callback

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "callbacks" not in checkpoint or "EMACallback" not in checkpoint["callbacks"]:
            error_msg = "EMA weights requested but not found in checkpoint"
            if strict:
                raise ValueError(error_msg) from e
            logger.warning(error_msg)
            return ema_callback

        ema_callback_state = checkpoint["callbacks"]["EMACallback"]
        ema_callback.load_state_dict(ema_callback_state)
        logger.info("Successfully loaded EMA state from checkpoint")
        return ema_callback
    except Exception as e:
        error_msg = f"Error loading EMA state from checkpoint: {e}"
        if strict:
            raise ValueError(error_msg) from e
        logger.error(error_msg)
        return ema_callback
