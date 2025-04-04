import logging
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoTokenizer

from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import (
    CompositeCriterion,
    FlowNllCriterion,
    IsotropyCriterion, NllCriterion,
    SelfConditioningConsistencyCriterionDecorator,
    SelfConditioningFlowMatchingCriterionDecorator,
    VelocityConsistencyCriterion, VelocityFlowMatchingCriterion, X0ConsistencyCriterion,
    X0FlowMatchingCriterion,
)
from shortcutfm.model.factory import FFNFactory, StackedEmbeddingTransformerNetModelFactory, TransformerNetModelFactory
from shortcutfm.model.model import FlowMatchingModel
from shortcutfm.shortcut_samplers import (
    LossSecondMomentResampler,
    ShortcutSampler,
    TimeAndShortcutSampler,
    UniformSampler,
)
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
    factory = create_factory(training_cfg)
    model = factory.build()

    # Initialize tokenizer
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(training_cfg.model.config_name)

    # Create base flow matching criterion
    flow_matching_criterion = create_flow_matching_criterion(model, tokenizer, training_cfg)

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

def create_factory(training_cfg: TrainingConfig):
    match training_cfg.architecture:
        case "transformer": return TransformerNetModelFactory(training_cfg.model)
        case "stacked": return StackedEmbeddingTransformerNetModelFactory(training_cfg.model)
        case "ffn": return FFNFactory(training_cfg.model)
        case _: raise ValueError(f"Unknown architecture: {training_cfg.architecture}")

def create_flow_matching_criterion(model, tokenizer, training_cfg: TrainingConfig):
    reduce_fn = get_reduction_fn(training_cfg.reduce_fn)

    if training_cfg.model.parametrization == "x0":
        flow_matching_criterion = X0FlowMatchingCriterion(
            model,
            diffusion_steps=training_cfg.model.diffusion_steps,
            tokenizer=tokenizer,
            reduce_fn=reduce_fn,
            training_cfg=training_cfg,
        )
    elif training_cfg.model.parametrization == "velocity":
        flow_matching_criterion = VelocityFlowMatchingCriterion(
            model,
            diffusion_steps=training_cfg.model.diffusion_steps,
            tokenizer=tokenizer,
            reduce_fn=reduce_fn,
            training_cfg=training_cfg,
        )
    else:
        raise ValueError(f"Unknown parametrization: {training_cfg.model.parametrization}")

    return flow_matching_criterion


def get_reduction_fn(reduce_fn: str) -> Callable[[torch.Tensor, int], torch.Tensor]:
    reduce_fn_map = {
        "mean": torch.mean,
        "sum": torch.sum,
    }
    reduce_fn = reduce_fn_map.get(reduce_fn, None)
    if reduce_fn is None:
        raise ValueError(f"Unknown reduce_fn: {reduce_fn}")

    return reduce_fn

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
    consistency_criterion = create_consistency_criterion(model, training_cfg)
    if training_cfg.model.sc_rate > 0:
        consistency_criterion = SelfConditioningConsistencyCriterionDecorator(
            consistency_criterion,
            self_conditioning_ratio=training_cfg.model.sc_rate
        )
    criteria.append(consistency_criterion)
    weights.append(training_cfg.consistency_loss_weight)

    # Add NLL criterion
    nll_criterion = NllCriterion(model, training_cfg.model.diffusion_steps, training_cfg)
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
    )
    loss_aware_sampler = LossSecondMomentResampler(diffusion_steps=training_cfg.model.diffusion_steps)

    # isotropy_loss_wieght = training_cfg.isotropy_loss_weight if training_cfg.isotropy_loss_weight is not None else 0
    # criteria.append(IsotropyCriterion(model, training_cfg.model.diffusion_steps))
    # weights.append(isotropy_loss_wieght)

    return CompositeCriterion(
        criteria=tuple(criteria),
        criteria_weights=tuple(weights),
        model=model,
        diffusion_steps=training_cfg.model.diffusion_steps,
        self_consistency_ratio=training_cfg.self_consistency_ratio,
        sampler=loss_aware_sampler,
        time_shortcut_sampler=time_and_shortcut_sampler,
        training_cfg=training_cfg,
    )


def create_consistency_criterion(model, training_cfg):
    reduce_fn = get_reduction_fn(training_cfg.reduce_fn)

    if training_cfg.model.parametrization == "x0":
        consistency_criterion = X0ConsistencyCriterion(model, training_cfg.model.diffusion_steps, reduce_fn, training_cfg)
    elif training_cfg.model.parametrization == "velocity":
        consistency_criterion = VelocityConsistencyCriterion(model, training_cfg.model.diffusion_steps, reduce_fn, training_cfg)
    else:
        raise ValueError(f"Unknown parametrization: {training_cfg.model.parametrization}")

    return consistency_criterion


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
    nll_criterion = NllCriterion(model, training_cfg.model.diffusion_steps, training_cfg)
    sampler = UniformSampler(training_cfg.model.diffusion_steps)
    return FlowNllCriterion(
        flow_matching_criterion=flow_matching_criterion,
        nll_criterion=nll_criterion,
        model=model,
        diffusion_steps=training_cfg.model.diffusion_steps,
        sampler=sampler,
        training_cfg=training_cfg,
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
