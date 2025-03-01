import logging
import sys
from pathlib import Path

import lightning as pl
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from shortcutfm.batch import collate
from shortcutfm.config import GenerationConfig, ModelConfig
from shortcutfm.criteria import X0FlowMatchingCriterion, CompositeCriterion, \
    SelfConditioningFlowMatchingCriterionDecorator
from shortcutfm.model.factory import TransformerNetModelFactory
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import EMACallback, SaveTestOutputsCallback
from shortcutfm.train.pl.train_unit import TrainModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_generation_config(config_path: str, args_list: list[str]) -> GenerationConfig:
    """Parse and validate generation config from YAML file"""
    if not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")

    # Load and merge configs
    with open("configs/generation/default.yaml", "r") as f:
        default_cfg = om.load(f)

    with open(config_path, "r") as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(default_cfg, yaml_cfg, om.from_cli(args_list))

    # Convert to GenerationConfig and validate
    cfg = GenerationConfig(**OmegaConf.to_container(merged_cfg, resolve=True))
    return cfg


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m shortcutfm <config_path> <cli_args>")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Parse and validate config
    cfg = parse_generation_config(yaml_path, args_list)
    logger.info("Final Configuration:\n" + om.to_yaml(cfg.model_dump()))

    # Initialize model
    model = TransformerNetModelFactory(ModelConfig(**cfg.model.model_dump())).build()

    # Define Criterions
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config_name)
    flow_matching_criterion = X0FlowMatchingCriterion(
        model,
        diffusion_steps=cfg.model.diffusion_steps,
        tokenizer=tokenizer
    )
    self_conditioning_flow_matching_criterion = SelfConditioningFlowMatchingCriterionDecorator(
        flow_matching_criterion,
        self_conditioning_ratio=None
    )

    criterion = CompositeCriterion(
        criteria=(self_conditioning_flow_matching_criterion,),
        criteria_weights=(1,),
        model=model,
        diffusion_steps=cfg.model.diffusion_steps,
        self_consistency_ratio=None,
        sampler=None,
    )

    # Load checkpoint
    unit = TrainModule.load_from_checkpoint(str(cfg.checkpoint_path), criterion=criterion)
    unit.set_prediction_shorcut_size(cfg.generation_shortcut_size)

    # Load dataset
    test_ds = Dataset.load_from_disk(cfg.test_data_path)
    test_text_ds = TextDataset(test_ds)

    test_dataloader = DataLoader(
        test_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # Setup callbacks
    callbacks = []

    # Add EMA callback if configured
    if cfg.ema is not None:
        ema_callback = EMACallback(
            decay=cfg.ema.smoothing,
            update_interval=cfg.ema.update_interval,
            ema_eval=True
        )

        checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
        if "EMACallback" in checkpoint["callbacks"]:
            ema_callback_state = checkpoint["callbacks"]["EMACallback"]
            ema_callback.load_state_dict(ema_callback_state)
            callbacks.append(ema_callback)
        else:
            raise ValueError("EMACallback not found in checkpoint")

    # Add output saving callback
    save_outputs_callback = SaveTestOutputsCallback(
        save_path=cfg.output_folder,
        diff_steps=cfg.model.diffusion_steps,
        shortcut_size=cfg.generation_shortcut_size,
        start_example_idx=1
    )
    callbacks.append(save_outputs_callback)

    # Initialize trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_test_batches=cfg.limit_test_batches,
    )

    trainer.test(unit, dataloaders=test_dataloader)
