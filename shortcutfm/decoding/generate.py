import logging
import sys
from pathlib import Path

import lightning as pl
from datasets import Dataset
from omegaconf import OmegaConf, OmegaConf as om
from torch.utils.data import DataLoader

from shortcutfm.batch import collate
from shortcutfm.config import GenerationConfig
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.callbacks import SaveTestOutputsCallback
from shortcutfm.train.pl.trainer_factory import create_criterion, get_ema_callback, load_unit_from_checkpoint

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
    if gen_cfg.use_ema_weights:
        callbacks.append(get_ema_callback(gen_cfg.training_config, gen_cfg.checkpoint_path))

    save_outputs_callback = SaveTestOutputsCallback(
        save_path=Path(gen_cfg.output_folder),
        diff_steps=gen_cfg.training_config.model.diffusion_steps,
        shortcut_size=gen_cfg.generation_shortcut_size,
        start_example_idx=1
    )
    callbacks.append(save_outputs_callback)

    criterion = create_criterion(gen_cfg.training_config)
    unit = load_unit_from_checkpoint(criterion, gen_cfg.checkpoint_path, gen_cfg.training_config)
    unit.set_prediction_shortcut_size(gen_cfg.generation_shortcut_size)
    test_dataloader = create_test_dataloader(gen_cfg)

    trainer = pl.Trainer(
        callbacks=callbacks,
        limit_test_batches=gen_cfg.limit_test_batches,
    )
    trainer.test(unit, dataloaders=test_dataloader)
