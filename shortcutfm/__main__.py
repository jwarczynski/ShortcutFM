import logging
import sys
from pathlib import Path

from lightning import seed_everything
from omegaconf import OmegaConf as om

from shortcutfm.config import TrainingConfig
from shortcutfm.train.pl.trainer import get_lightning_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("exca").setLevel(logging.DEBUG)


def parse_config(config_path: str, args_list: list[str]) -> TrainingConfig:
    """Parse and validate training config from YAML file"""
    if not Path(config_path).exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(yaml_cfg, om.from_cli(args_list))

    # Convert to dict and validate with Pydantic
    config_dict = om.to_container(merged_cfg, resolve=True)
    training_config = TrainingConfig(**config_dict)

    return training_config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m shortcutfm <config_path>")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    cfg = parse_config(yaml_path, args_list)
    logger.info("Final Configuration:\n" + om.to_yaml(cfg.model_dump()))

    if cfg.use_exca:
        if cfg.dry_run:
            logger.info("Final config: \n" + om.to_yaml(cfg.model_dump()))
        else:
            cfg.train()
    else:
        seed_everything(cfg.seed)
        trainer, model, train_dataloader, val_dataloader = get_lightning_trainer(cfg)
        if not cfg.dry_run:
            logger.info("Starting training...")
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.checkpoint.path)
