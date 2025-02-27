import logging
import sys

from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf as om

from shortcutfm.train.mosaic.mosaic_trainer import get_composer_trainer
from shortcutfm.train.pl.trainer import get_lightning_trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Load default configuration (hardcoded defaults)
    with open("configs/default.yaml", "r") as f:
        default_cfg = om.load(f)

    with open(yaml_path, "r") as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    cfg = om.merge(default_cfg, yaml_cfg, om.from_cli(args_list))
    cfg = DictConfig(cfg)
    logger.info("Final Merged Configuration:\n" + om.to_yaml(cfg, resolve=True))

    if cfg.use_composer:
        trainer = get_composer_trainer(cfg)
        if not getattr(cfg, "dry_run", False):
            logger.info("Starting training...")
            trainer.fit()

    else:
        seed_everything(cfg.training_config.seed)
        trainer, model, train_dataloader, val_dataloader = get_lightning_trainer(cfg)
        if not getattr(cfg, "dry_run", False):
            logger.info("Starting training...")
            trainer.fit(model, train_dataloader, val_dataloader)
