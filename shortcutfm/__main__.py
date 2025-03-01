import logging

from lightning import seed_everything
from omegaconf import OmegaConf as om

from shortcutfm.train.mosaic.mosaic_trainer import get_composer_trainer
from shortcutfm.train.pl.trainer import get_lightning_trainer
from shortcutfm.utils import parse_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    cfg = parse_args(defult_config_path="configs/default.yaml")
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
