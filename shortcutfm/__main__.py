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

    if not cfg.use_exca:
        seed_everything(cfg.seed)
        logger.info("Final Configuration:\n" + om.to_yaml(cfg.model_dump()))
        trainer, model, train_dataloader, val_dataloader = get_lightning_trainer(cfg)
        if not cfg.dry_run:
            logger.info("Starting training...")
            trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.checkpoint.path)

    else:
        modern_bert_cfg = {
            "model.init_pretrained": "modern_bert",
            "model.config_name": "answerdotai/ModernBERT-base",
            "model.vocab_size": 50368,
            "model.word_embedding_std": 0.5,

            "training_data_path": "datasets/tokenized/ModernBERT-base/QQP-Official/train",
            "validation_data_path": "datasets/tokenized/ModernBERT-base/QQP-Official/valid",

            "checkpoint.save_folder": "checkpoints/qqp/ModernBERT",
        }

        bert_base_cfg = {
            "model.word_embedding_std": 1.0,

            "training_data_path": "datasets/tokenized/bert-base-uncased/QQP-Official/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/QQP-Official/valid",

            "checkpoint.save_folder": "checkpoints/qqp/bert-base/stacked",
        }

        with cfg.infra.job_array() as array:
            # for name, bert_cfg in zip(("modern", "base"), [modern_bert_cfg, bert_base_cfg]):
            for name, bert_cfg in zip(("base",), [bert_base_cfg]):
                # for gc in [0.5, 0.1]:
                for gc in [None]:
                    # for activation in ["gelu", "tanh"]:
                    for activation in ["gelu"]:
                        for lr in [3e-3]:
                            for freeze_emb in [True]:
                                for num_overfit_batches in [2]:
                            # for isotropy_loss_weight in [100, 1000]:
                            # for isotropy_loss_weight in [100, 1000]:
                            #     for nll_loss_weight in [10]:
                                    array.append(
                                        cfg.infra.clone_obj(
                                            {
                                                "overfit_batches": num_overfit_batches,
                                                # "nll_loss_weight": nll_loss_weight,
                                                # "isotropy_loss_weight": isotropy_loss_weight,
                                                "optimizer.scheduler.lr": lr,
                                                "model.projection_activation": activation,
                                                "model.freeze_word_embedding": freeze_emb,
                                                "gradient_clipping": gc,
                                                "prediction_shortcut_size": 32,
                                                # "wandb.run_name": f"lr{lr}_mean_{activation}_zero",
                                                # "wandb.run_name": f"lr{lr}_mean_{activation}_ilw{isotropy_loss_weight}_nll{nll_loss_weight}",
                                                **bert_cfg,
                                            }
                                        )
                                    )
                                # array.append(
                                #         cfg.infra.clone_obj(
                                #             {
                                #                 "overfit_batches": num_overfit_batches,
                                #                 # "nll_loss_weight": nll_loss_weight,
                                #                 # "isotropy_loss_weight": isotropy_loss_weight,
                                #                 "optimizer.scheduler.lr": lr,
                                #                 "model.sc_rate": 0.5,
                                #                 "model.projection_activation": activation,
                                #                 "model.freeze_word_embedding": freeze_emb,
                                #                 "gradient_clipping": gc,
                                #                 "prediction_shortcut_size": 32,
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_zero",
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_ilw{isotropy_loss_weight}_nll{nll_loss_weight}",
                                #                 **bert_cfg,
                                #             }
                                #         )
                                #     )
                                #
                                # array.append(
                                #         cfg.infra.clone_obj(
                                #             {
                                #                 "overfit_batches": num_overfit_batches,
                                #                 # "nll_loss_weight": nll_loss_weight,
                                #                 # "isotropy_loss_weight": isotropy_loss_weight,
                                #                 "optimizer.scheduler.lr": lr,
                                #                 "model.sc_rate": 0.0,
                                #                 "self_consistency_ratio": 0.25,
                                #                 "model.hidden_shortcut_dim": 128,
                                #                 "model.projection_activation": activation,
                                #                 "model.freeze_word_embedding": freeze_emb,
                                #                 "gradient_clipping": gc,
                                #                 "prediction_shortcut_size": 32,
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_zero",
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_ilw{isotropy_loss_weight}_nll{nll_loss_weight}",
                                #                 **bert_cfg,
                                #             }
                                #         )
                                #     )
                                #
                                # array.append(
                                #         cfg.infra.clone_obj(
                                #             {
                                #                 "overfit_batches": num_overfit_batches,
                                #                 # "nll_loss_weight": nll_loss_weight,
                                #                 # "isotropy_loss_weight": isotropy_loss_weight,
                                #                 "optimizer.scheduler.lr": lr,
                                #                 "model.sc_rate": 0.5,
                                #                 "self_consistency_ratio": 0.25,
                                #                 "model.hidden_shortcut_dim": 128,
                                #                 "model.projection_activation": activation,
                                #                 "model.freeze_word_embedding": freeze_emb,
                                #                 "gradient_clipping": gc,
                                #                 "prediction_shortcut_size": 32,
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_zero",
                                #                 # "wandb.run_name": f"lr{lr}_mean_{activation}_ilw{isotropy_loss_weight}_nll{nll_loss_weight}",
                                #                 **bert_cfg,
                                #             }
                                #         )
                                #     )
