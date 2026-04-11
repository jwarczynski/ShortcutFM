
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

    with open(config_path) as f:
        yaml_cfg = om.load(f)

    # Merge: defaults -> YAML -> CLI args (CLI takes highest priority)
    merged_cfg = om.merge(yaml_cfg, om.from_cli(args_list))

    # Convert to dict and validate with Pydantic
    config_dict = om.to_container(merged_cfg, resolve=True)
    training_config = TrainingConfig(**config_dict) # type: ignore

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
        quasar_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/Quasar-T/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/Quasar-T/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        wiki_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/Wiki-alignment/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/Wiki-alignment/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        common_sense_conversation_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/CommonsenseConversation/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/CommonsenseConversation/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        parasci_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/parasci/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/parasci/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
            "model.max_position_embeddings": 256,
        }

        paws_wiki_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/paws_wiki/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/paws_wiki/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        qqp_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/QQP-Official/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/QQP-Official/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        grammar_correction_config = {
            "training_data_path": "datasets/tokenized/bert-base-uncased/grammar_correction/train",
            "validation_data_path": "datasets/tokenized/bert-base-uncased/grammar_correction/valid",
            "check_val_every_n_epoch": None,
            "val_interval": 5000,
        }

        baseline_config = {
            "model.input_dims": 128,
            "model.output_dims": 128,
            "consistency_loss_weight": 0.0,
            "model.hidden_shortcut_dim": None,
            "model.sc_rate": 0.0,
        }

        scut_config = {
            "model.input_dims": 768,
            "model.output_dims": 768,
            "consistency_loss_weight": 1.0,
            "model.hidden_shortcut_dim": 128,
            "model.sc_rate": 0.0,
        }

        dataset_name_to_abbreviation_mapping = {
            "Quasar-T": "quasar",
            "Wiki-alignment": "wiki",
            "CommonsenseConversation": "commonsense",
            "parasci": "parasci",
            "paws_wiki": "pawswiki",
            "grammar_correction": "grammar",
            "QQP-Official": "qqp",
        }

        with cfg.infra.job_array() as array:
            # array.append(cfg.infra.clone_obj({}))
            for dataset_config in [qqp_config, parasci_config, paws_wiki_config]: #, quasar_config, wiki_config, common_sense_conversation_config]:
            # for dataset_config in [grammar_correction_config]:
            # for dataset_config in [wiki_config, common_sense_conversation_config]:
                training_data_path = dataset_config['training_data_path']
                ds_name = training_data_path.split('/')[-2]
                abbrev_ds_name = dataset_name_to_abbreviation_mapping[ds_name]

            #     # baseline
                for base_cfg in [baseline_config]:
                    combined_cfg = {**base_cfg, **dataset_config}

                    run_name = f"baseline_{abbrev_ds_name}_128"
                    save_folder = f"checkpoints/{abbrev_ds_name}/baseline_128"

                    array.append(cfg.infra.clone_obj({
                        "wandb.run_name": run_name,
                        "checkpoint.save_folder": save_folder,
                        **combined_cfg
                    }))

                # scut
                # for scut_cfg in [scut_config]:
                #     combined_cfg = {**scut_cfg, **dataset_config}

                #     run_name = f"scut_{abbrev_ds_name}_768"
                #     save_folder = f"checkpoints/{abbrev_ds_name}/scut_768"

                #     array.append(cfg.infra.clone_obj({
                #         "wandb.run_name": run_name,
                #         "checkpoint.save_folder": save_folder,
                #         **combined_cfg
                #     }))

        #    array.append(cfg.infra.clone_obj({
        #        "wandb.run_name": "quasar_baseline_128",
        #        "checkpoint.save_folder": "checkpoints/quasar/baseline_128",
        #        "model.input_dims": 128,
        #        "model.output_dims": 128,
        #        "training_data_path": "datasets/tokenized/bert-base-uncased/Quasar-T/train",
        #        "validation_data_path": "datasets/tokenized/bert-base-uncased/Quasar-T/valid"
        #    }))

            # qqp
            # array.append(
            #     cfg.infra.clone_obj({
            #         "wandb.run_name": "qqp_scut_bert-pt-l-tied",
            #         "checkpoint.save_folder": "checkpoints/qqp/scut_bert-pt-l-tied/",
            #     })
            # )
            # # webnlg
            # array.append(
            #     cfg.infra.clone_obj(
            #         {
            #             "training_data_path": "datasets/tokenized/bert-base-uncased/webnlg/train",
            #             "validation_data_path": "datasets/tokenized/bert-base-uncased/webnlg/valid",
            #             "checkpoint.save_folder": "checkpoints/webnlg/scut_emb-pt-frze-tied/",
            #             "wandb.run_name": "nlg_scut_emb-pt-frze-tied",
            #             "max_steps": 60000
            #         }
            #     )
            # )
            # wmt19
            # array.append(
            #     cfg.infra.clone_obj(
            #         {
            #             "training_data_path": "datasets/tokenized/opus-mt-en-de/wmt/train",
            #             "validation_data_path": "datasets/tokenized/opus-mt-en-de/wmt/valid",
            #             "checkpoint.save_folder": "checkpoints/wmt19/scut_dim128_w=.1/",
            #             "check_val_every_n_epoch": None,
            #             "val_interval": 5000,
            #             "model.config_name": "bert-base-uncased",
            #             "model.tokenizer_config_name": "Helsinki-NLP/opus-mt-en-de",
            #             "model.vocab_size": 58101,
            #             "model.null_token_id": 1,
            #             "wandb.run_name": "wmt_scut_dim128_w=.1",
            #             "max_steps": 60000
            #         }
            #     )
            # )
