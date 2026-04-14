"""Submit a single training job via exca.

Usage:
    uv run python scripts/submit_training.py configs/training/qqp_scut_768_fix.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf as om
from shortcutfm.config import TrainingConfig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml> [cli_overrides...]")
        sys.exit(1)

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    with open(yaml_path) as f:
        yaml_cfg = om.load(f)

    if args_list:
        yaml_cfg = om.merge(yaml_cfg, om.from_cli(args_list))

    config_dict = om.to_container(yaml_cfg, resolve=True)
    cfg = TrainingConfig(**config_dict)

    print(f"Submitting: {cfg.wandb.run_name}")
    print(f"  Checkpoint: {cfg.checkpoint.save_folder}")
    print(f"  GPUs: {cfg.infra.gpus_per_node}")
    print(f"  Node: {(cfg.infra.slurm_additional_parameters or {}).get('nodelist', 'any')}")
    print(f"  Pretrained weights: {cfg.model.use_pretrained_weights}")
    print(f"  Pretrained embeddings: {cfg.model.use_pretrained_embeddings}")

    cfg.train()
