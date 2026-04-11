"""
Generate texts for one or more checkpoints across multiple NFE values.
Submits one exca job per (checkpoint, NFE) combination.

Usage:
    # Single config:
    uv run python scripts/generate_nfe_sweep.py configs/generation/qqp_scut_768_nfe.yaml

    # Multiple configs:
    uv run python scripts/generate_nfe_sweep.py config1.yaml config2.yaml config3.yaml

    # With CLI overrides (applied to all configs):
    uv run python scripts/generate_nfe_sweep.py config1.yaml config2.yaml -- force_regeneration=true
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from shortcutfm.config import GenerationConfig

DIFFUSION_STEPS = 2048

# NFE values to sweep — step_size = diffusion_steps / nfe
NFE_VALUES = [1, 2, 4, 8, 64, 128, 256]


def extract_checkpoint_info(checkpoint_path: str) -> tuple[str, str, str]:
    """Extract dataset, subdir, run_id from checkpoint path."""
    parts = Path(checkpoint_path).parts
    ckpt_idx = next(i for i, p in enumerate(parts) if p == "checkpoints")
    return parts[ckpt_idx + 1], parts[ckpt_idx + 2], parts[ckpt_idx + 3]


def submit_sweep(yaml_path: str, cli_args: list[str]) -> None:
    """Submit NFE sweep for a single config file."""
    with open(yaml_path) as f:
        yaml_cfg = OmegaConf.load(f)

    if cli_args:
        yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.from_cli(cli_args))

    base_dict = OmegaConf.to_container(yaml_cfg, resolve=True)

    checkpoint_path = base_dict.get("checkpoint_path", "")
    dataset, subdir, run_id = extract_checkpoint_info(checkpoint_path)

    # Build a base config with NFE=1 to get infra handle
    base_dict["generation_shortcut_size"] = DIFFUSION_STEPS
    base_dict["denoising_step_size"] = DIFFUSION_STEPS
    base_dict["output_folder"] = f"generation_outputs/{dataset}/{subdir}/{run_id}/scut={DIFFUSION_STEPS}"
    base_cfg = GenerationConfig(**base_dict)

    print(f"\n=== {dataset}/{subdir}/{run_id} ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Submitting {len(NFE_VALUES)} jobs for NFE values: {NFE_VALUES}")

    with base_cfg.infra.job_array() as array:
        for nfe in NFE_VALUES:
            step_size = DIFFUSION_STEPS // nfe
            output_folder = f"generation_outputs/{dataset}/{subdir}/{run_id}/scut={step_size}"

            cfg_copy = base_cfg.infra.clone_obj(
                {
                    "generation_shortcut_size": step_size,
                    "denoising_step_size": step_size,
                    "output_folder": output_folder,
                }
            )
            print(f"  NFE={nfe}: shortcut_size={step_size}, output={output_folder}")
            array.append(cfg_copy)

    print("Jobs submitted.")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config1.yaml> [config2.yaml ...] [-- cli_overrides...]")
        sys.exit(1)

    args = sys.argv[1:]

    # Split on "--" separator
    if "--" in args:
        sep_idx = args.index("--")
        yaml_paths = args[:sep_idx]
        cli_args = args[sep_idx + 1:]
    else:
        yaml_paths = [a for a in args if a.endswith(".yaml") or a.endswith(".yml")]
        cli_args = [a for a in args if not (a.endswith(".yaml") or a.endswith(".yml"))]

    for yaml_path in yaml_paths:
        submit_sweep(yaml_path, cli_args)


if __name__ == "__main__":
    main()
