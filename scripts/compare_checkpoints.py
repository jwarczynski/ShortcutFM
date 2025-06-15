import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import Dataset

sys.path.append("../shortcutfm")

from shortcutfm.analysis.denoising import denoise_with_velocity_tracking
from shortcutfm.batch import collate
from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import CompositeCriterion
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.train_unit import TrainModule
from shortcutfm.train.pl.trainer_factory import create_criterion


def extract_config_params(config: TrainingConfig, checkponit_dir: Path) -> dict[str, Any]:
    """Extract key parameters from training config for legend."""
    return {
        "sc_p": config.shortcut_target_x_start_probability,
        "type": config.model.type,
        "sc_r": config.self_consistency_ratio,
        "cl_w": config.consistency_loss_weight,
        "d_sc": config.model.default_shortcut,
        "run_id": checkponit_dir.name.replace("run_", ""),
        "run_name": config.wandb.run_name,
    }


def create_abbreviated_legend(params: dict[str, Any]) -> str:
    """Create abbreviated legend from parameters."""
    return params.get("run_id", params.get("run_name", "unknown"))
    return (
        f"sc_p={params['sc_p']:.1f}, "
        f"type={params['type']}, "
        f"sc_r={params['sc_r']:.1f}, "
        f"cl_w={params['cl_w']:.1f}, "
        f"d_sc={params['d_sc']}"
    )


def load_unit_from_checkpoint(
    criterion: CompositeCriterion,
    checkpoint_path: Path | str,
    training_config: TrainingConfig,
    denoising_step_size: int = None,
    prediction_shortcut_size: int = None,
) -> TrainModule:
    """Load and configure training unit from checkpoint with key remapping.

    :param criterion: Criterion instance to use for training
    :type criterion: CompositeCriterion | FlowNllCriterion
    :param checkpoint_path: Path to the checkpoint file
    :type checkpoint_path: Path | str
    :param training_config: Training configuration containing optimizer settings
    :type training_config: TrainingConfig
    :param denoising_step_size: Number of denoising steps (optional)
    :type denoising_step_size: int | None
    :param prediction_shortcut_size: Size of prediction shortcut (optional)
    :type prediction_shortcut_size: int | None
    :return: Configured training unit loaded from checkpoint
    :rtype: TrainModule
    """
    denoising_step_size = denoising_step_size or training_config.denoising_step_size
    prediction_shortcut_size = prediction_shortcut_size or training_config.prediction_shortcut_size

    # Load the checkpoint
    checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False)

    # Get the state dictionary from the checkpoint
    state_dict = checkpoint.get("state_dict", checkpoint)

    new_state_dict = {}
    for key, value in state_dict.items():
        # Remap keys: replace 'nll' with 'consistency_criterion' or 'embedding_criterion'
        if "criterion.nll" in key or "criterion.flow_matching_criterion" in key:
            # Determine which criterion to map to based on context or configuration
            # For simplicity, we assume mapping to 'consistency_criterion' for some keys
            # and 'embedding_criterion' for others. Adjust this logic as needed.
            new_key = key.replace("criterion.nll", "criterion.embedding_criterion")
            new_key = key.replace("criterion.flow_matching_criterion", "criterion.embedding_criterion")
            # Alternatively, for embedding_criterion, you might need a condition
            # new_key = key.replace('criterion.nll', 'criterion.embedding_criterion')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # Initialize the TrainModule
    unit = TrainModule(
        criterion=criterion,
        optimizer_config=training_config.optimizer.scheduler,
        prediction_shortcut_size=prediction_shortcut_size,
        denoising_step_size=denoising_step_size,
    )

    # Load the remapped state dictionary into the model
    try:
        unit.load_state_dict(new_state_dict, strict=False)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        raise

    return unit


def analyze_checkpoint(
    checkpoint_dir: Path,
    test_dataset: TextDataset,
    batch_size: int = 32,
    shortcut_size: int = 1024,
) -> dict[str, Any]:
    """Analyze a single checkpoint."""
    # Load checkpoint and config
    checkpoint_path = checkpoint_dir / "last.ckpt"
    if not checkpoint_path.exists():
        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        checkpoint_path = ckpt_files[0]
    config_path = checkpoint_dir / "training_config.yaml"

    with open(config_path) as f:
        yaml_cfg = OmegaConf.load(f)
    config = TrainingConfig(**OmegaConf.to_container(yaml_cfg, resolve=True))  # type: ignore

    # Load model
    criterion = create_criterion(config)  # noqa: F821
    unit = load_unit_from_checkpoint(criterion, checkpoint_path, config)
    unit.eval()

    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # Get a batch
    batch = next(iter(dataloader))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch.seqs = batch.seqs.to(device)
    batch.input_ids_mask = batch.input_ids_mask.to(device)
    batch.padding_mask = batch.padding_mask.to(device)

    # Run denoising analysis
    results = denoise_with_velocity_tracking(
        unit.criterion.flow_matching_criterion,
        batch,
        shortcut_size=shortcut_size,
        per_token_cosine=True,
    )

    # Extract config parameters
    params = extract_config_params(config, checkpoint_dir)

    return {
        "results": results,
        "params": params,
        "legend": create_abbreviated_legend(params),
    }


def plot_comparison(
    analyses: list[dict[str, Any]],
    output_dir: Path,
    figsize: tuple[int, int] = (14, 8),
) -> None:
    """Create comparison plots for all checkpoints."""
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot cosine similarities
    plt.figure(figsize=figsize)
    for analysis in analyses:
        results = analysis["results"]
        timesteps = results["timesteps"]
        cosine_similarities = [cs.mean().item() for cs in results["cosine_similarities"]]
        plt.plot(timesteps, cosine_similarities, marker=".", linestyle="-", label=analysis["legend"])

    plt.title("Cosine Similarity Comparison Across Models")
    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.ylim(-1.1, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "cosine_similarity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot velocity norms
    plt.figure(figsize=figsize)
    for analysis in analyses:
        results = analysis["results"]
        timesteps = results["timesteps"]
        pred_norms = [n.mean().item() for n in results["predicted_velocity_norms"]]
        gt_norms = [n.mean().item() for n in results["ground_truth_velocity_norms"]]
        plt.plot(timesteps, pred_norms, marker=".", linestyle="-", label=f"{analysis['legend']} (pred)")
        plt.plot(timesteps, gt_norms, marker=".", linestyle="--", label=f"{analysis['legend']} (gt)")

    plt.title("Velocity Norm Comparison Across Models")
    plt.xlabel("Timestep")
    plt.ylabel("Velocity Norm")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_norm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Configuration

    run_dirs = os.listdir("checkpoints")
    excluded_dirs = ["run_jwpwmptz", "run_wkkbo3i3"]
    run_dirs = [dir for dir in run_dirs if "run_" in dir and dir not in excluded_dirs]
    checkpoint_dirs = [Path(f"checkpoints/{dir}") for dir in run_dirs]
    test_data_path = "datasets/tokenized/bert-base-uncased/QQP-Official/test"
    output_dir = Path("results/analysis_results")
    batch_size = 32
    shortcut_size = 128

    # Load test dataset
    test_ds = Dataset.load_from_disk(test_data_path)
    test_dataset = TextDataset(test_ds)

    # Analyze each checkpoint
    analyses = []
    for checkpoint_dir in checkpoint_dirs:
        print(f"Analyzing checkpoint: {checkpoint_dir}")
        analysis = analyze_checkpoint(
            checkpoint_dir,
            test_dataset,
            batch_size,
            shortcut_size,
        )
        analyses.append(analysis)

    # Create comparison plots
    plot_comparison(analyses, output_dir)


if __name__ == "__main__":
    main()
