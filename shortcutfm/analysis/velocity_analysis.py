"""Velocity analysis module for ShortcutFM.

This module provides functions for analyzing velocity predictions during the denoising process,
including calculating and visualizing cosine similarities between predicted and ground truth velocities.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def denoise_with_tracking(model, batch, shortcut_size, per_token_cosine=True, device="cuda") -> dict[str, Any]:
    """
    Perform denoising while tracking model predictions and ground truth velocities.

    Args:
        model: The model to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        per_token_cosine: If True, calculate cosine similarity for each token separately and take the mean.
                          If False, flatten all tokens into a single vector before calculating similarity.
        device: The device to use for computation

    Returns:
        A dictionary containing:
        - predicted_velocities: Model's predicted velocities at each step
        - ground_truth_velocities: Ground truth velocities at each step
        - timesteps: Timesteps used during denoising
        - cosine_similarities: Cosine similarities between predicted and ground truth velocities
        - predicted_velocity_norms: L2 norms of predicted velocities at each step
        - ground_truth_velocity_norms: L2 norms of ground truth velocities at each step
        - model_outputs: Model's predicted clean embeddings at each step
        - ground_truth_embeddings: Ground truth embeddings
        - l2_distances: L2 distances between predicted and ground truth embeddings at each step
        - velocity_l2_distances: L2 distances between predicted and ground truth velocities at each step
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    diffusion_steps = model.criterion.model.diffusion_steps

    # Move individual tensors in batch to device instead of the whole batch
    seqs = batch.seqs.to(device)
    input_ids_mask = batch.input_ids_mask.to(device)
    padding_mask = batch.padding_mask.to(device)

    # Initialize tracking variables
    predicted_velocities = []
    ground_truth_velocities = []
    timesteps_list = []
    cosine_similarities = []
    predicted_velocity_norms = []
    ground_truth_velocity_norms = []
    model_outputs = []
    l2_distances = []
    velocity_l2_distances = []

    with torch.no_grad():
        # Get input mask and embeddings
        input_mask: Tensor = input_ids_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        embeddings = model.criterion.model.get_embeddings(seqs)  # [batch_size, seq_len, hidden_dim]

        # Initialize with noise where mask is 0
        noise = torch.randn_like(embeddings)
        x_t = torch.where(input_mask == 0, embeddings, noise)

        # Store the original embeddings as ground truth x0
        x0_ground_truth = embeddings.clone()

        # Denoising loop
        shortcuts = torch.tensor(shortcut_size, device=device).repeat(input_mask.shape[0])

        for t in torch.arange(diffusion_steps, 0, -shortcut_size, device=device):
            t_batch = t.repeat(input_mask.shape[0])
            timesteps_list.append(t.item())

            # Get model prediction (x0_hat, not velocity)
            model_output = model.criterion.model(x_t, t_batch, shortcuts)

            # Store model output
            model_outputs.append(model_output.clone())

            # Restore input part based on mask
            model_output = torch.where(input_mask == 0, x_t, model_output)

            # Calculate predicted velocity (v_hat)
            v_hat = model_output - x_t
            v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)

            # Calculate ground truth velocity
            v_ground_truth = x0_ground_truth - x_t

            # Store velocities
            predicted_velocities.append(v_hat.clone())
            ground_truth_velocities.append(v_ground_truth.clone())

            # Calculate cosine similarity between predicted and ground truth velocities
            cos_sim, pred_norm, gt_norm = calculate_batch_cosine_similarity(
                v_hat, v_ground_truth, input_mask, padding_mask.unsqueeze(-1), per_token=per_token_cosine
            )
            cosine_similarities.append(cos_sim)

            # Store velocity norms from cosine similarity calculation
            predicted_velocity_norms.append(pred_norm.mean().item())
            ground_truth_velocity_norms.append(gt_norm.mean().item())

            # Calculate L2 distance between model prediction and ground truth for valid tokens
            # Create mask for tokens we want to compute L2 distance for:
            # - input_mask == 0 (input tokens)
            # - padding_mask == 1 (non-padding tokens)
            valid_token_mask = (input_mask.squeeze(-1) == 1) & (padding_mask == 1)

            # Calculate L2 distances only for valid tokens
            l2_dists = torch.norm(
                model_output[valid_token_mask] - x0_ground_truth[valid_token_mask], dim=-1
            )  # [num_valid_tokens]

            # Average L2 distance across valid tokens
            l2_dist = l2_dists.mean().item()
            l2_distances.append(l2_dist)

            # Calculate L2 distance between predicted and ground truth velocities for valid tokens
            velocity_l2_dists = torch.norm(
                v_hat[valid_token_mask] - v_ground_truth[valid_token_mask], dim=-1
            )  # [num_valid_tokens]

            # Average velocity L2 distance across valid tokens
            velocity_l2_dist = velocity_l2_dists.mean().item()
            velocity_l2_distances.append(velocity_l2_dist)

            # Update x_t for next step
            x0_hat = x_t + (shortcuts / diffusion_steps)[:, None, None] * v_hat
            x_t = x0_hat

    return {
        "predicted_velocities": predicted_velocities,
        "ground_truth_velocities": ground_truth_velocities,
        "timesteps": timesteps_list,
        "cosine_similarities": cosine_similarities,
        "predicted_velocity_norms": predicted_velocity_norms,
        "ground_truth_velocity_norms": ground_truth_velocity_norms,
        "model_outputs": model_outputs,
        "ground_truth_embeddings": x0_ground_truth,
        "l2_distances": l2_distances,
        "velocity_l2_distances": velocity_l2_distances,
    }


def calculate_batch_cosine_similarity(
    pred_velocities: Tensor, gt_velocities: Tensor, input_mask: Tensor, padding_mask: Tensor, per_token: bool = True
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Calculate cosine similarity between predicted and ground truth velocities for a batch.
    Only considers positions that are both non-input (input_mask=1) and non-padding (padding_mask=1).

    Args:
        pred_velocities: Predicted velocities [batch_size, seq_len, hidden_dim]
        gt_velocities: Ground truth velocities [batch_size, seq_len, hidden_dim]
        input_mask: Mask indicating input positions (0 for input, 1 for target) [batch_size, seq_len, 1]
        padding_mask: Mask indicating padding (0 for padding, 1 for actual tokens) [batch_size, seq_len, 1]
        per_token: If True, calculate cosine similarity for each token separately and take the mean.
                   If False, flatten all tokens into a single vector before calculating similarity.

    Returns:
        Tensor of cosine similarities for each example in the batch [batch_size]
    """
    batch_size = pred_velocities.shape[0]
    cos_sims = []
    predicted_velocity_norms = []
    ground_truth_velocity_norms = []

    for i in range(batch_size):
        # Get masks for this example and combine them
        example_input_mask = input_mask[i].bool().squeeze(-1)  # [seq_len]
        example_padding_mask = padding_mask[i].bool().squeeze(-1)  # [seq_len]

        # Combined mask: positions that are both non-input AND non-padding
        combined_mask = example_input_mask & example_padding_mask

        # Get velocities for valid positions (non-input and non-padding)
        pred_vel = pred_velocities[i][combined_mask]  # [num_valid, hidden_dim]
        gt_vel = gt_velocities[i][combined_mask]  # [num_valid, hidden_dim]

        if per_token:
            # Calculate cosine similarity for each token separately
            if pred_vel.shape[0] > 0:
                # Calculate cosine similarity for each token [num_valid]
                token_cos_sims = torch.nn.functional.cosine_similarity(pred_vel, gt_vel)
                # Take mean across tokens
                cos_sim = token_cos_sims.mean()

                pred_norm = torch.norm(pred_vel, dim=1).mean()
                gt_norm = torch.norm(gt_vel, dim=1).mean()
            else:
                cos_sim = torch.tensor(1.0, device=pred_velocities.device)
        else:
            # Flatten to treat all non-input positions as a single vector
            if pred_vel.shape[0] > 0:
                pred_vel_flat = pred_vel.reshape(-1)  # [num_valid * hidden_dim]
                gt_vel_flat = gt_vel.reshape(-1)  # [num_valid * hidden_dim]

                # Calculate cosine similarity for the flattened vector
                cos_sim = torch.nn.functional.cosine_similarity(pred_vel_flat.unsqueeze(0), gt_vel_flat.unsqueeze(0))
            else:
                cos_sim = torch.tensor(1.0, device=pred_velocities.device)

        cos_sims.append(cos_sim)
        pred_norm = torch.norm(pred_vel, dim=1).mean()
        gt_norm = torch.norm(gt_vel, dim=1).mean()

        # Store the batch norms
        predicted_velocity_norms.append(pred_norm)
        ground_truth_velocity_norms.append(gt_norm)

    return torch.stack(cos_sims), torch.stack(predicted_velocity_norms), torch.stack(ground_truth_velocity_norms)


def visualize_cosine_similarities(
    results: dict[str, Any], figsize: tuple[int, int] = (12, 6), save_path: str | None = None
) -> None:
    """
    Visualize cosine similarities between predicted and ground truth velocities.

    Args:
        results: Results from denoise_with_velocity_tracking
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)
    """
    timesteps = results["timesteps"]
    cosine_similarities = [cs.cpu().numpy() for cs in results["cosine_similarities"]]

    # Calculate mean cosine similarity across batch
    mean_cosine_similarities = np.array([cs.mean().item() for cs in cosine_similarities])

    # Plot mean cosine similarities over timesteps
    plt.figure(figsize=figsize)
    plt.plot(timesteps, mean_cosine_similarities, marker="o", linestyle="-", linewidth=2)
    plt.title("Cosine Similarity Between Predicted and Ground Truth Velocities During Denoising")
    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.ylim(-1.1, 1.1)  # Cosine similarity range
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_per_example_cosine_similarities(
    results: dict[str, Any], figsize: tuple[int, int] = (14, 8), save_path: str | None = None
) -> None:
    """
    Visualize cosine similarities for each example in the batch.

    Args:
        results: Results from denoise_with_velocity_tracking
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)
    """
    timesteps = results["timesteps"]
    cosine_similarities = results["cosine_similarities"]

    # Get number of examples in batch
    num_examples = cosine_similarities[0].shape[0]

    # Plot cosine similarities for each example
    plt.figure(figsize=figsize)

    for i in range(num_examples):
        example_cos_sims = [cs[i].item() for cs in cosine_similarities]
        plt.plot(timesteps, example_cos_sims, marker=".", linestyle="-", label=f"Example {i + 1}")

    plt.title("Cosine Similarity by Example During Denoising")
    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.ylim(-1.1, 1.1)  # Cosine similarity range
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def compare_cosine_similarity_methods(
    model, batch, shortcut_size, figsize: tuple[int, int] = (12, 6), save_path: str | None = None
) -> dict[str, Any]:
    """
    Compare per-token vs flattened cosine similarity calculations.

    Args:
        model: The model to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)

    Returns:
        Dictionary with results from both methods
    """
    # Run denoising with both methods
    results_per_token = denoise_with_tracking(model, batch, shortcut_size, per_token_cosine=True)
    results_flattened = denoise_with_tracking(model, batch, shortcut_size, per_token_cosine=False)

    # Extract results
    timesteps = results_per_token["timesteps"]
    cosine_similarities_per_token = np.array([cs.mean().item() for cs in results_per_token["cosine_similarities"]])
    cosine_similarities_flattened = np.array([cs.mean().item() for cs in results_flattened["cosine_similarities"]])

    # Plot comparison
    plt.figure(figsize=figsize)
    plt.plot(timesteps, cosine_similarities_per_token, marker="o", linestyle="-", linewidth=2, label="Per-token")
    plt.plot(timesteps, cosine_similarities_flattened, marker="s", linestyle="--", linewidth=2, label="Flattened")
    plt.title("Comparison of Cosine Similarity Calculation Methods")
    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.ylim(-1.1, 1.1)  # Cosine similarity range
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return {"timesteps": timesteps, "per_token": results_per_token, "flattened": results_flattened}


def analyze_multiple_batches(
    model,
    dataloader,
    shortcut_size,
    num_batches: int = 5,
    per_token_cosine: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """
    Analyze velocity predictions across multiple batches.

    Args:
        model: The model to use for denoising
        dataloader: DataLoader to get batches from
        shortcut_size: The shortcut size to use for denoising
        num_batches: Number of batches to process
        per_token_cosine: If True, calculate cosine similarity for each token separately and take the mean.
                          If False, flatten all tokens into a single vector before calculating similarity.
        device: The device to use for computation

    Returns:
        Dictionary with aggregated statistics
    """
    all_cosine_sims = []
    timesteps = None

    # Process multiple batches
    dataloader_iter = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break

        # Run denoising with tracking
        results = denoise_with_tracking(model, batch, shortcut_size, per_token_cosine=per_token_cosine)

        # Store timesteps from first batch
        if timesteps is None:
            timesteps = results["timesteps"]

        # Extract and store cosine similarities
        batch_cos_sims = [cs.cpu().numpy() for cs in results["cosine_similarities"]]
        all_cosine_sims.append(batch_cos_sims)

    # Combine results across batches
    combined_cos_sims = []
    for t in range(len(timesteps)):
        # Collect cosine similarities for this timestep across all batches
        timestep_cos_sims = np.concatenate([batch[t] for batch in all_cosine_sims])
        combined_cos_sims.append(timestep_cos_sims)

    # Calculate statistics
    mean_cos_sims = np.array([cs.mean() for cs in combined_cos_sims])
    std_cos_sims = np.array([cs.std() for cs in combined_cos_sims])
    min_cos_sims = np.array([cs.min() for cs in combined_cos_sims])
    max_cos_sims = np.array([cs.max() for cs in combined_cos_sims])

    return {
        "timesteps": timesteps,
        "mean": mean_cos_sims,
        "std": std_cos_sims,
        "min": min_cos_sims,
        "max": max_cos_sims,
        "all_data": combined_cos_sims,
    }


def visualize_batch_statistics(
    batch_results: dict[str, Any], figsize: tuple[int, int] = (14, 8), save_path: str | None = None
) -> None:
    """
    Visualize statistics from multiple batch analysis.

    Args:
        batch_results: Results from analyze_multiple_batches
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)
    """
    timesteps = batch_results["timesteps"]
    mean_cos_sims = batch_results["mean"]
    std_cos_sims = batch_results["std"]

    plt.figure(figsize=figsize)

    plt.plot(timesteps, mean_cos_sims, "b-", linewidth=2, label="Mean Cosine Similarity")
    plt.fill_between(
        timesteps, mean_cos_sims - std_cos_sims, mean_cos_sims + std_cos_sims, alpha=0.3, color="b", label="±1 Std Dev"
    )

    plt.plot(timesteps, batch_results["min"], "r--", linewidth=1, label="Min Cosine Similarity")
    plt.plot(timesteps, batch_results["max"], "g--", linewidth=1, label="Max Cosine Similarity")

    plt.title("Mean Cosine Similarity Between Predicted and Ground Truth Velocities (Multiple Batches)")
    plt.xlabel("Timestep")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)
    plt.ylim(-1.1, 1.1)  # Cosine similarity range
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_cosine_similarity_distribution(
    batch_results: dict[str, Any],
    num_timesteps: int = 5,
    figsize: tuple[int, int] = (18, 12),
    save_path: str | None = None,
) -> None:
    """
    Visualize the distribution of cosine similarities at different timesteps.

    Args:
        batch_results: Results from analyze_multiple_batches
        num_timesteps: Number of timesteps to visualize
        figsize: Figure size for the plot
        save_path: Path to save the figure (optional)
    """
    timesteps = batch_results["timesteps"]
    _ = batch_results["all_data"]

    # Select timesteps to visualize
    num_steps = len(timesteps)
    if num_steps <= num_timesteps:
        indices_to_plot = list(range(num_steps))
        timesteps_to_plot = timesteps
    else:
        # Select evenly spaced timesteps
        indices_to_plot = [int(i * (num_steps - 1) / (num_timesteps - 1)) for i in range(num_timesteps)]
        timesteps_to_plot = [timesteps[i] for i in indices_to_plot]

    # Create subplots
    fig, axes = plt.subplots(nrows=(num_timesteps + 2) // 3, ncols=min(3, num_timesteps), figsize=figsize)
    axes = axes.flatten() if num_steps > 1 else [axes]

    for i, (idx, ts) in enumerate(zip(indices_to_plot, timesteps_to_plot, strict=False)):
        ax = axes[i]
        cos_sims = batch_results["all_data"][idx]

        ax.hist(cos_sims, bins=30, alpha=0.7, color="blue")
        ax.axvline(x=cos_sims.mean(), color="r", linestyle="--", linewidth=2, label=f"Mean: {cos_sims.mean():.4f}")

        ax.set_title(f"Distribution of Cosine Similarities at Timestep {ts}")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-1.1, 1.1)
        ax.legend()

    # Hide unused subplots
    for i in range(len(indices_to_plot), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def calculate_noise_to_target_distance(
    target_embeddings: Tensor,
    input_mask: Tensor,
    padding_mask: Tensor,
    num_samples: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, float]:
    """
    Calculate L2 distance between random noise and target embeddings.

    Args:
        target_embeddings: Target embeddings [batch_size, seq_len, hidden_dim]
        input_mask: Mask indicating input positions (0 for input, 1 for target) [batch_size, seq_len, 1]
        padding_mask: Mask indicating padding (0 for padding, 1 for actual tokens) [batch_size, seq_len, 1]
        num_samples: Number of random noise samples to generate
        device: The device to use for computation

    Returns:
        Dictionary containing:
        - mean_distance: Mean L2 distance across all samples
        - std_distance: Standard deviation of L2 distances
        - min_distance: Minimum L2 distance observed
        - max_distance: Maximum L2 distance observed
    """
    distances = []

    # Create mask for tokens we want to compute L2 distance for:
    # - input_mask == 0 (input tokens)
    # - padding_mask == 1 (non-padding tokens)
    valid_token_mask = (input_mask.squeeze(-1) == 1) & (padding_mask == 1)

    for _ in range(num_samples):
        # Generate random noise with same shape as target embeddings
        noise = torch.randn_like(target_embeddings)

        # Calculate L2 distances only for valid tokens
        l2_dists = torch.norm(
            noise[valid_token_mask] - target_embeddings[valid_token_mask], dim=-1
        )  # [num_valid_tokens]

        # Average L2 distance across valid tokens
        mean_dist = l2_dists.mean().item()
        distances.append(mean_dist)

    distances = torch.tensor(distances)
    return {
        "mean_distance": distances.mean().item(),
        "std_distance": distances.std().item(),
        "min_distance": distances.min().item(),
        "max_distance": distances.max().item(),
    }


def analyze_velocity_predictions(
    model,
    batch,
    shortcut_size,
    per_token_cosine: bool = True,
    compare_methods: bool = False,
    output_dir: str | None = None,
    prefix: str = "velocity_analysis",
) -> dict[str, Any]:
    """
    Analyze velocity predictions during denoising.

    Args:
        model: The model to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        per_token_cosine: If True, calculate cosine similarity for each token separately and take the mean.
                          If False, flatten all tokens into a single vector before calculating similarity.
        compare_methods: Whether to compare per-token vs flattened methods
        output_dir: Directory to save figures (optional)
        prefix: Prefix for saved figure filenames

    Returns:
        Results from denoise_with_tracking
    """
    print("Running velocity tracking...")

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Compare methods if requested
    if compare_methods:
        save_path = os.path.join(output_dir, f"{prefix}_method_comparison.png") if output_dir else None
        comparison_results = compare_cosine_similarity_methods(model, batch, shortcut_size, save_path=save_path)
        results = comparison_results["per_token"] if per_token_cosine else comparison_results["flattened"]
    else:
        # Run denoising with velocity tracking
        results = denoise_with_tracking(model, batch, shortcut_size, per_token_cosine=per_token_cosine)

        # Visualize cosine similarities
        save_path = os.path.join(output_dir, f"{prefix}_cosine_similarities.png") if output_dir else None
        visualize_cosine_similarities(results, save_path=save_path)

        # Visualize per-example cosine similarities
        save_path = os.path.join(output_dir, f"{prefix}_per_example.png") if output_dir else None
        visualize_per_example_cosine_similarities(results, save_path=save_path)

    print("Velocity analysis complete.")
    return results


if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze velocity predictions during denoising")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint directory")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Specific checkpoint file (if not provided, will use the latest in the directory)",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="datasets/tokenized/bert-base-uncased/QQP-Official/test",
        help="Path to test data",
    )
    parser.add_argument("--shortcut_size", type=int, default=1024, help="Shortcut size for denoising")
    parser.add_argument("--per-token", action="store_true", help="Calculate cosine similarity per token")
    parser.add_argument("--compare-methods", action="store_true", help="Compare per-token vs flattened methods")
    parser.add_argument("--output-dir", type=str, default="reports/figures", help="Directory to save figures")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to analyze")
    args = parser.parse_args()

    # Import required modules
    from pathlib import Path

    import lightning as pl
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from datasets import Dataset
    from shortcutfm.batch import collate
    from shortcutfm.config import TrainingConfig
    from shortcutfm.text_datasets import TextDataset
    from shortcutfm.train.pl.trainer_factory import create_criterion, load_unit_from_checkpoint

    # Set checkpoint directory and path
    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Checkpoint directory: {checkpoint_dir}")
    if args.checkpoint:
        checkpoint_path = checkpoint_dir / args.checkpoint
    else:
        # Find the latest checkpoint in the directory
        checkpoints = list(checkpoint_dir.glob("*.ckpt"))
        if not checkpoints:
            # Try listing all files in the directory
            all_files = list(checkpoint_dir.glob("**/*.ckpt"))
            checkpoints = [f for f in all_files if f.name.endswith(".ckpt")]

        if not checkpoints:
            raise ValueError(f"No checkpoints found in {checkpoint_dir} or its subdirectories")

        checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
        print(f"Found checkpoint: {checkpoint_path}")

    # Find training config file
    training_config_path = checkpoint_dir / "training_config.yaml"
    if not training_config_path.exists():
        # Try looking in the parent directory of the checkpoint
        if isinstance(checkpoint_path, Path) and checkpoint_path.parent != checkpoint_dir:
            training_config_path = checkpoint_path.parent / "training_config.yaml"

    if not training_config_path.exists():
        raise ValueError(f"Training config not found at {training_config_path}")

    print(f"Using training config from: {training_config_path}")

    # Load training configuration
    with open(training_config_path) as f:
        yaml_cfg = OmegaConf.load(f)

    training_config = TrainingConfig(**OmegaConf.to_container(yaml_cfg, resolve=True))
    print(f"Loaded training config from {training_config_path}")

    # Set random seed for reproducibility
    pl.seed_everything(training_config.seed)

    # Create criterion and load model from checkpoint
    criterion = create_criterion(training_config)
    unit = load_unit_from_checkpoint(criterion, checkpoint_path, training_config)
    print(f"Loaded model from {checkpoint_path}")

    # Set the model to evaluation mode
    unit.eval()

    # Load test dataset
    test_data_path = args.test_data_path
    test_ds = Dataset.load_from_disk(test_data_path)
    test_text_ds = TextDataset(test_ds)

    # Create a dataloader for analysis
    test_dataloader = DataLoader(
        test_text_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_batches > 1:
        # Analyze multiple batches
        print(f"Analyzing {args.num_batches} batches...")
        batch_results = analyze_multiple_batches(
            unit, test_dataloader, args.shortcut_size, num_batches=args.num_batches, per_token_cosine=args.per_token
        )

        # Visualize batch statistics
        save_path = os.path.join(args.output_dir, "velocity_batch_statistics.png")
        visualize_batch_statistics(batch_results, save_path=save_path)

        # Visualize cosine similarity distribution
        save_path = os.path.join(args.output_dir, "velocity_cosine_distribution.png")
        visualize_cosine_similarity_distribution(batch_results, save_path=save_path)

        print(f"Analysis complete. Figures saved to {args.output_dir}")
    else:
        # Get a single batch for analysis
        test_batch = next(iter(test_dataloader))
        print(f"Loaded test batch with {len(test_batch.seqs)} examples")

        # Run analysis
        analyze_velocity_predictions(
            unit,
            test_batch,
            args.shortcut_size,
            per_token_cosine=args.per_token,
            compare_methods=args.compare_methods,
            output_dir=args.output_dir,
        )

        print(f"Analysis complete. Figures saved to {args.output_dir}")
