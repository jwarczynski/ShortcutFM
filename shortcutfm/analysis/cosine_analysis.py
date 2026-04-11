"""
Cosine similarity analysis module for generation evaluation.
Based on the cosine_similarity_analysis.ipynb notebook.
"""

import json
import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Helper function to safely convert tensors or arrays to numpy
def to_numpy_mean(data):
    """Safely convert torch tensors or numpy arrays to scalar means."""
    if hasattr(data, 'cpu'):  # torch tensor
        return data.cpu().numpy().mean()
    elif hasattr(data, 'mean'):  # numpy array
        return data.mean()
    else:  # scalar
        return float(data)

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("Matplotlib/numpy not available. Skipping plot generation.")


def plot_cosine_and_l2(results_per_token, save_path: Path, title_suffix: str = ""):
    """Plot cosine similarities and L2 distances from velocity tracking results."""
    if not HAS_PLOTTING:
        logger.warning("Skipping plot generation - matplotlib not available")
        return

    # Extract results
    timesteps = results_per_token["timesteps"]
    # Take mean across batch dimension for each metric
    cosine_similarities_per_token = [to_numpy_mean(cs) for cs in results_per_token["cosine_similarities"]]
    l2_distances = [to_numpy_mean(d) for d in results_per_token["l2_distances"]]
    velocity_l2_distances = [to_numpy_mean(d) for d in results_per_token["velocity_l2_distances"]]
    predicted_velocity_norms = [to_numpy_mean(n) for n in results_per_token["predicted_velocity_norms"]]
    ground_truth_velocity_norms = [to_numpy_mean(n) for n in results_per_token["ground_truth_velocity_norms"]]

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # First subplot - Cosine similarities and L2 distances
    ax1.plot(timesteps, cosine_similarities_per_token, marker='o', linestyle='-', linewidth=2, color='blue', label='Per-token Cosine')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Cosine Similarity', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True)

    # Create twin axis for L2 distances
    ax1_twin = ax1.twinx()
    ax1_twin.plot(timesteps, l2_distances, marker='^', linestyle=':', linewidth=2, color='red', label='L2 Distance')
    ax1_twin.set_ylabel('L2 Distance', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')

    # Add title and legend for first subplot
    ax1.set_title(f'Cosine Similarities and L2 Distance{title_suffix}')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Second subplot - Velocity L2 distances and norms
    ax2.plot(timesteps, velocity_l2_distances, marker='o', linestyle='-', linewidth=2, color='purple', label='Velocity L2 Distance')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Velocity L2 Distance', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.grid(True)

    # Create twin axis for velocity norms
    ax2_twin = ax2.twinx()
    ax2_twin.plot(timesteps, predicted_velocity_norms, marker='s', linestyle='--', linewidth=2, color='orange', label='Predicted Velocity Norm')
    ax2_twin.plot(timesteps, ground_truth_velocity_norms, marker='^', linestyle=':', linewidth=2, color='brown', label='Ground Truth Velocity Norm')
    ax2_twin.set_ylabel('Velocity Norm', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    # Add title and legend for second subplot
    ax2.set_title(f'Velocity L2 Distance and Norms{title_suffix}')
    lines3, labels3 = ax2.get_legend_handles_labels()
    lines4, labels4 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines3 + lines4, labels3 + labels4, loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cosine similarity plot to {save_path}")


def run_cosine_similarity_analysis(
    unit,
    test_dataloader: DataLoader,
    output_dir: Path,
    generation_shortcut_size: int,
    denoising_step_size: int,
    analysis_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run cosine similarity analysis comparing ground truth interpolation vs without.

    Args:
        unit: The loaded model unit
        test_dataloader: DataLoader for test data
        output_dir: Directory to save plots and analysis results
        generation_shortcut_size: Shortcut size used for generation
        denoising_step_size: Step size for denoising
        analysis_config: Configuration for analysis parameters

    Returns:
        Dictionary with analysis results
    """
    if analysis_config is None:
        analysis_config = {
            "num_examples": 3,
            "use_ground_truth_embeddings": [True, False]
        }

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Get a test batch for analysis
    test_batch = next(iter(test_dataloader))

    # Move batch to the same device as the model
    device = next(unit.parameters()).device
    test_batch.seqs = test_batch.seqs.to(device)
    test_batch.input_ids_mask = test_batch.input_ids_mask.to(device)
    test_batch.padding_mask = test_batch.padding_mask.to(device)

    logger.info(f"Running cosine similarity analysis with shortcut_size={generation_shortcut_size}, step_size={denoising_step_size}")

    results = {}

    # Test different ground truth interpolation settings
    for use_ground_truth in analysis_config.get("use_ground_truth_embeddings", [True, False]):
        try:
            # Import the denoising function
            from shortcutfm.analysis.denoising import denoise_with_velocity_tracking

            # Run denoising with velocity tracking
            velocity_results = denoise_with_velocity_tracking(
                unit.criterion.flow_matching_criterion,
                test_batch,
                shortcut_size=generation_shortcut_size,
                per_token_cosine=True,
                step_size=denoising_step_size,
                use_ground_truth_interpolation=use_ground_truth,
                velocity_scale=1.0,
            )

            # Create plot
            gt_suffix = "_with_gt_interpolation" if use_ground_truth else "_without_gt_interpolation"
            title_suffix = " (with GT interpolation)" if use_ground_truth else " (without GT interpolation)"
            plot_filename = f"cosine_velocity_analysis_scut_{generation_shortcut_size}_step_{denoising_step_size}{gt_suffix}.png"
            plot_path = plots_dir / plot_filename

            plot_cosine_and_l2(velocity_results, plot_path, title_suffix)

            # Store results for summary
            timesteps = velocity_results["timesteps"]
            cosine_sims = [to_numpy_mean(cs) for cs in velocity_results["cosine_similarities"]]
            results[f"ground_truth_interpolation_{use_ground_truth}"] = {
                "timesteps": timesteps,
                "mean_cosine_similarity": float(sum(cosine_sims) / len(cosine_sims)) if cosine_sims else None,
                "final_cosine_similarity": float(cosine_sims[-1]) if cosine_sims else None,
                "plot_file": str(plot_filename)
            }

            logger.info(f"Completed analysis with ground truth interpolation = {use_ground_truth}")

        except ImportError as e:
            logger.warning(f"Could not import denoising analysis module: {e}")
            results[f"ground_truth_interpolation_{use_ground_truth}"] = {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in cosine similarity analysis with ground truth = {use_ground_truth}: {e}")
            results[f"ground_truth_interpolation_{use_ground_truth}"] = {"error": str(e)}

    # Create comparison plot if both analyses succeeded
    if HAS_PLOTTING and len(results) == 2 and all("error" not in r for r in results.values()):
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            for use_gt in [True, False]:
                result_key = f"ground_truth_interpolation_{use_gt}"
                if result_key in results and "timesteps" in results[result_key]:
                    timesteps = results[result_key]["timesteps"]
                    # Re-run analysis to get detailed cosine similarities for comparison plot
                    velocity_results = denoise_with_velocity_tracking(
                        unit.criterion.flow_matching_criterion,
                        test_batch,
                        shortcut_size=generation_shortcut_size,
                        per_token_cosine=True,
                        step_size=denoising_step_size,
                        use_ground_truth_interpolation=use_gt,
                        velocity_scale=1.0,
                    )
                    # Use the same safe conversion function
                    cosine_sims = [to_numpy_mean(cs) for cs in velocity_results["cosine_similarities"]]

                    label = "With GT interpolation" if use_gt else "Without GT interpolation"
                    style = '-' if use_gt else '--'
                    ax.plot(timesteps, cosine_sims, label=label, linestyle=style, linewidth=2, marker='o')

            ax.set_xlabel('Timestep')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f'Cosine Similarity Comparison (shortcut={generation_shortcut_size}, step={denoising_step_size})')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 1.1)

            comparison_plot_path = plots_dir / f"cosine_similarity_comparison_scut_{generation_shortcut_size}_step_{denoising_step_size}.png"
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            results["comparison_plot"] = str(comparison_plot_path.name)
            logger.info(f"Saved comparison plot to {comparison_plot_path}")

        except Exception as e:
            logger.error(f"Error creating comparison plot: {e}")

    # Save results summary (but not as analysis_results.json to avoid confusion)
    summary_file = plots_dir / "cosine_analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Cosine similarity analysis completed. Results saved to {plots_dir}")
    return results
