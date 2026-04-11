"""
Visualize denoising flow paths using PCA/t-SNE.

Two visualization variants:
1. "global": Flatten all token embeddings per example into a single vector,
   then t-SNE across denoising steps → shows the trajectory of the full sequence.
2. "per_token": For a single example, treat each token embedding independently,
   then t-SNE across (token, step) pairs → shows how individual tokens move.

Usage:
    uv run python scripts/visualize_flow_paths.py \
        --training_config checkpoints/qqp/scut_768/run_ll9cnmi5/training_config.yaml \
        --checkpoint checkpoints/qqp/scut_768/run_ll9cnmi5/epoch=147-step=21000-val_bleu=0.0000.ckpt \
        --output_dir flow_visualizations/qqp_scut_768 \
        --nfe 1 2 4 8 64 128 256 \
        --example_idx 0 \
        --seed 44
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer

import lightning as pl
from shortcutfm.batch import EncoderBatch, collate
from shortcutfm.config import TrainingConfig
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.trainer_factory import (
    create_criterion,
    get_ema_callback,
    load_unit_from_checkpoint,
)

from datasets import Dataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DIFFUSION_STEPS = 2048


@torch.no_grad()
def collect_trajectories(
    criterion,
    batch: EncoderBatch,
    step_size: int,
    shortcut_size: int,
) -> list[torch.Tensor]:
    """Run denoising and capture x_t at every step.

    Returns list of tensors, each [batch_size, seq_len, hidden_dim].
    First element is the initial noisy state, last is the final prediction.
    """
    criterion.model.eval()
    criterion._reset()

    input_mask = batch.input_ids_mask.unsqueeze(-1)
    embeddings = criterion.model.get_embeddings(batch.seqs)
    noise = torch.randn_like(embeddings)
    x_t = torch.where(input_mask == 0, embeddings, noise)

    trajectory = [x_t.cpu()]

    shortcuts = torch.tensor(shortcut_size, device=input_mask.device).repeat(input_mask.shape[0])
    for t_val in torch.arange(DIFFUSION_STEPS, 0, -step_size, device=input_mask.device):
        t = t_val.repeat(input_mask.shape[0])
        model_output = criterion.infere_model(x_t, t, shortcuts, input_mask)
        v_hat = criterion.compute_velocity(model_output, noise, input_mask)
        x0_hat = x_t + (step_size / DIFFUSION_STEPS) * v_hat
        x_t = x0_hat
        trajectory.append(x_t.cpu())
        criterion._reset()

    # Also capture the ground truth embedding
    gt_embedding = embeddings.cpu()
    return trajectory, gt_embedding


def reduce_dim(points: np.ndarray, method: str = "tsne", perplexity: int = 30) -> np.ndarray:
    """Reduce to 2D using PCA or t-SNE."""
    if points.shape[0] < 4:
        # Too few points for t-SNE, use PCA
        method = "pca"
    if method == "pca":
        return PCA(n_components=2).fit_transform(points)
    else:
        perp = min(perplexity, max(2, points.shape[0] - 1))
        return TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000).fit_transform(points)


def make_colored_line(xs, ys, cmap_name="viridis", linewidth=2.0, alpha=0.8):
    """Create a line collection with color gradient along the path."""
    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(xs) - 1)
    lc = LineCollection(segments, cmap=cmap_name, norm=norm, linewidth=linewidth, alpha=alpha)
    lc.set_array(np.arange(len(xs)))
    return lc


def plot_global_trajectories(
    all_trajectories: dict[int, tuple[list[torch.Tensor], torch.Tensor]],
    example_indices: list[int],
    output_dir: Path,
    method: str = "tsne",
):
    """Variant 1: Flatten all token embeddings → single vector per step, then reduce."""
    for ex_idx in example_indices:
        # Collect all points across all NFEs for joint embedding
        all_points = []
        nfe_labels = []
        step_labels = []

        for nfe, (trajectory, gt_emb) in sorted(all_trajectories.items()):
            for step_idx, x_t in enumerate(trajectory):
                vec = x_t[ex_idx].flatten().numpy()  # [seq_len * hidden_dim]
                all_points.append(vec)
                nfe_labels.append(nfe)
                step_labels.append(step_idx)
            # Add ground truth
            gt_vec = gt_emb[ex_idx].flatten().numpy()
            all_points.append(gt_vec)
            nfe_labels.append(-1)  # marker for GT
            step_labels.append(-1)

        all_points = np.array(all_points)
        coords_2d = reduce_dim(all_points, method=method)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.get_cmap("tab10")

        nfe_values = sorted(all_trajectories.keys())
        for i, nfe in enumerate(nfe_values):
            mask = np.array(nfe_labels) == nfe
            pts = coords_2d[mask]
            color = cmap(i / max(len(nfe_values) - 1, 1))

            # Draw path
            lc = make_colored_line(pts[:, 0], pts[:, 1], linewidth=2.0, alpha=0.6)
            ax.add_collection(lc)

            # Mark start and end
            ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=100, zorder=5, edgecolors="black")
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="*", s=200, zorder=5, edgecolors="black")

            # Intermediate steps
            if len(pts) > 2:
                ax.scatter(pts[1:-1, 0], pts[1:-1, 1], color=color, marker=".", s=30, alpha=0.5, zorder=4)

            ax.plot([], [], color=color, label=f"NFE={nfe} ({len(pts)} steps)", linewidth=2)

        # Plot ground truth
        gt_mask = np.array(nfe_labels) == -1
        gt_pts = coords_2d[gt_mask]
        if len(gt_pts) > 0:
            ax.scatter(gt_pts[0, 0], gt_pts[0, 1], color="red", marker="X", s=300, zorder=10,
                       edgecolors="black", linewidths=1.5, label="Ground Truth")

        ax.legend(fontsize=9, loc="best")
        ax.set_title(f"Flow Paths — Global View (example {ex_idx}, {method.upper()})")
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.autoscale()
        plt.tight_layout()

        fname = output_dir / f"global_{method}_example{ex_idx}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def plot_per_token_trajectories(
    all_trajectories: dict[int, tuple[list[torch.Tensor], torch.Tensor]],
    example_idx: int,
    token_indices: list[int],
    output_dir: Path,
    method: str = "tsne",
):
    """Variant 2: Per-token embeddings for a single example across steps."""
    for nfe, (trajectory, gt_emb) in sorted(all_trajectories.items()):
        num_steps = len(trajectory)
        if num_steps < 2:
            continue

        # Collect points: for each selected token, gather its embedding at each step
        all_points = []
        token_labels = []
        step_labels_arr = []

        for tok_idx in token_indices:
            for step_idx, x_t in enumerate(trajectory):
                vec = x_t[example_idx, tok_idx].numpy()  # [hidden_dim]
                all_points.append(vec)
                token_labels.append(tok_idx)
                step_labels_arr.append(step_idx)
            # Ground truth for this token
            gt_vec = gt_emb[example_idx, tok_idx].numpy()
            all_points.append(gt_vec)
            token_labels.append(tok_idx)
            step_labels_arr.append(-1)

        all_points = np.array(all_points)
        if all_points.shape[0] < 3:
            continue

        coords_2d = reduce_dim(all_points, method=method)

        fig, ax = plt.subplots(figsize=(12, 10))
        cmap = plt.cm.get_cmap("tab10")

        for i, tok_idx in enumerate(token_indices):
            mask = np.array(token_labels) == tok_idx
            step_mask = np.array(step_labels_arr) != -1
            pts = coords_2d[mask & step_mask]
            color = cmap(i / max(len(token_indices) - 1, 1))

            if len(pts) > 1:
                lc = make_colored_line(pts[:, 0], pts[:, 1], linewidth=2.0, alpha=0.6)
                ax.add_collection(lc)

            ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=80, zorder=5, edgecolors="black")
            ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="*", s=150, zorder=5, edgecolors="black")

            # Ground truth
            gt_pt = coords_2d[mask & ~step_mask]
            if len(gt_pt) > 0:
                ax.scatter(gt_pt[0, 0], gt_pt[0, 1], color=color, marker="X", s=200, zorder=10,
                           edgecolors="black", linewidths=1.5)

            ax.plot([], [], color=color, label=f"Token {tok_idx}", linewidth=2)

        ax.legend(fontsize=9, loc="best")
        ax.set_title(f"Per-Token Flow Paths — NFE={nfe} (example {example_idx}, {method.upper()})")
        ax.set_xlabel(f"{method.upper()} dim 1")
        ax.set_ylabel(f"{method.upper()} dim 2")
        ax.autoscale()
        plt.tight_layout()

        fname = output_dir / f"per_token_{method}_nfe{nfe}_example{example_idx}.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Visualize denoising flow paths")
    parser.add_argument("--training_config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 64, 128, 256])
    parser.add_argument("--example_idx", type=int, default=0, help="Example index for per-token viz")
    parser.add_argument("--num_tokens", type=int, default=8, help="Number of tokens to visualize in per-token mode")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--method", choices=["tsne", "pca", "both"], default="both")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training config
    with open(args.training_config) as f:
        yaml_cfg = OmegaConf.load(f)
    training_cfg = TrainingConfig(**OmegaConf.to_container(yaml_cfg, resolve=True))

    tokenizer = AutoTokenizer.from_pretrained(training_cfg.model.tokenizer_config_name)

    # Load model
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    criterion = create_criterion(training_cfg, tokenizer=tokenizer)
    unit = load_unit_from_checkpoint(
        criterion, args.checkpoint, training_cfg,
        tokenizer=tokenizer,
        denoising_step_size=DIFFUSION_STEPS,
        prediction_shortcut_size=DIFFUSION_STEPS,
    )

    if args.use_ema:
        ema_cb = get_ema_callback(training_cfg, args.checkpoint)
        ema_cb.on_test_start(None, unit)

    unit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit = unit.to(device)

    # Get the flow matching criterion from the composite
    fm_criterion = unit.criterion
    if hasattr(fm_criterion, "flow_matching_criterion"):
        fm_criterion = fm_criterion.flow_matching_criterion
    # Unwrap decorator if needed
    while hasattr(fm_criterion, "criterion"):
        fm_criterion = fm_criterion.criterion

    # Load test data — just one batch
    from shortcutfm.decoding.generation_runner import determine_test_data_path
    test_data_path = determine_test_data_path(args.training_config, args.split)
    test_ds = Dataset.load_from_disk(test_data_path)
    test_text_ds = TextDataset(test_ds)
    test_loader = DataLoader(test_text_ds, batch_size=args.batch_size, collate_fn=collate, shuffle=False)
    batch = next(iter(test_loader))
    batch = batch.to(device)
    logger.info(f"Loaded batch: {batch.seqs.shape[0]} examples, seq_len={batch.seqs.shape[1]}")

    # Collect trajectories for each NFE
    all_trajectories: dict[int, tuple[list[torch.Tensor], torch.Tensor]] = {}
    for nfe in args.nfe:
        step_size = DIFFUSION_STEPS // nfe
        shortcut_size = step_size  # For shortcut models, these match
        logger.info(f"Collecting trajectory for NFE={nfe} (step_size={step_size})")

        # Reset seed for consistent noise across NFEs
        torch.manual_seed(args.seed)
        trajectory, gt_emb = collect_trajectories(fm_criterion, batch, step_size, shortcut_size)
        all_trajectories[nfe] = (trajectory, gt_emb)
        logger.info(f"  Captured {len(trajectory)} states (including initial noise)")

    # Select token indices for per-token visualization
    # Pick tokens from the target part (where input_ids_mask == 1)
    mask = batch.input_ids_mask[args.example_idx].cpu().numpy()
    target_token_indices = np.where(mask == 1)[0]
    if len(target_token_indices) == 0:
        target_token_indices = np.arange(min(args.num_tokens, batch.seqs.shape[1]))
    else:
        step = max(1, len(target_token_indices) // args.num_tokens)
        target_token_indices = target_token_indices[::step][:args.num_tokens]
    logger.info(f"Per-token visualization tokens: {target_token_indices.tolist()}")

    # Generate visualizations
    methods = ["pca", "tsne"] if args.method == "both" else [args.method]
    example_indices = [args.example_idx]

    for method in methods:
        logger.info(f"Generating {method.upper()} visualizations...")
        plot_global_trajectories(all_trajectories, example_indices, output_dir, method=method)
        plot_per_token_trajectories(
            all_trajectories, args.example_idx, target_token_indices.tolist(), output_dir, method=method
        )

    logger.info(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
