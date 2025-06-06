"""Token analysis module for ShortcutFM.

This module provides functions for analyzing token probabilities and L2 distances
during the denoising process.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def denoise_with_token_tracking(model, batch, shortcut_size, top_k=5, example_idx=0, device="cpu") -> dict[str, Any]:
    """
    Perform denoising while tracking model predictions and token probabilities for visualization.

    Args:
        model: The model to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        top_k: Number of top tokens to track at each step
        example_idx: Index of the example in the batch to track
        device: The device to use for computation

    Returns:
        A dictionary containing:
        - timesteps: Timesteps used during denoising
        - token_probs: Top-k token probabilities at each step for each position
        - token_ids: Top-k token IDs at each step for each position
        - token_texts: Decoded tokens at each step for each position
        - l2_distances: L2 distances between predicted embeddings and token embeddings
        - loss_positions: Positions that contribute to the loss (non-input, non-padding)
        - ground_truth_tokens: Ground truth tokens at each position
        - input_tokens: Input tokens at each position
        - input_mask: Mask indicating input positions (0 for input, 1 for target)
        - padding_mask: Mask indicating padding (0 for padding, 1 for actual tokens)
        - x0_hat: Predicted clean embeddings at each timestep
        - model: The model used for denoising
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get model parameters
    diffusion_steps = model.criterion.model.diffusion_steps
    # Access tokenizer through flow_matching_criterion in composite criterion
    tokenizer = model.criterion.flow_matching_criterion.tokenizer

    # Move individual tensors in batch to device
    seqs = batch.seqs.to(device)
    input_ids_mask = batch.input_ids_mask.to(device)
    padding_mask = batch.padding_mask.to(device)

    # Select a single example to track
    seq = seqs[example_idx : example_idx + 1]
    input_mask: Tensor = input_ids_mask[example_idx : example_idx + 1].unsqueeze(-1)
    pad_mask = padding_mask[example_idx : example_idx + 1].unsqueeze(-1)

    # Get embeddings
    embeddings = model.criterion.model.get_embeddings(seq)

    # Calculate loss mask (positions that contribute to loss)
    loss_mask = (input_mask.bool() & pad_mask.bool()).squeeze(-1).squeeze(0)
    loss_positions = loss_mask.nonzero().squeeze(-1).cpu().numpy()

    # Get ground truth tokens
    ground_truth_tokens = []
    input_tokens = []
    for i, token_id in enumerate(seq[0].cpu().numpy()):
        token_text = tokenizer.decode([token_id])
        if i in loss_positions:
            ground_truth_tokens.append(token_text)
        else:
            input_tokens.append(token_text)

    # Initialize tracking variables
    timesteps_list = []
    token_probs_list = []
    token_ids_list = []
    token_texts_list = []
    l2_distances_list = []
    x0_hat_list = []  # Store predicted clean embeddings

    # Denoising loop
    shortcuts = torch.tensor(shortcut_size, device=device).repeat(input_mask.shape[0])

    # Get word embedding matrix for L2 distance calculation
    word_embeddings = model.criterion.model.module.word_embedding.weight

    with torch.no_grad():
        # Start with random noise for non-input positions
        noise = torch.randn_like(embeddings)
        x_t = torch.where(input_mask == 0, embeddings, noise)

        # Denoising steps
        for t in torch.arange(diffusion_steps, 0, -shortcut_size, device=device):
            # Convert to tensor and move to device
            timesteps = torch.full((input_mask.shape[0],), t, device=device, dtype=torch.long)

            # Get model prediction (x0, not velocity)
            x0_hat = model.criterion.model(x_t, timesteps, shortcuts)
            x0_hat_list.append(x0_hat)  # Store predicted clean embedding

            # Calculate velocity from x0_hat and x_t
            v_hat = x0_hat - x_t

            # Calculate logits and probabilities based on x0_hat (the clean data prediction)
            logits = model.criterion.model.compute_logits(x0_hat)
            probs = torch.softmax(logits, dim=-1)

            # Get top-k tokens and their probabilities for positions that contribute to loss
            top_probs, top_indices = [], []
            l2_distances = []

            for pos in loss_positions:
                # Get top-k tokens and probabilities for this position
                pos_probs, pos_indices = torch.topk(probs[0, pos], k=top_k)
                top_probs.append(pos_probs.cpu().numpy())
                top_indices.append(pos_indices.cpu().numpy())

                # Calculate L2 distances between predicted clean embedding (x0_hat) and token embeddings
                current_emb = x0_hat[0, pos].unsqueeze(0)  # [1, hidden_dim]
                token_embs = word_embeddings[pos_indices]  # [top_k, hidden_dim]
                l2_dist = torch.norm(current_emb - token_embs, dim=1).cpu().numpy()
                l2_distances.append(l2_dist)

            # Store results for this timestep
            timesteps_list.append(t)
            token_probs_list.append(top_probs)
            token_ids_list.append(top_indices)
            token_texts_list.append([[tokenizer.decode([idx]) for idx in pos_indices] for pos_indices in top_indices])
            l2_distances_list.append(l2_distances)

            # Update x_t for next step (simple Euler step)
            x_t = x_t + (shortcut_size / diffusion_steps) * v_hat

    return {
        "timesteps": timesteps_list,
        "token_probs": token_probs_list,
        "token_ids": token_ids_list,
        "token_texts": token_texts_list,
        "l2_distances": l2_distances_list,
        "loss_positions": loss_positions,
        "ground_truth_tokens": ground_truth_tokens,
        "input_tokens": input_tokens,
        "input_mask": input_ids_mask[example_idx].cpu().numpy(),
        "padding_mask": padding_mask[example_idx].cpu().numpy(),
        "original_sequence": seq[0].cpu().numpy(),
        "x0_hat": x0_hat_list,  # Add predicted clean embeddings to results
        "model": model,  # Add model to results for access to embeddings
    }


def visualize_top_k_tokens(
    token_results: dict[str, Any], timestep_indices=None, figsize=(15, 20), save_path=None
) -> None:
    """
    Visualize top-k tokens for selected timesteps.

    Args:
        token_results: Results from denoise_with_token_tracking
        timestep_indices: Indices of timesteps to visualize (default: first, middle, last)
        figsize: Figure size for the plots
        save_path: Path to save the figure (optional)
    """
    timesteps = token_results["timesteps"]
    token_probs = token_results["token_probs"]
    token_texts = token_results["token_texts"]
    l2_distances = token_results["l2_distances"]
    loss_positions = token_results["loss_positions"]
    ground_truth_tokens = token_results["ground_truth_tokens"]

    # Select timesteps to visualize if not provided
    if timestep_indices is None:
        num_steps = len(timesteps)
        timestep_indices = [0, num_steps // 2, num_steps - 1]  # First, middle, last

    # Create figure with two rows (probabilities and L2 distances) for each timestep
    fig, axes = plt.subplots(len(timestep_indices), 2, figsize=figsize)

    # If only one timestep, make axes 2D
    if len(timestep_indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(timestep_indices):
        t = timesteps[idx]
        probs = token_probs[idx]
        texts = token_texts[idx]
        l2_dists = l2_distances[idx]

        # Plot token probabilities
        ax_prob = axes[i, 0]

        # Prepare data for stacked bar chart
        num_positions = len(loss_positions)
        num_tokens = len(probs[0])  # Number of tokens per position (top-k)

        # Create x positions for bars
        x = np.arange(num_positions)
        width = 0.8

        # Create position labels with ground truth tokens
        position_labels = [f"Pos {pos}\n({ground_truth_tokens[k]})" for k, pos in enumerate(loss_positions)]

        # Plot stacked bars for each position
        bottom = np.zeros(num_positions)
        for j in range(num_tokens):
            # Extract probabilities for the j-th token at each position
            token_j_probs = [pos_probs[j] for pos_probs in probs]

            # Get token texts for labels
            token_j_texts = [pos_texts[j] for pos_texts in texts]

            # Plot bar
            bars = ax_prob.bar(x, token_j_probs, width, bottom=bottom, label=f"Token {j + 1}", alpha=0.8)

            # Add token text labels to bars
            for k, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0.05:  # Only add label if bar is tall enough
                    # Add token text and probability
                    ax_prob.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[k] + height / 2,
                        f"{token_j_texts[k]}\n{token_j_probs[k]:.2f}",
                        ha="center",
                        va="center",
                        rotation=0,
                        fontsize=8,
                        color="black",
                        fontweight="bold",
                    )

            # Update bottom for next stack
            bottom += token_j_probs

        # Set title and labels
        ax_prob.set_title(f"Top-{num_tokens} Token Probabilities at Timestep {t}")
        ax_prob.set_xlabel("Token Position (Ground Truth)")
        ax_prob.set_ylabel("Probability")
        ax_prob.set_xticks(x)
        ax_prob.set_xticklabels(position_labels, rotation=90, ha="center")
        ax_prob.set_ylim(0, 1.05)  # Set y-limit to make room for labels
        ax_prob.grid(axis="y", linestyle="--", alpha=0.7)
        ax_prob.legend(loc="upper right")

        # Plot L2 distances
        ax_l2 = axes[i, 1]

        # Create a grouped bar chart for L2 distances
        bar_width = width / num_tokens
        offsets = np.linspace(-(width / 2) + (bar_width / 2), (width / 2) - (bar_width / 2), num_tokens)

        # Use a colormap for better visual distinction
        colors = plt.cm.viridis(np.linspace(0, 1, num_tokens))

        # Store token texts and L2 values for table annotation
        table_data = []

        for j in range(num_tokens):
            # Extract L2 distances for the j-th token at each position
            token_j_l2 = [pos_l2[j] for pos_l2 in l2_dists]

            # Get token texts for labels
            token_j_texts = [pos_texts[j] for pos_texts in texts]

            # Plot bar with color from colormap
            bars = ax_l2.bar(x + offsets[j], token_j_l2, bar_width, label=f"Token {j + 1}", color=colors[j], alpha=0.8)

            # Store data for table
            for k, (token, l2) in enumerate(zip(token_j_texts, token_j_l2, strict=False)):
                if len(table_data) <= k:
                    table_data.append([])
                table_data[k].append((token, l2, colors[j]))

        # Set title and labels
        ax_l2.set_title(f"L2 Distances at Timestep {t}")
        ax_l2.set_xlabel("Token Position (Ground Truth)")
        ax_l2.set_ylabel("L2 Distance")
        ax_l2.set_xticks(x)
        ax_l2.set_xticklabels(position_labels, rotation=90, ha="center")
        ax_l2.grid(axis="y", linestyle="--", alpha=0.7)

        # Add legend with better placement
        ax_l2.legend(loc="upper right")

        # Adjust y-axis to make sure all labels are visible
        y_max = max([max(pos_l2) for pos_l2 in l2_dists]) * 1.2
        ax_l2.set_ylim(0, y_max)

        # Add token text and L2 values directly on the bars
        for j in range(num_tokens):
            # Extract L2 distances for the j-th token at each position
            token_j_l2 = [pos_l2[j] for pos_l2 in l2_dists]

            # Get token texts
            token_j_texts = [pos_texts[j] for pos_texts in texts]

            # Add vertical text labels directly on each bar
            for k, (l2_val, token_text) in enumerate(zip(token_j_l2, token_j_texts, strict=False)):
                # Calculate x position (center of the bar)
                x_pos = x[k] + offsets[j]

                # Add token text and L2 value vertically on the bar
                ax_l2.text(
                    x_pos,
                    l2_val / 2,
                    f"{token_text}",
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=6,
                    color="black",
                    fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.7, pad=1),
                )

                # Add exact L2 value on top of the bar
                ax_l2.text(
                    x_pos,
                    l2_val + 0.05,
                    f"{l2_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="black",
                    fontweight="bold",
                )

    # Apply tight_layout to ensure proper spacing
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_sequence_info(token_results: dict[str, Any]) -> None:
    """
    Print information about the sequence being analyzed.

    Args:
        token_results: Results from denoise_with_token_tracking
    """
    loss_positions = token_results["loss_positions"]
    ground_truth_tokens = token_results["ground_truth_tokens"]
    _ = token_results["input_tokens"]
    original_sequence = token_results["original_sequence"]
    input_mask = token_results["input_mask"]
    padding_mask = token_results["padding_mask"]

    print("\nInput sequence (showing only non-padding tokens):")
    tokens = []
    for i, (token_id, is_padding) in enumerate(zip(original_sequence, padding_mask, strict=False)):
        if is_padding:
            is_input = "Input" if input_mask[i] == 0 else "Target"
            tokens.append(f"{i}: {token_id} ({is_input})")

    print("\n".join(tokens))

    print("\nPositions that contribute to the loss (non-input, non-padding):")
    print(loss_positions)

    print("\nGround truth tokens at loss positions:")
    for _, (pos, token) in enumerate(zip(loss_positions, ground_truth_tokens, strict=False)):
        print(f"Position {pos}: {token}")


def print_final_timestep_tokens(token_results: dict[str, Any]) -> None:
    """
    Print the top tokens at the final timestep.

    Args:
        token_results: Results from denoise_with_token_tracking
    """
    final_timestep = token_results["timesteps"][-1]
    final_token_texts = token_results["token_texts"][-1]
    final_token_probs = token_results["token_probs"][-1]
    loss_positions = token_results["loss_positions"]
    ground_truth_tokens = token_results["ground_truth_tokens"]

    print(f"\nTop tokens at the final timestep ({final_timestep}):")

    for pos_idx, pos in enumerate(loss_positions):
        print(f"Position {pos} (Ground truth: {ground_truth_tokens[pos_idx]}):")
        for token_idx, (token, prob) in enumerate(
            zip(final_token_texts[pos_idx], final_token_probs[pos_idx], strict=False)
        ):
            print(f"  {token_idx + 1}. {token} (prob: {prob:.4f})")
        print()


def analyze_token_predictions(
    model, batch, shortcut_size, top_k=5, example_idx=0, timestep_indices=None, save_path=None, figsize=(15, 20)
) -> dict[str, Any]:
    """
    Analyze token predictions during denoising.

    Args:
        model: The model to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        top_k: Number of top tokens to track at each step
        example_idx: Index of the example in the batch to track
        timestep_indices: Indices of timesteps to visualize (default: first, middle, last)
        save_path: Path to save the figure (optional)
        figsize: Figure size for the plots

    Returns:
        Results from denoise_with_token_tracking
    """
    # Run denoising with token tracking
    print("Running token tracking...")
    token_results = denoise_with_token_tracking(model, batch, shortcut_size, top_k=top_k, example_idx=example_idx)

    # Print sequence information
    print_sequence_info(token_results)

    # Select timesteps to visualize if not provided
    if timestep_indices is None:
        num_steps = len(token_results["timesteps"])
        if num_steps >= 5:
            timestep_indices = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
        else:
            timestep_indices = list(range(num_steps))

    # Visualize the selected timesteps
    visualize_top_k_tokens(token_results, timestep_indices=timestep_indices, figsize=figsize, save_path=save_path)

    # Print the top tokens at the final timestep
    print_final_timestep_tokens(token_results)

    return token_results


def get_ground_truth_token_id_and_embedding(
    tokenizer, word_embeddings, original_sequence, loss_positions, position_idx
):
    """
    Get the ground truth token id and embedding for the given position.
    """
    gt_pos = loss_positions[position_idx]
    ground_truth_id = int(original_sequence[gt_pos])
    ground_truth_emb = word_embeddings[ground_truth_id]
    return gt_pos, ground_truth_id, ground_truth_emb


def get_knn_indices_and_distances(word_embeddings, predicted_emb, ground_truth_id, k):
    """
    Get kNN indices and distances, excluding the ground truth if present.
    """
    with torch.no_grad():
        distances = torch.norm(word_embeddings - predicted_emb, dim=1)
        knn_distances, knn_indices = torch.topk(distances, k=k + 1, largest=False)
        knn_indices = knn_indices.cpu().numpy()
        knn_distances = knn_distances.cpu().numpy()
        if ground_truth_id in knn_indices:
            mask = knn_indices != ground_truth_id
            knn_distances = knn_distances[mask][:k]
            knn_indices = knn_indices[mask][:k]
        else:
            knn_distances = knn_distances[:k]
            knn_indices = knn_indices[:k]
    return knn_indices, knn_distances


def get_topk_logit_indices(model, x0_hat, gt_pos, k):
    """
    Get top-k logit indices for the given position.
    """
    with torch.no_grad():
        logits = model.criterion.model.compute_logits(x0_hat)  # [batch, seq, vocab]
        pos_logits = logits[0, gt_pos]  # [vocab]
        _, topk_logit_indices = torch.topk(pos_logits, k=k)
        topk_logit_indices = topk_logit_indices.cpu().numpy()
    return topk_logit_indices


def build_unique_embeddings_and_roles(
    predicted_emb, ground_truth_emb, ground_truth_id, word_embeddings, knn_indices, topk_logit_indices
):
    """
    Build unique embedding list, token ids, and roles for each embedding.
    """
    import collections

    unique_embeddings = []
    unique_token_ids = []  # None for predicted, int for vocab tokens
    role_map = {"predicted": None, "ground_truth": None, "knn": [], "logit": []}

    # 1. Add predicted embedding (not a vocab token)
    unique_embeddings.append(predicted_emb)
    unique_token_ids.append(None)
    idx_pred = 0
    role_map["predicted"] = idx_pred

    # 2. Add ground truth token embedding
    unique_embeddings.append(ground_truth_emb)
    unique_token_ids.append(ground_truth_id)
    idx_gt = 1
    role_map["ground_truth"] = idx_gt

    # 3. Add kNN tokens, deduplicating
    for idx in knn_indices:
        if idx not in unique_token_ids:
            unique_embeddings.append(word_embeddings[idx])
            unique_token_ids.append(idx)
            role_map["knn"].append(len(unique_embeddings) - 1)
        else:
            role_map["knn"].append(unique_token_ids.index(idx))

    # 4. Add top-k logits tokens, deduplicating
    for idx in topk_logit_indices:
        if idx not in unique_token_ids:
            unique_embeddings.append(word_embeddings[idx])
            unique_token_ids.append(idx)
            role_map["logit"].append(len(unique_embeddings) - 1)
        else:
            role_map["logit"].append(unique_token_ids.index(idx))

    # Build roles_for_index: for each unique embedding index, store a set of roles
    roles_for_index = collections.defaultdict(set)
    roles_for_index[idx_pred].add("predicted")
    roles_for_index[idx_gt].add("ground_truth")
    for idx in role_map["knn"]:
        if idx != idx_pred:
            roles_for_index[idx].add("knn")
    for idx in role_map["logit"]:
        if idx != idx_pred:
            roles_for_index[idx].add("logit")

    return unique_embeddings, unique_token_ids, roles_for_index, idx_pred


def run_tsne(unique_embeddings, figsize, random_state=42):
    import torch
    from sklearn.manifold import TSNE

    embeddings_to_plot = torch.stack(unique_embeddings, dim=0)
    n_points = embeddings_to_plot.shape[0]
    perplexity = min(n_points - 1, 5)
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings_to_plot.cpu().numpy())
    return embeddings_2d


def plot_embeddings_and_edges(
    embeddings_2d,
    unique_embeddings,
    unique_token_ids,
    roles_for_index,
    idx_pred,
    word_embeddings,
    tokenizer,
    gt_pos,
    timesteps,
    timestep_idx,
    save_path=None,
    figsize=(12, 10),
):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.figure(figsize=figsize)
    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=14, label="Predicted"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="green", markersize=14, label="Ground Truth"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="kNN"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="orange", markersize=10, label="Top-k Logits"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="purple", markersize=10, label="kNN & Logit"),
    ]
    # Plot predicted embedding
    plt.scatter(embeddings_2d[idx_pred, 0], embeddings_2d[idx_pred, 1], c="red", s=200, marker="*", zorder=10)
    plt.text(
        embeddings_2d[idx_pred, 0],
        embeddings_2d[idx_pred, 1],
        "Predicted",
        fontsize=10,
        ha="center",
        va="bottom",
        color="red",
    )
    for idx in range(1, len(unique_embeddings)):
        roles = roles_for_index[idx]
        token_id = unique_token_ids[idx]
        if token_id is None:
            continue
        if "ground_truth" in roles:
            color = "green"
            marker = "*"
            size = 200
        elif "knn" in roles and "logit" in roles:
            color = "purple"
            marker = "D"
            size = 120
        elif "knn" in roles:
            color = "blue"
            marker = "o"
            size = 100
        elif "logit" in roles:
            color = "orange"
            marker = "s"
            size = 100
        else:
            color = "gray"
            marker = "x"
            size = 80
        role_str = ",".join(sorted(roles))
        token = tokenizer.decode([token_id])
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], c=color, s=size, alpha=0.7, marker=marker, zorder=5)
        plt.text(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            f"{token}\n({role_str})",
            fontsize=10,
            ha="center",
            va="bottom",
            color=color,
        )
    # Draw edges from predicted to each unique point (except predicted itself)
    for idx in range(1, len(unique_embeddings)):
        roles = roles_for_index[idx]
        token_id = unique_token_ids[idx]
        if token_id is None:
            continue
        if "ground_truth" in roles:
            edge_color = "g"
            edge_style = "-"
        elif "knn" in roles and "logit" in roles:
            edge_color = "purple"
            edge_style = "-."
        elif "knn" in roles:
            edge_color = "b"
            edge_style = "--"
        elif "logit" in roles:
            edge_color = "orange"
            edge_style = ":"
        else:
            edge_color = "gray"
            edge_style = "-."
        distance = torch.norm(unique_embeddings[idx_pred] - word_embeddings[token_id]).item()
        plt.plot(
            [embeddings_2d[idx_pred, 0], embeddings_2d[idx, 0]],
            [embeddings_2d[idx_pred, 1], embeddings_2d[idx, 1]],
            color=edge_color,
            alpha=0.5,
            linestyle=edge_style,
            linewidth=2,
        )
        mid_x = (embeddings_2d[idx_pred, 0] + embeddings_2d[idx, 0]) / 2
        mid_y = (embeddings_2d[idx_pred, 1] + embeddings_2d[idx, 1]) / 2
        plt.text(
            mid_x,
            mid_y,
            f"{distance:.2f}",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.7, pad=1),
            color=edge_color,
        )
    plt.title(f"KNN vs Top-k Logits at Timestep {timesteps[timestep_idx]}, Position {gt_pos}")
    plt.legend(handles=legend_handles)
    plt.grid(True, alpha=0.3)
    if save_path:
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def visualize_knn_embeddings(
    token_results: dict[str, Any],
    timestep_idx: int,
    position_idx: int,
    k: int = 5,
    figsize: tuple[int, int] = (12, 10),
    save_path: str | None = None,
) -> None:
    """
    Visualize k-nearest neighbors of token embeddings in 2D space using t-SNE, and compare to top-k tokens by logits.
    Ensures each unique token embedding is only plotted once, even if it appears in multiple roles.
    Always marks and connects the ground truth token.
    """
    model = token_results["model"]
    tokenizer = model.criterion.flow_matching_criterion.tokenizer
    word_embeddings = model.criterion.model.module.word_embedding.weight
    timesteps = token_results["timesteps"]
    loss_positions = token_results["loss_positions"]
    original_sequence = token_results["original_sequence"]
    x0_hat = token_results["x0_hat"][timestep_idx]
    predicted_emb = x0_hat[0, loss_positions[position_idx]]
    gt_pos, ground_truth_id, ground_truth_emb = get_ground_truth_token_id_and_embedding(
        tokenizer, word_embeddings, original_sequence, loss_positions, position_idx
    )
    knn_indices, knn_distances = get_knn_indices_and_distances(word_embeddings, predicted_emb, ground_truth_id, k)
    topk_logit_indices = get_topk_logit_indices(model, x0_hat, gt_pos, k)
    unique_embeddings, unique_token_ids, roles_for_index, idx_pred = build_unique_embeddings_and_roles(
        predicted_emb, ground_truth_emb, ground_truth_id, word_embeddings, knn_indices, topk_logit_indices
    )
    embeddings_2d = run_tsne(unique_embeddings, figsize)
    plot_embeddings_and_edges(
        embeddings_2d,
        unique_embeddings,
        unique_token_ids,
        roles_for_index,
        idx_pred,
        word_embeddings,
        tokenizer,
        gt_pos,
        timesteps,
        timestep_idx,
        save_path=save_path,
        figsize=figsize,
    )


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    sys.path.insert(0, project_root)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze token predictions during denoising")
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
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tokens to track")
    parser.add_argument("--example_idx", type=int, default=0, help="Index of the example to analyze")
    parser.add_argument("--output_dir", type=str, default="reports/figures", help="Directory to save figures")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for data loading")
    parser.add_argument("--timestep_idx", type=int, default=-1, help="Timestep index for KNN visualization")
    parser.add_argument("--position_idx", type=int, default=0, help="Position index for KNN visualization")
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

    # Create a small batch for analysis
    test_dataloader = DataLoader(
        test_text_ds,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=False,
    )

    # Get a single batch for analysis
    test_batch = next(iter(test_dataloader))
    print(f"Loaded test batch with {len(test_batch.seqs)} examples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis
    output_path = os.path.join(args.output_dir, f"token_analysis_example_{args.example_idx}.png")
    token_results = analyze_token_predictions(
        unit, test_batch, args.shortcut_size, top_k=args.top_k, example_idx=args.example_idx, save_path=output_path
    )

    print(f"Analysis complete. Figure saved to {output_path}")

    # Visualize KNN embeddings
    if args.timestep_idx >= 0:
        knn_output_path = os.path.join(args.output_dir, f"knn_visualization_example_{args.example_idx}.png")
        visualize_knn_embeddings(
            token_results,
            timestep_idx=args.timestep_idx,
            position_idx=args.position_idx,
            k=args.top_k,
            save_path=knn_output_path,
        )
        print(f"KNN visualization saved to {knn_output_path}")
