"""Shared denoising functionality for analysis.

This module provides functions for denoising with various tracking capabilities,
combining features from velocity and token analysis while maintaining a clean API.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from shortcutfm.batch import EncoderBatch
from shortcutfm.criteria import FlowMatchingCriterion


def denoise_with_tracking(
    criterion: FlowMatchingCriterion,
    batch: EncoderBatch,
    shortcut_size: int | None = None,
    step_size: int | None = None,
    guidance_scale: float | None = None,
    tracking_fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], dict[str, Any]] | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_ground_truth_interpolation: bool = False,
) -> dict[str, Any]:
    """
    Perform denoising while tracking model predictions and optionally applying custom tracking.

    Args:
        criterion: The criterion to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        step_size: The step size to use for denoising when shortcut_size is None or 0
        guidance_scale: The guidance scale to use for classifier-free guidance
        tracking_fn: Optional function to track additional metrics during denoising
        device: The device to use for computation
        use_ground_truth_interpolation: If True, use ground truth interpolation between original
                                        embedding and noise based on timestep t instead of model prediction

    Returns:
        A dictionary containing:
        - timesteps: Timesteps used during denoising
        - model_outputs: Model's predicted clean embeddings at each step
        - ground_truth_embeddings: Ground truth embeddings
        - tracking_results: Results from tracking_fn if provided
    """
    if shortcut_size is None and step_size is None:
        raise ValueError("Either shortcut_size or step_size must be provided")
    if (shortcut_size == 0 or shortcut_size is None) and step_size is None:
        raise ValueError("step_size must be provided when shortcut_size is 0 or None")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion.model.eval()
    diffusion_steps: int = criterion.model.diffusion_steps

    # Move individual tensors in batch to device
    seqs = batch.seqs.to(device)
    input_ids_mask = batch.input_ids_mask.to(device)
    padding_mask = batch.padding_mask.to(device)

    # Initialize tracking variables
    timesteps_list = []
    model_outputs = []
    tracking_results = [] if tracking_fn else None

    with torch.no_grad():
        # Get input mask and embeddings
        input_mask: Tensor = input_ids_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        embeddings = criterion.model.get_embeddings(seqs)  # [batch_size, seq_len, hidden_dim]

        # Initialize with noise where mask is 0
        noise = torch.randn_like(embeddings)
        x_t = torch.where(input_mask == 0, embeddings, noise)

        # Store the original embeddings as ground truth x0
        x0_ground_truth = embeddings.clone()

        # Use step_size if shortcut_size is None or 0
        effective_step = step_size or shortcut_size
        shortcut_size = shortcut_size or 0

        # Denoising loop
        shortcuts = torch.tensor(shortcut_size, device=device).repeat(input_mask.shape[0])

        for t in torch.arange(diffusion_steps, 0, -effective_step, device=device):
            t_batch = t.repeat(input_mask.shape[0])
            timesteps_list.append(t.item())

            # Get model prediction with optional guidance
            model_output = criterion.infere_model(x_t, t_batch, shortcuts, input_mask, guidance_scale=guidance_scale)

            # Store model output
            model_outputs.append(model_output.clone())

            # Apply custom tracking if provided
            if tracking_fn:
                tracking_result = tracking_fn(
                    model_output, x_t, x0_ground_truth, input_mask, padding_mask.unsqueeze(-1)
                )
                tracking_results.append(tracking_result)

            # Update x_t for next step
            if use_ground_truth_interpolation:
                # Use ground truth interpolation based on timestep t
                # Calculate the next timestep
                next_t = max(t.item() - effective_step, 0)
                # Interpolate between ground truth and noise based on next timestep
                next_t_scaled = next_t / diffusion_steps
                # Apply the interpolation formula: x0 + (noise - x0) * t
                x0_hat = torch.where(
                    input_mask == 0,
                    x0_ground_truth,  # Keep original embeddings for input tokens
                    x0_ground_truth + (noise - x0_ground_truth) * next_t_scaled,  # Interpolate for masked tokens
                )
            else:
                # Use model prediction (original behavior)
                v_hat = criterion.compute_velocity(model_output=model_output, noise=noise, input_mask=input_mask)
                x0_hat = x_t + (effective_step / diffusion_steps) * v_hat

            x_t = x0_hat

    results = {
        "timesteps": timesteps_list,
        "model_outputs": model_outputs,
        "ground_truth_embeddings": x0_ground_truth,
    }

    if tracking_results:
        results["tracking_results"] = tracking_results

    return results


def denoise_with_velocity_tracking(
    criterion: FlowMatchingCriterion,
    batch: EncoderBatch,
    shortcut_size: int | None = None,
    step_size: int | None = None,
    guidance_scale: float | None = None,
    per_token_cosine: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_ground_truth_interpolation: bool = False,
) -> dict[str, Any]:
    """
    Perform denoising while tracking velocity predictions.

    Args:
        criterion: The criterion to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        step_size: The step size to use for denoising when shortcut_size is None or 0
        guidance_scale: The guidance scale to use for classifier-free guidance
        per_token_cosine: If True, calculate cosine similarity for each token separately
        device: The device to use for computation
        use_ground_truth_interpolation: If True, use ground truth interpolation between original
                                        embedding and noise based on timestep t instead of model prediction

    Returns:
        A dictionary containing velocity tracking results
    """
    from shortcutfm.analysis.velocity_analysis import calculate_batch_cosine_similarity

    def velocity_tracking_fn(
        model_output: Tensor,
        x_t: Tensor,
        x0_ground_truth: Tensor,
        input_mask: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Any]:
        # Calculate predicted velocity
        v_hat = model_output - x_t
        v_hat = torch.where(input_mask == 0, torch.zeros_like(v_hat), v_hat)

        # Calculate ground truth velocity
        v_ground_truth = x0_ground_truth - x_t

        # Calculate cosine similarity
        cos_sim, pred_norm, gt_norm = calculate_batch_cosine_similarity(
            v_hat, v_ground_truth, input_mask, padding_mask, per_token=per_token_cosine
        )

        # Calculate L2 distances
        valid_token_mask = (input_mask.squeeze(-1) == 1) & (padding_mask.squeeze(-1) == 1)
        l2_dists = torch.norm(model_output[valid_token_mask] - x0_ground_truth[valid_token_mask], dim=-1).mean()
        velocity_l2_dists = torch.norm(v_hat[valid_token_mask] - v_ground_truth[valid_token_mask], dim=-1).mean()

        return {
            "predicted_velocity": v_hat,
            "ground_truth_velocity": v_ground_truth,
            "cosine_similarity": cos_sim,
            "predicted_velocity_norm": pred_norm,
            "ground_truth_velocity_norm": gt_norm,
            "l2_distance": l2_dists,
            "velocity_l2_distance": velocity_l2_dists,
        }

    results = denoise_with_tracking(
        criterion,
        batch,
        shortcut_size=shortcut_size,
        step_size=step_size,
        guidance_scale=guidance_scale,
        tracking_fn=velocity_tracking_fn,
        device=device,
        use_ground_truth_interpolation=use_ground_truth_interpolation,
    )

    # Extract and organize tracking results
    tracking_results = results.pop("tracking_results")
    results.update(
        {
            "predicted_velocities": [r["predicted_velocity"] for r in tracking_results],
            "ground_truth_velocities": [r["ground_truth_velocity"] for r in tracking_results],
            "cosine_similarities": [r["cosine_similarity"] for r in tracking_results],
            "predicted_velocity_norms": [r["predicted_velocity_norm"] for r in tracking_results],
            "ground_truth_velocity_norms": [r["ground_truth_velocity_norm"] for r in tracking_results],
            "l2_distances": [r["l2_distance"] for r in tracking_results],
            "velocity_l2_distances": [r["velocity_l2_distance"] for r in tracking_results],
        }
    )

    return results


def denoise_with_token_tracking(
    criterion: FlowMatchingCriterion,
    batch: EncoderBatch,
    shortcut_size: int | None = None,
    step_size: int | None = None,
    guidance_scale: float | None = None,
    top_k: int = 5,
    example_idx: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_ground_truth_interpolation: bool = False,
) -> dict[str, Any]:
    """
    Perform denoising while tracking token probabilities and L2 distances.

    Args:
        criterion: The criterion to use for denoising
        batch: The batch to denoise
        shortcut_size: The shortcut size to use for denoising
        step_size: The step size to use for denoising when shortcut_size is None or 0
        guidance_scale: The guidance scale to use for classifier-free guidance
        top_k: Number of top tokens to track at each step
        example_idx: Index of the example in the batch to track
        device: The device to use for computation
        use_ground_truth_interpolation: If True, use ground truth interpolation between original
                                        embedding and noise based on timestep t instead of model prediction

    Returns:
        A dictionary containing token tracking results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = criterion.tokenizer
    word_embeddings = criterion.model.module.word_embedding.weight

    # Select a single example to track
    seq = batch.seqs[example_idx : example_idx + 1].to(device)
    input_mask = batch.input_ids_mask[example_idx : example_idx + 1].to(device).unsqueeze(-1)
    pad_mask = batch.padding_mask[example_idx : example_idx + 1].to(device).unsqueeze(-1)

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

    def token_tracking_fn(
        model_output: Tensor,
        x_t: Tensor,
        x0_ground_truth: Tensor,
        input_mask: Tensor,
        padding_mask: Tensor,
    ) -> dict[str, Any]:
        # Calculate logits and probabilities
        logits = criterion.model.compute_logits(model_output)
        probs = torch.softmax(logits, dim=-1)

        # Get top-k tokens and their probabilities for positions that contribute to loss
        top_probs, top_indices = [], []
        l2_distances = []

        for pos in loss_positions:
            # Get top-k tokens and probabilities for this position
            pos_probs, pos_indices = torch.topk(probs[0, pos], k=top_k)
            top_probs.append(pos_probs.cpu().numpy())
            top_indices.append(pos_indices.cpu().numpy())

            # Calculate L2 distances between predicted clean embedding and token embeddings
            current_emb = model_output[0, pos].unsqueeze(0)  # [1, hidden_dim]
            token_embs = word_embeddings[pos_indices]  # [top_k, hidden_dim]
            l2_dist = torch.norm(current_emb - token_embs, dim=1).cpu().numpy()
            l2_distances.append(l2_dist)

        # Get token texts
        token_texts = [[tokenizer.decode([idx]) for idx in pos_indices] for pos_indices in top_indices]

        return {
            "token_probs": top_probs,
            "token_ids": top_indices,
            "token_texts": token_texts,
            "l2_distances": l2_distances,
        }

    results = denoise_with_tracking(
        criterion,
        batch,
        shortcut_size=shortcut_size,
        step_size=step_size,
        guidance_scale=guidance_scale,
        tracking_fn=token_tracking_fn,
        device=device,
        use_ground_truth_interpolation=use_ground_truth_interpolation,
    )

    # Extract and organize tracking results
    tracking_results = results.pop("tracking_results")
    results.update(
        {
            "token_probs": [r["token_probs"] for r in tracking_results],
            "token_ids": [r["token_ids"] for r in tracking_results],
            "token_texts": [r["token_texts"] for r in tracking_results],
            "l2_distances": [r["l2_distances"] for r in tracking_results],
            "loss_positions": loss_positions,
            "ground_truth_tokens": ground_truth_tokens,
            "input_tokens": input_tokens,
            "input_mask": batch.input_ids_mask[example_idx].cpu().numpy(),
            "padding_mask": batch.padding_mask[example_idx].cpu().numpy(),
            "original_sequence": seq[0].cpu().numpy(),
        }
    )

    return results
