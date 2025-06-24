import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Dataset
from scripts.evaluate_seq import process_sequence

sys.path.append("../shortcutfm")

from shortcutfm.analysis.denoising import denoise_with_tracking, denoise_with_velocity_tracking
from shortcutfm.batch import collate
from shortcutfm.config import TrainingConfig
from shortcutfm.criteria import CompositeCriterion
from shortcutfm.text_datasets import TextDataset
from shortcutfm.train.pl.train_unit import TrainModule, _extract_clean_predicted_text
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
    step_size: int = 1,
    use_ground_truth_interpolation: bool = False,
    velocity_scale: float | str = 1.0,
    noise_std_dev: float = 1.0,
    force_recompute: bool = False,
) -> dict[str, Any]:
    """Analyze a single checkpoint."""
    # check if json file with existing results exists
    results_file = checkpoint_dir / "analysis_results.json"
    if results_file.exists() and not force_recompute:
        # load existing results_file
        import json

        with open(results_file) as f:
            existing_results = json.load(f)
            # Ensure loaded results have the correct structure for plotting
            if "results" in existing_results:
                plot_results = existing_results["results"]
        return existing_results

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
    velocity_results = denoise_with_velocity_tracking(
        unit.criterion.flow_matching_criterion,
        batch,
        shortcut_size=shortcut_size,
        per_token_cosine=True,
        step_size=step_size,
        use_ground_truth_interpolation=use_ground_truth_interpolation,
        velocity_scale=velocity_scale,
        noise_std_dev=noise_std_dev,
    )

    # Extract config parameters
    params = extract_config_params(config, checkpoint_dir)

    bleu_results = {}
    for step_size in (
        2048,
        1024,
        512,
        256,
    ):
        for gt_interpolation in (False, True):
            # compute BLEU for each step size and add to results
            bleu_results.setdefault(gt_interpolation, {})
            bleu_result = compute_bleu(
                unit,
                dataloader,
                shortcut_size=step_size,
                step_size=step_size,
                use_ground_truth_interpolation=gt_interpolation,
                velocity_scale=velocity_scale,
                noise_std_dev=noise_std_dev,
            )
            bleu_results[gt_interpolation][step_size] = bleu_result["bleu"]

    plot_results = {
        "timesteps": [t for t in velocity_results["timesteps"]],
        "cosine_similarities": [cs.mean().item() for cs in velocity_results["cosine_similarities"]],
        "predicted_velocity_norms": [n.mean().item() for n in velocity_results["predicted_velocity_norms"]],
        "ground_truth_velocity_norms": [n.mean().item() for n in velocity_results["ground_truth_velocity_norms"]],
    }

    results = {
        "results": plot_results,
        "bleu_results": bleu_results,
        "params": params,
        "legend": create_abbreviated_legend(params),
    }

    # Save results to file
    def tensor_to_python(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_python(v) for v in obj]
        else:
            return obj

    with open(results_file, "w") as f:
        import json

        json.dump(tensor_to_python(results), f, indent=4)

    return results


def probe(self, hidden_representation, return_logits: bool = False) -> Tensor:
    """Predicts sequence of tokens based on hidden_representation.

    :param hidden_representation: Hidden representation from the model
    :type hidden_representation: Tensor
    :param return_logits: Whether to return logits instead of token IDs
    :type return_logits: bool

    :return: Either logits or token IDs
    :rtype: Tensor
    """
    logits = self.model.compute_logits(hidden_representation)
    if return_logits:
        return logits
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.argmax(probs, dim=-1)
    return tokens


def compute_bleu(
    unit: TrainModule,
    dataloader: DataLoader,
    shortcut_size: int = 1024,
    step_size: int = 1,
    use_ground_truth_interpolation: bool = False,
    velocity_scale: float | str = 1.0,
    noise_std_dev: float = 1.0,
    limit_test_batches: int | None = None,
) -> dict[str, Any]:
    """Compute BLEU scores for the given dataloader."""
    limit_test_batches = 16

    # Load the dataset
    test_dataloader = dataloader
    if limit_test_batches is not None:
        test_dataloader = islice(test_dataloader, limit_test_batches)

    tokenizer = unit.criterion.flow_matching_criterion.tokenizer
    inputs = []
    predictions = []

    denoising_step_size = step_size if step_size is not None else unit.denoising_step_size

    total_batches = len(test_dataloader) if not isinstance(test_dataloader, islice) else limit_test_batches
    for _, test_batch in enumerate(tqdm(test_dataloader, desc="Evaluating", total=total_batches)):
        test_batch = test_batch.to(unit.device)
        model_hidden_outputs = denoise_with_tracking(
            unit.criterion.flow_matching_criterion,
            test_batch,
            shortcut_size=shortcut_size,
            step_size=denoising_step_size,
            use_ground_truth_interpolation=use_ground_truth_interpolation,
            velocity_scale=velocity_scale,
            std_dev=noise_std_dev,
        )["model_outputs"][-1]  # Get the last step outputs
        predicted_ids = probe(unit.criterion.flow_matching_criterion, model_hidden_outputs, return_logits=False)

        inputs.append(test_batch.seqs.detach().cpu())
        predictions.append(predicted_ids.detach().cpu())

    all_inputs = torch.cat(inputs, dim=0)
    all_predictions = torch.cat(predictions, dim=0)

    input_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in all_inputs]
    sources, references = zip(*[process_sequence(text, tokenizer) for text in input_texts], strict=False)
    references = list(references)  # Convert tuple to list

    # Process predictions (take last step predictions)
    pred_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in all_predictions]
    hypotheses = _extract_clean_predicted_text(pred_texts)

    # BLEU score
    bleu = evaluate.load("bleu")
    print(
        f"Computing BLEU for {len(hypotheses)} examples with shortcut size {shortcut_size} and step size {step_size}"
        f"using {'ground truth' if use_ground_truth_interpolation else 'predicted'} interpolation."
    )
    try:
        bleu_score: dict | None = bleu.compute(predictions=hypotheses, references=[[ref] for ref in references])
        print(f"BLEU score: {bleu_score['bleu']:.4f}")  # type: ignore
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        bleu_score = {}
        bleu_score["bleu"] = 0.0  # Fallback to zero if BLEU computation fails

    return {
        "bleu": bleu_score["bleu"],  # type: ignore
        "predictions": hypotheses,
        "references": references,
        "inputs": sources,
    }


def bleu_results_to_df(analyses, gt_interpolation=True):
    """
    Convert BLEU results from analyses into a DataFrame suitable for plotting.

    Args:
        analyses (list): List of analysis dicts.
        gt_interpolation (bool): Whether to use ground truth interpolation key.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestep', 'bleu', 'checkpoint', 'step_size'].
    """
    rows = []
    key = str(gt_interpolation).lower()
    for analysis in analyses:
        legend = analysis["legend"]
        params = analysis.get("params", {})
        bleu_results = analysis["bleu_results"].get(key, {})
        if isinstance(bleu_results, dict):
            for timestep, bleu in bleu_results.items():
                rows.append(
                    {
                        "timestep": int(timestep) if timestep is not None else None,
                        "bleu": bleu,
                        "checkpoint": legend,
                        "step_size": int(timestep) if timestep is not None else None,
                    }
                )
        elif isinstance(bleu_results, float) or isinstance(bleu_results, int):
            rows.append(
                {
                    "timestep": None,
                    "bleu": bleu_results,
                    "checkpoint": legend,
                    "step_size": params.get("step_size", None),
                }
            )
    return pd.DataFrame(rows)


def plot_bleu_bar(df, output_path, title):
    if df.empty:
        print(f"Warning: DataFrame is empty, skipping plot for {title}")
        return
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="checkpoint", y="bleu", hue="step_size", errorbar=None, dodge=True)
    plt.title(title)
    plt.xlabel("Checkpoint")
    plt.ylabel("BLEU Score")
    plt.legend(title="Generation Steps", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
        cosine_similarities = [cs for cs in results["cosine_similarities"]]
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
        pred_norms = [n for n in results["predicted_velocity_norms"]]
        gt_norms = [n for n in results["ground_truth_velocity_norms"]]
        plt.plot(timesteps, pred_norms, marker=".", linestyle="-", label=f"{analysis['legend']} (pred)")
        plt.plot(timesteps, gt_norms, marker=".", linestyle="--", label=f"{analysis['legend']} (gt)")

    plt.title("Velocity Norm Comparison Across Models")
    plt.xlabel("Timestep")
    plt.ylabel("Velocity Norm")
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_norm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    for gt_interpolation in (False, True):
        df = bleu_results_to_df(analyses, gt_interpolation=gt_interpolation)
        if df.empty:
            print(f"Warning: No BLEU results for gt_interpolation={gt_interpolation}, skipping plot.")
            continue
        plot_bleu_bar(
            df,
            output_dir / f"bleu_results_{'gt' if gt_interpolation else 'non_gt'}_interpolation.png",
            title=f"BLEU Scores {'with' if gt_interpolation else 'without'} GT Interpolation",
        )


def main():
    # Configuration

    run_dirs = os.listdir("checkpoints")
    excluded_dirs = ["run_jwpwmptz", "run_wkkbo3i3"]

    # include dirs shoulf take precedence over run_dirs
    include_dirs = ["run_u0ompcg8", "run_k4180ptk"]
    if include_dirs:
        run_dirs = include_dirs
    else:
        run_dirs = [dir for dir in run_dirs if "run_" in dir and dir not in excluded_dirs]

    checkpoint_dirs = [Path(f"checkpoints/{dir}") for dir in run_dirs]
    test_data_path = "datasets/tokenized/bert-base-uncased/QQP-Official/valid"
    output_dir = Path("results/analysis_results")
    batch_size = 32
    shortcut_size = 512

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
            step_size=shortcut_size,
            use_ground_truth_interpolation=False,
            velocity_scale=1.0,
            noise_std_dev=1.0,
            force_recompute=False,
        )
        analyses.append(analysis)

    # Create comparison plots
    plot_comparison(analyses, output_dir)


if __name__ == "__main__":
    main()
