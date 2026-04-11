"""
Metrics analysis module for generation evaluation.

This module provides comprehensive functionality for loading, processing, and visualizing
metrics from different experimental runs across datasets (QQP, WebNLG, WMT19).
"""

import colorsys
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_plot_style() -> None:
    """Configure matplotlib and seaborn styling for better plots."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

# Metric display name mapping
METRIC_DISPLAY_NAMES = {
    "bleu_bleu": "BLEU",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
    "rouge_rouge-l": "ROUGE-L",
    "bertscore_precision": "BERTScore Precision",
    "bertscore_recall": "BERTScore Recall",
    "bertscore_f1": "BERTScore",
    "distinct_1": "Distinct-1",
    "distinct_2": "Distinct-2",
    "distinct_3": "Distinct-3",
    "distinct_4": "Distinct-4",
    "meteor_meteor": "METEOR",
    "bert_score_bert_score": "BERTScore",
}


def clean_metric_name(metric_name: str) -> str:
    """
    Convert raw metric names to clean display names.

    Args:
        metric_name: Raw metric name (e.g., 'bleu_bleu')

    Returns:
        Clean display name (e.g., 'BLEU')
    """
    if not metric_name:
        return ""

    # Check exact match first
    if metric_name in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric_name]

    # For unknown metrics, try to make them look nicer
    # Convert underscores to spaces and title case
    cleaned = metric_name.replace("_", " ").title()
    return cleaned


def calculate_nfe(shortcut_size: int) -> float:
    """
    Calculate Number of Function Evaluations (NFE) from shortcut size.

    Args:
        shortcut_size: The shortcut size value

    Returns:
        NFE value (2048 / shortcut_size)
    """
    return 2048.0 / shortcut_size


def flatten_dictionary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten dictionary columns into separate columns.

    For single key dictionaries like {'bleu': 0.5}, creates column 'bleu_bleu'.
    For multiple key dictionaries like {'precision': 0.5, 'recall': 0.6},
    creates columns 'original_precision', 'original_recall'.

    Args:
        df: DataFrame with potential dictionary columns

    Returns:
        DataFrame with flattened columns
    """
    df_flattened = df.copy()
    columns_to_drop = []

    for col in df.columns:
        if df[col].dtype == "object":
            # Check if any values are dictionaries
            sample_vals = df[col].dropna()
            if not sample_vals.empty and isinstance(sample_vals.iloc[0], dict):
                # Extract all possible keys from all dictionaries in this column
                all_keys = set()
                for val in sample_vals:
                    if isinstance(val, dict):
                        all_keys.update(val.keys())

                # Create new columns first
                if len(all_keys) == 1:
                    # Single key: use column_key format to avoid naming conflicts
                    key = list(all_keys)[0]
                    new_col_name = f"{col}_{key}"
                    df_flattened[new_col_name] = df[col].apply(
                        lambda x, k=key: x.get(k) if isinstance(x, dict) else None
                    )
                else:
                    # Multiple keys: use original_column_key format
                    for key in all_keys:
                        new_col_name = f"{col}_{key}"
                        df_flattened[new_col_name] = df[col].apply(
                            lambda x, k=key: x.get(k) if isinstance(x, dict) else None
                        )

                # Mark original column for dropping
                columns_to_drop.append(col)

    # Drop all original dictionary columns at once
    if columns_to_drop:
        df_flattened = df_flattened.drop(columns=columns_to_drop)

    return df_flattened


def add_nfe_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NFE column to DataFrame based on shortcut_size.

    Args:
        df: DataFrame with shortcut_size column

    Returns:
        DataFrame with added nfe column
    """
    df_copy = df.copy()
    df_copy["nfe"] = df_copy["shortcut_size"].apply(calculate_nfe)
    return df_copy


def parse_path_metadata(file_path: str) -> dict[str, str | int]:
    """
    Parse metadata from file path structure.

    Expected structure:
    generation_outputs/{dataset}/{experiment_type}/{run_id}/step={step}/scut={size}/seed_{n}/metrics_file

    Args:
        file_path: Path to metrics file

    Returns:
        Dictionary with extracted metadata

    Raises:
        ValueError: If path doesn't match expected structure
    """
    path_parts = Path(file_path).parts

    # Find generation_outputs index
    try:
        gen_outputs_idx = path_parts.index("generation_outputs")
    except ValueError as exc:
        raise ValueError(f"Path must contain 'generation_outputs': {file_path}") from exc

    # Extract components after generation_outputs
    remaining_parts = path_parts[gen_outputs_idx + 1 :]

    if len(remaining_parts) < 4:
        raise ValueError(f"Invalid path structure: {file_path}")

    dataset = remaining_parts[0]
    experiment_type = remaining_parts[1]
    run_id = remaining_parts[2]

    # Find step and shortcut size
    step_info = None
    shortcut_size = None
    seed_info = None
    metrics_file = remaining_parts[-1]

    # Look for step=X and scut=X patterns in the path
    for part in remaining_parts[3:]:
        if part.startswith("step="):
            step_info = part
        elif part.startswith("scut="):
            shortcut_size = int(part.split("=")[1])
        elif part.startswith("seed_"):
            seed_info = part

    if shortcut_size is None:
        raise ValueError(f"Could not find shortcut size in path: {file_path}")

    return {
        "dataset": dataset,
        "experiment_type": experiment_type,
        "run_id": run_id,
        "step": step_info or "unknown",
        "shortcut_size": shortcut_size,
        "seed": seed_info or "unknown",
        "metrics_file": metrics_file,
    }


def extract_step_info(step_str: str) -> tuple[str, str, int]:
    """
    Extract step information from step string.

    Args:
        step_str: Step string like 'step=40000' or 'step=final'

    Returns:
        Tuple of (step_type, step_display, step_number)
    """
    if not step_str or step_str == "unknown":
        return "unknown", "No Step Info", 0

    if step_str == "step=final":
        return "final", "Final Checkpoint", 0

    # Extract number from step=XXXX
    match = re.search(r"step=(\d+)", step_str)
    if match:
        step_num = int(match.group(1))
        return "training", f"Step {step_num}", step_num

    return "unknown", step_str, 0


def find_metrics_files(base_path: str, metrics_filename: str = "metrics_nltk_fallback_test.json") -> list[str]:
    """
    Find all metrics files recursively in the base path.

    Args:
        base_path: Base directory to search
        metrics_filename: Name of the metrics file to find

    Returns:
        List of file paths
    """
    base_dir = Path(base_path)
    if not base_dir.exists():
        return []

    # Search recursively for metrics files
    metrics_files = list(base_dir.rglob(metrics_filename))
    return [str(f) for f in metrics_files]


def load_metrics_file(file_path: str) -> dict:
    """
    Load a single metrics JSON file.

    Args:
        file_path: Path to the metrics file

    Returns:
        Dictionary with metrics data
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return {}


def load_all_metrics_with_metadata(base_path: str, metrics_filename: str = "metrics_nltk_fallback_test.json") -> pd.DataFrame:
    """
    Load all metrics files with metadata extraction.

    Args:
        base_path: Base directory containing metrics files

    Returns:
        DataFrame with all metrics and metadata
    """
    metrics_files = find_metrics_files(base_path, metrics_filename)

    if not metrics_files:
        print(f"No metrics files found in {base_path}")
        return pd.DataFrame()

    all_data = []

    for file_path in metrics_files:
        try:
            # Parse metadata from path
            metadata = parse_path_metadata(file_path)

            # Load metrics data
            metrics_data = load_metrics_file(file_path)

            if not metrics_data:
                continue

            # Combine metadata and metrics
            combined_data = {**metadata, **metrics_data}
            combined_data["file_path"] = file_path

            all_data.append(combined_data)

        except Exception as e:
            print(f"Warning: Error processing {file_path}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # Flatten dictionary-valued columns (like bleu, bertscore)
    df = flatten_dictionary_columns(df)

    # Add derived columns
    df = add_nfe_column(df)

    # Add step information
    if "step" in df.columns:
        step_info = df["step"].apply(extract_step_info)
        df["step_type"] = [info[0] for info in step_info]
        df["step_display"] = [info[1] for info in step_info]
        df["step_number"] = [info[2] for info in step_info]

    return df


def load_dataset_metrics(dataset_name: str, base_dir: str = "../generation_outputs", metrics_filename: str = "metrics_nltk_fallback_test.json") -> pd.DataFrame:
    """
    Load and process metrics data for a specific dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'qqp', 'webnlg', 'wmt19', 'parasci')
        base_dir: Base directory containing generation outputs
        metrics_filename: Name of the metrics file to search for (e.g., 'metrics.json', 'metrics_nltk_fallback.json')

    Returns:
        DataFrame with loaded metrics and metadata
    """
    dataset_path = Path(base_dir) / dataset_name

    if not dataset_path.exists():
        print(f"Dataset path not found: {dataset_path}")
        return pd.DataFrame()

    print(f"Loading metrics from: {dataset_path}")
    df = load_all_metrics_with_metadata(str(dataset_path), metrics_filename)

    # if not df.empty:
        # print(f"Loaded {len(df)} metrics files for {dataset_name}")

    return df


def get_metric_columns(df: pd.DataFrame) -> list[str]:
    """
    Get list of metric columns from DataFrame.

    Args:
        df: DataFrame with metrics

    Returns:
        List of metric column names
    """
    # Common metadata columns to exclude
    metadata_cols = {
        "file_path",
        "dataset",
        "experiment_type",
        "run_id",
        "step",
        "shortcut_size",
        "seed",
        "metrics_file",
        "nfe",
        "step_type",
        "step_display",
        "step_number",
    }

    # Get all columns that might be metrics
    all_cols = set(df.columns)
    metric_cols = all_cols - metadata_cols

    # Filter to only include numeric columns that look like metrics
    numeric_metric_cols = []
    for col in metric_cols:
        if df[col].dtype in ["float64", "int64", "float32", "int32"]:
            numeric_metric_cols.append(col)

    return sorted(numeric_metric_cols)


def prepare_comparison_data(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Prepare data for baseline comparison.

    Args:
        df: DataFrame with metrics
        metric_col: Metric column to analyze

    Returns:
        DataFrame prepared for comparison
    """
    # Filter to required columns and remove NaN values
    required_cols = ["experiment_type", "shortcut_size", "nfe", metric_col]
    df_clean = df[required_cols].dropna()

    # Group by experiment and NFE, taking mean of metric values
    comparison_data = df_clean.groupby(["experiment_type", "nfe", "shortcut_size"])[metric_col].mean().reset_index()

    return comparison_data


def create_baseline_comparison_plots(
    df: pd.DataFrame, metric_col: str, dataset_name: str, baseline_name: str = "baseline_dim128_tied"
) -> None:
    """
    Create comparison plots between baseline and other experiments.

    Args:
        df: DataFrame with metrics
        metric_col: Metric column to plot
        dataset_name: Name of the dataset for plot titles
        baseline_name: Name of the baseline experiment
    """
    # Prepare data
    plot_data = prepare_comparison_data(df, metric_col)

    if plot_data.empty:
        print(f"No data available for metric: {metric_col}")
        return

    # Get clean metric name for display
    clean_metric = clean_metric_name(metric_col)

    # Get list of experiments
    experiments = plot_data["experiment_type"].unique()
    baseline_data = plot_data[plot_data["experiment_type"] == baseline_name]

    if baseline_data.empty:
        print(f"No baseline data found for {baseline_name}")
        return

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Line plot showing trends
    for exp in experiments:
        exp_data = plot_data[plot_data["experiment_type"] == exp].sort_values("nfe")
        if not exp_data.empty:
            label = "Baseline" if exp == baseline_name else exp
            linestyle = "-" if exp == baseline_name else "--"
            ax1.plot(exp_data["nfe"], exp_data[metric_col], marker="o", label=label, linestyle=linestyle, linewidth=2)

    ax1.set_xlabel("Number of Function Evaluations (NFE)")
    ax1.set_ylabel(clean_metric)
    ax1.set_title(f"{clean_metric} vs NFE - {dataset_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar plot for direct comparison
    nfe_values = sorted(plot_data["nfe"].unique())
    x_pos = np.arange(len(nfe_values))
    width = 0.35

    baseline_values = []
    other_exp_values = []
    other_exp_name = None

    for nfe in nfe_values:
        baseline_val = baseline_data[baseline_data["nfe"] == nfe][metric_col].values
        baseline_values.append(baseline_val[0] if len(baseline_val) > 0 else 0)

    # Find the most common non-baseline experiment for comparison
    non_baseline_exps = [exp for exp in experiments if exp != baseline_name]
    if non_baseline_exps:
        other_exp_name = non_baseline_exps[0]  # Take the first one
        other_exp_data = plot_data[plot_data["experiment_type"] == other_exp_name]

        for nfe in nfe_values:
            other_val = other_exp_data[other_exp_data["nfe"] == nfe][metric_col].values
            other_exp_values.append(other_val[0] if len(other_val) > 0 else 0)

        # Create bar plot
        ax2.bar(x_pos - width / 2, baseline_values, width, label="Baseline", alpha=0.8)
        ax2.bar(x_pos + width / 2, other_exp_values, width, label=other_exp_name, alpha=0.8)

        ax2.set_xlabel("Number of Function Evaluations (NFE)")
        ax2.set_ylabel(clean_metric)
        ax2.set_title(f"{clean_metric} Comparison - {dataset_name}")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{nfe:.1f}" for nfe in nfe_values])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_baseline_comparison_summary(
    df: pd.DataFrame, dataset_name: str, baseline_name: str = "baseline_dim128_tied"
) -> pd.DataFrame:
    """
    Generate summary statistics comparing all experiments to baseline.

    Args:
        df: DataFrame with metrics
        dataset_name: Name of the dataset
        baseline_name: Name of the baseline experiment

    Returns:
        DataFrame with comparison summary
    """
    metric_cols = get_metric_columns(df)
    summary_data = []

    for metric_col in metric_cols:
        plot_data = prepare_comparison_data(df, metric_col)

        if plot_data.empty:
            continue

        baseline_data = plot_data[plot_data["experiment_type"] == baseline_name]

        if baseline_data.empty:
            continue

        # Calculate average performance for each experiment
        exp_averages = plot_data.groupby("experiment_type")[metric_col].mean()
        baseline_avg = exp_averages.get(baseline_name, 0)

        for exp_name, exp_avg in exp_averages.items():
            vs_baseline_diff = exp_avg - baseline_avg
            vs_baseline_pct = (vs_baseline_diff / baseline_avg * 100) if baseline_avg != 0 else 0

            summary_data.append(
                {
                    "dataset": dataset_name,
                    "metric": clean_metric_name(metric_col),
                    "metric_raw": metric_col,
                    "experiment": exp_name,
                    "avg_score": exp_avg,
                    "baseline_score": baseline_avg,
                    "vs_baseline_diff": vs_baseline_diff,
                    "vs_baseline_pct": vs_baseline_pct,
                }
            )

    return pd.DataFrame(summary_data)


def create_correlation_heatmap(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Create correlation heatmap for metrics.

    Args:
        df: DataFrame with metrics
        dataset_name: Name of the dataset for plot title
    """
    metric_cols = get_metric_columns(df)

    if len(metric_cols) < 2:
        print("Not enough metrics for correlation analysis")
        return

    # Get correlation matrix
    corr_data = df[metric_cols].corr()

    # Create clean labels
    clean_labels = [clean_metric_name(col) for col in metric_cols]
    corr_data = corr_data.rename(
        index=dict(zip(metric_cols, clean_labels, strict=True)),
        columns=dict(zip(metric_cols, clean_labels, strict=True)),
    )

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_data, annot=True, cmap="coolwarm", center=0, square=True, fmt=".3f", cbar_kws={"label": "Correlation"}
    )
    plt.title(f"Metric Correlations - {dataset_name}")
    plt.tight_layout()
    plt.show()


def create_hierarchical_legend_mapping() -> dict[str, str]:
    """
    Create a hierarchical legend mapping based on experiment patterns.

    Returns:
        Dictionary mapping experiment names to hierarchical display labels with indentation.
    """
    return {
        # Baseline experiments
        "baseline": "Baseline",
        "baseline_dim128_tied": "Baseline (128d)",
        "baseline_cfg": "Baseline + CFG",
        "baseline-cfg": "Baseline + CFG",
        "baseline_emb-pt-l-tied": "Baseline + Emb. Init",
        "baseline_bert-pt-l-tied": "Baseline + Transformer Init",
        "baseline-emb-pt-frze-tied": " Baseline + Emb. Init Frozen",
        "baseline_dim128_sc=.5": "Baseline (128d) + Self-Conditioning",
        "baseline_sc=.5": "Baseline + Self-Conditioning",
        "baseline_vel": "Baseline + Velocity Param.",

        # Shortcut experiments
        "scut": "Shortcut",
        "scut_dim128_w=.1": "Shortcut (128d)",
        "scut-cfg": "Shortcut + CFG",
        "scut_cfg": "Shortcut + CFG",
        "scut_emb-pt-l-tied": "Shortcut + Embedding Init",
        "scut_bert-pt-l-tied": "Shortcut + Transformer Init",
        "scut-emb-pt-frze-tied": "Shortcut + Emb Init Frozen",
        "scut_emb-pt-frze-tied": "Shortcut + Emb Init Frozen",
        "scut_sc=.5": "Shortcut (128d) + Self-Conditioning",
        "scut_delayed=10k": "Shortcut + Delayed Self-Consistency",
        'scut_dim128_w=0.1': "Shortcut (128d) + Self-Consistency weight = 0.1",
        'scut_dim128_w=1': "Shortcut (128d)",
        'scut_dim128_w=2': "Shortcut (128d) + Self-Consistency weight = 2"
    }


def apply_experiment_labels(
    experiment_name: str, experiment_labels: dict[str, str] | None = None
) -> str:
    """
    Apply experiment label mapping if provided, otherwise return original name.

    Args:
        experiment_name: Original experiment name
        experiment_labels: Optional mapping of experiment names to display labels

    Returns:
        Display label for the experiment
    """
    if experiment_labels and experiment_name in experiment_labels:
        return experiment_labels[experiment_name]
    return experiment_name


def assign_colors_to_extensions(experiment_names: list[str]) -> dict[str, str]:
    """
    Assign distinct colors to each experiment.

    Args:
        experiment_names: List of experiment names

    Returns:
        Dictionary mapping experiment name to color
    """
    # Expanded color palette with more distinct colors for better visibility
    colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#ff9896",  # Light red
        "#c5b0d5",  # Light purple
        "#c49c94",  # Light brown
        "#f7b6d3",  # Light pink
        "#c7c7c7",  # Light gray
        "#dbdb8d",  # Light olive
        "#9edae5",  # Light cyan
        "#ffbb78",  # Light orange
        "#98df8a",  # Light green
        "#aec7e8",  # Light blue
    ]

    # Map each unique experiment to a unique color
    color_mapping = {}
    sorted_experiments = sorted(experiment_names)

    for i, exp in enumerate(sorted_experiments):
        color_mapping[exp] = colors[i % len(colors)]

    return color_mapping


def map_line_styles(
    experiment_names: list[str], baseline_experiments: list[str], shortcut_experiments: list[str]
) -> dict[str, str]:
    """
    Map line styles to experiments based on baseline vs shortcut classification.

    Args:
        experiment_names: List of all experiment names
        baseline_experiments: List of baseline experiment names
        shortcut_experiments: List of shortcut experiment names

    Returns:
        Dictionary mapping experiment name to line style
    """
    line_style_mapping = {}

    for exp in experiment_names:
        if exp in baseline_experiments:
            line_style_mapping[exp] = "-"  # solid line
        elif exp in shortcut_experiments:
            line_style_mapping[exp] = "--"  # dashed line
        else:
            # Default to solid for unknown experiments
            line_style_mapping[exp] = "-"

    return line_style_mapping


def extract_extension_type(experiment_name: str) -> str:
    """
    Extract extension type from experiment name.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Extension type string
    """
    if "baseline" in experiment_name:
        if "cfg" in experiment_name:
            return "cfg"
        elif "sc" in experiment_name and "_sc" in experiment_name:
            return "sc"
        else:
            return "baseline"
    elif "scut" in experiment_name:
        if "cfg" in experiment_name:
            return "cfg"
        elif "_sc" in experiment_name:  # Only match _sc suffix
            return "sc"
        else:
            return "scut"
    else:
        return experiment_name


def filter_experiments_by_group(
    df: pd.DataFrame, selected_experiments: list[str], exp_name_col: str = "experiment_type"
) -> pd.DataFrame:
    """
    Filter DataFrame to include only selected experiments.

    Args:
        df: Input DataFrame
        selected_experiments: List of experiment names to include
        exp_name_col: Column name for experiment names

    Returns:
        Filtered DataFrame
    """
    return df[df[exp_name_col].isin(selected_experiments)].copy()


def map_markers(experiment_names: list[str]) -> dict[str, str]:
    """
    Assign different markers to each experiment for better readability.

    Args:
        experiment_names: List of experiment names

    Returns:
        Dictionary mapping experiment name to marker style
    """
    # Define a variety of clear, distinct markers
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x", "d"]

    marker_mapping = {}
    for i, exp_name in enumerate(sorted(experiment_names)):
        marker_mapping[exp_name] = markers[i % len(markers)]

    return marker_mapping


def assign_colors_to_pairs(
    experiment_pairs: list[tuple[str, ...]],
    color_registry: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Assign consistent colors to experiment pairs with hue variations.

    Each pair gets a base color, with experiments within the pair getting
    slight hue variations of that base color for visual grouping.

    Args:
        experiment_pairs: List of tuples containing related experiment names
        color_registry: Mutable registry for cross-plot color consistency

    Returns:
        Dictionary mapping experiment name to hex color code

    Example:
        pairs = [("baseline_A", "shortcut_A"), ("baseline_B", "shortcut_B")]
        colors = assign_colors_to_pairs(pairs)
        # baseline_A and shortcut_A get similar colors (e.g., blue variations)
        # baseline_B and shortcut_B get different similar colors (e.g., green variations)
    """
    if color_registry is None:
        color_registry = {}

    # Base color palette with distinct, visually appealing colors
    base_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Olive
        "#17becf",  # Cyan
        "#ff9896",  # Light red
        "#c5b0d5",  # Light purple
        "#c49c94",  # Light brown
        "#f7b6d3",  # Light pink
        "#c7c7c7",  # Light gray
        "#dbdb8d",  # Light olive
        "#9edae5",  # Light cyan
        "#ffbb78",  # Light orange
        "#98df8a",  # Light green
        "#aec7e8",  # Light blue
    ]

    color_mapping = {}

    for pair_idx, pair in enumerate(experiment_pairs):
        # Get base color for this pair
        base_color = base_colors[pair_idx % len(base_colors)]

        # Convert to HSV for hue manipulation
        hex_color = base_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        # Create very subtle variations for experiments within the pair
        for exp_idx, exp_name in enumerate(pair):
            # Check if this experiment already has a color in registry
            if exp_name in color_registry:
                color_mapping[exp_name] = color_registry[exp_name]
                continue

            # Option 1: Use identical colors (uncomment this for identical colors)
            # hex_color = base_color

            # Option 2: Slightly more distinct brightness variations
            # Adjust brightness more noticeably while keeping hue and saturation identical
            brightness_shift = (exp_idx * 0.15) - 0.075  # Max ±7.5% brightness change
            new_v = max(0.2, min(1.0, v + brightness_shift))

            # Keep hue and saturation exactly the same
            new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, new_v)
            hex_color = f"#{int(new_r*255):02x}{int(new_g*255):02x}{int(new_b*255):02x}"

            color_mapping[exp_name] = hex_color
            color_registry[exp_name] = hex_color

    return color_mapping


def map_paired_visual_properties(
    experiment_pairs: list[tuple[str, ...]]
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Map visual properties (markers, line styles) based on position in pairs.
    
    Args:
        experiment_pairs: List of tuples containing related experiment names
    
    Returns:
        Tuple of (marker_mapping, line_style_mapping) dictionaries
    
    Rules:
        - Position 0 (baseline): square marker, solid line
        - Position 1 (shortcut): triangle marker, dashed line
        - Additional positions follow alternating patterns
    """
    marker_mapping = {}
    line_style_mapping = {}

    # Define marker and line style patterns
    markers = ["s", "^", "o", "D", "v", "<", ">", "p"]  # Start with square, triangle
    line_styles = ["-", "--", "-.", ":"]  # Start with solid, dashed

    for pair in experiment_pairs:
        for exp_idx, exp_name in enumerate(pair):
            # Assign markers: square for position 0, triangle for position 1, etc.
            marker_mapping[exp_name] = markers[exp_idx % len(markers)]

            # Assign line styles: solid for even positions, dashed for odd positions
            line_style_mapping[exp_name] = line_styles[exp_idx % len(line_styles)]

    return marker_mapping, line_style_mapping


def plot_experiment_curves(
    df: pd.DataFrame,
    metric: str,
    nfe_col: str,
    color_mapping: dict[str, str],
    line_style_mapping: dict[str, str],
    marker_mapping: dict[str, str],
    exp_name_col: str = "experiment_type",
    experiment_labels: dict[str, str] | None = None,
) -> None:
    """
    Plot individual experiment curves with assigned colors, line styles, and markers.

    Args:
        df: DataFrame with experiment data
        metric: Metric column name
        nfe_col: NFE column name
        color_mapping: Dictionary mapping extension type to color
        line_style_mapping: Dictionary mapping experiment name to line style
        marker_mapping: Dictionary mapping experiment name to marker style
        exp_name_col: Column name for experiment names
        experiment_labels: Optional mapping of experiment names to display labels
    """
    for exp_name in df[exp_name_col].unique():
        exp_data = df[df[exp_name_col] == exp_name].copy()
        if exp_data.empty:
            continue

        # Sort by NFE for proper line plotting
        exp_data = exp_data.sort_values(nfe_col)

        # Extract color directly from experiment name
        color = color_mapping.get(exp_name, "#1f77b4")
        line_style = line_style_mapping.get(exp_name, "-")
        marker = marker_mapping.get(exp_name, "o")

        # Get display label for legend
        display_label = apply_experiment_labels(exp_name, experiment_labels)

        # Plot the curve with unique marker
        plt.plot(
            exp_data[nfe_col],
            exp_data[metric],
            color=color,
            linestyle=line_style,
            marker=marker,
            label=display_label,
            linewidth=2,
            markersize=8,
        )

# TODO: sohrcut)size param should be named nfe, now it extremely confusign
def plot_metric_over_training_steps(
    df: pd.DataFrame,
    metric: str,
    baseline_experiments: list[str] | None = None,
    shortcut_experiments: list[str] | None = None,
    shortcut_size: int = 256,
    step_col: str = "step_number",
    nfe_col: str = "nfe",
    exp_name_col: str = "experiment_type",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    legend_position: str = "upper left",
    legend_bbox: tuple[float, float] | None = None,
    xscale: str = "linear",
    experiment_labels: dict[str, str] | None = None,
    experiment_pairs: list[tuple[str, ...]] | None = None,
    color_registry: dict[str, str] | None = None,
    save_path: str | None = None,
    force_save: bool = False,
) -> None:
    """
    Plot how the chosen metric evolves over training steps for baseline
    and shortcut experiments at a given shortcut size.

    Args:
        df: Data containing experiment, training steps, NFE, and metric values.
        metric: Metric to plot (e.g., 'bleu_bleu', 'bertscore_f1').
        baseline_experiments: List of baseline experiment names.
        shortcut_experiments: List of shortcut experiment names.
        shortcut_size: The NFE/shortcut size to filter by.
        step_col: Column name for training steps.
        nfe_col: Column name for number of function evaluations.
        exp_name_col: Column name for experiment identifiers.
        title: Plot title. If None, auto-generated.
        figsize: Figure size as (width, height).
        legend_position: Legend position ('upper left', 'lower right', etc.).
        legend_bbox: Fine-grained legend position as (x, y) coordinates for bbox_to_anchor.
                    When provided, overrides legend_position. Example: (0.02, 0.98) for top left
        xscale: Scale for x-axis. Options: "linear", "log", "symlog", "logit"
        experiment_labels: Optional mapping from experiment names to display labels.
        experiment_pairs: Optional list of experiment pairs for consistent colors.
        color_registry: Optional color registry for cross-plot consistency.
        save_path: Optional path to save the plot (e.g., 'figures/training_evolution.png').
                  Supports common formats: .png, .pdf, .svg, .jpg, .eps
        force_save: If True, overwrite existing files; if False, skip saving if file exists
    """
    # Validate inputs
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in DataFrame columns")
        return

    if step_col not in df.columns:
        print(f"Step column '{step_col}' not found in DataFrame columns")
        return

    if exp_name_col not in df.columns:
        raise ValueError(f"Experiment name column '{exp_name_col}' not found in DataFrame columns")

    # Handle experiment selection - use pairs if provided, otherwise use baseline/shortcut lists
    if experiment_pairs is not None:
        # Extract all experiments from pairs
        all_experiments = []
        for pair in experiment_pairs:
            all_experiments.extend(pair)
        selected_experiments = list(set(all_experiments))

        # Assign colors using pairing logic
        if color_registry is None:
            color_registry = {}
        color_mapping = assign_colors_to_pairs(experiment_pairs, color_registry)

        # Map visual properties using pairing logic
        marker_mapping, line_style_mapping = map_paired_visual_properties(experiment_pairs)
    else:
        # Legacy mode: use baseline and shortcut experiments
        baseline_experiments = baseline_experiments or []
        shortcut_experiments = shortcut_experiments or []
        selected_experiments = list(set(baseline_experiments + shortcut_experiments))

        # Use traditional color assignment
        color_mapping = assign_colors_to_extensions(selected_experiments)
        line_style_mapping = map_line_styles(selected_experiments, baseline_experiments, shortcut_experiments)
        marker_mapping = map_markers(selected_experiments)

    # Filter data to selected experiments and shortcut size
    filtered_df = df[
        (df[exp_name_col].isin(selected_experiments))
        & (df[nfe_col] == shortcut_size)
        & (df["step_type"] == "training")  # Only training steps
    ].copy()

    if filtered_df.empty:
        print(f"No data found for selected experiments at shortcut_size={shortcut_size}")
        return

    # Remove rows with missing metric values
    filtered_df = filtered_df.dropna(subset=[metric, step_col])

    if filtered_df.empty:
        print(f"No valid data found for metric '{metric}' and step column '{step_col}'")
        return

    # Check if experiments have multiple steps
    step_counts = filtered_df.groupby(exp_name_col)[step_col].nunique()
    multi_step_experiments = step_counts[step_counts > 1].index.tolist()

    if not multi_step_experiments:
        print(f"No experiments found with multiple training steps at shortcut_size={shortcut_size}")
        return

    print(f"Plotting experiments with multiple steps: {multi_step_experiments}")

    # Filter to only experiments with multiple steps
    filtered_df = filtered_df[filtered_df[exp_name_col].isin(multi_step_experiments)]

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot training step evolution for each experiment
    for exp_name in multi_step_experiments:
        exp_data = filtered_df[filtered_df[exp_name_col] == exp_name].copy()
        if exp_data.empty:
            continue

        # Sort by training steps for proper line plotting
        exp_data = exp_data.sort_values(step_col)

        # Get visual properties
        color = color_mapping.get(exp_name, "#1f77b4")
        line_style = line_style_mapping.get(exp_name, "-")
        marker = marker_mapping.get(exp_name, "o")

        # Get display label for legend
        display_label = apply_experiment_labels(exp_name, experiment_labels)

        # Plot the evolution curve
        plt.plot(
            exp_data[step_col],
            exp_data[metric],
            color=color,
            linestyle=line_style,
            marker=marker,
            label=display_label,
            linewidth=2,
            markersize=8,
        )

    # Customize plot
    metric_display_name = clean_metric_name(metric)
    plt.xlabel("Training Steps")
    plt.ylabel(metric_display_name)

    # Apply x-axis scale with base-2 logarithmic support
    if xscale == "log":
        plt.xscale("log", base=2)
        # Set tick locations based on actual data values, not axis limits
        actual_step_values = sorted(filtered_df[step_col].dropna().unique())
        if actual_step_values:
            # For training steps, generate powers of 2 that match or bracket the actual data
            data_min, data_max = min(actual_step_values), max(actual_step_values)
            tick_powers = []
            power = 0
            while 2**power <= data_max:
                if 2**power >= data_min:
                    tick_powers.append(2**power)
                power += 1

            # Only set ticks if we have any valid powers of 2
            if tick_powers:
                ax = plt.gca()
                ax.set_xticks(tick_powers)
    else:
        plt.xscale(xscale)

    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)

    # Create legend with configurable position
    if legend_bbox is not None:
        # Use fine-grained positioning with custom bbox coordinates
        plt.legend(bbox_to_anchor=legend_bbox, loc="upper left")
    elif legend_position == "outside":
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        plt.legend(loc=legend_position)

    plt.tight_layout()

    # Handle plot saving
    if save_path is not None:
        from pathlib import Path

        save_file = Path(save_path)

        # Create directory if it doesn't exist
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and handle force_save logic
        if save_file.exists():
            if force_save:
                print(f"⚠️  Overwriting existing plot: {save_path}")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved: {save_path}")
            else:
                print(f"ℹ️  Plot already exists at {save_path}. Use force_save=True to overwrite.")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")

    plt.show()


def plot_experiment_performance(
    df: pd.DataFrame,
    metric: str,
    baseline_experiments: list[str] | None = None,
    shortcut_experiments: list[str] | None = None,
    extension_col: str = "extension_type",
    nfe_col: str = "nfe",
    exp_name_col: str = "experiment_type",
    title: str | None = None,
    xlabel: str = "Number of Function Evaluations (NFE)",
    figsize: tuple[int, int] = (10, 6),
    legend_position: str = "outside",
    legend_bbox: tuple[float, float] | None = None,
    xscale: str = "linear",
    experiment_labels: dict[str, str] | None = None,
    experiment_pairs: list[tuple[str, ...]] | None = None,
    color_registry: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    force_save: bool = False,
) -> None:
    """
    Plot performance curves for baseline vs shortcut flow matching experiments
    and their extensions.

    Args:
        df: Data with columns for experiment name, NFE, metric values, and extension type
        metric: The metric to plot (e.g., 'bleu_bleu', 'bertscore_f1')
        baseline_experiments: Experiment names considered as baseline or baseline+extensions
        shortcut_experiments: Experiment names considered as shortcut or shortcut+extensions
        extension_col: Column name in df indicating extension type (optional, for future use)
        nfe_col: Column name for number of function evaluations
        exp_name_col: Column name for experiment identifiers
        title: Custom title for the plot (optional)
        figsize: Figure size tuple (width, height)
        legend_position: Legend position - "outside" for next to plot, or matplotlib location
                        strings like "upper left", "upper right", "lower left", "lower right",
                        "center", "upper center", "lower center", "center left", "center right"
        legend_bbox: Fine-grained legend position as (x, y) coordinates for bbox_to_anchor.
                    When provided, overrides legend_position. Example: (1.02, 0.8) for slightly
                    higher than bottom right
        xscale: Scale for x-axis. Options: "linear", "log", "symlog", "logit"
        experiment_labels: Optional mapping of experiment names to display labels
        experiment_pairs: List of tuples containing paired experiment names for consistent coloring
        color_registry: Mutable registry for cross-plot color consistency (used with experiment_pairs)
        save_path: Optional path to save the plot (e.g., 'figures/comparison.png').
                  Supports common formats: .png, .pdf, .svg, .jpg, .eps
        force_save: If True, overwrite existing files; if False, skip saving if file exists
    """
    # Validate inputs
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame columns")

    if nfe_col not in df.columns:
        raise ValueError(f"NFE column '{nfe_col}' not found in DataFrame columns")

    if exp_name_col not in df.columns:
        raise ValueError(f"Experiment name column '{exp_name_col}' not found in DataFrame columns")

    # Determine plotting mode and validate parameters
    if experiment_pairs is not None:
        # New pairing mode
        if not experiment_pairs:
            raise ValueError("experiment_pairs cannot be empty when provided")

        # Extract all experiment names from pairs
        selected_experiments = []
        for pair in experiment_pairs:
            selected_experiments.extend(pair)
        selected_experiments = list(set(selected_experiments))  # Remove duplicates

        # Assign colors and visual properties using pairing system
        color_mapping = assign_colors_to_pairs(experiment_pairs, color_registry)
        marker_mapping, line_style_mapping = map_paired_visual_properties(experiment_pairs)

    else:
        # Legacy mode - backward compatibility
        if baseline_experiments is None or shortcut_experiments is None:
            raise ValueError(
                "baseline_experiments and shortcut_experiments must be provided when experiment_pairs is None"
            )

        # Combine all selected experiments
        selected_experiments = list(set(baseline_experiments + shortcut_experiments))

        # Assign colors, line styles, and markers using legacy system
        color_mapping = assign_colors_to_extensions(selected_experiments)
        line_style_mapping = map_line_styles(selected_experiments, baseline_experiments, shortcut_experiments)
        marker_mapping = map_markers(selected_experiments)

    # Filter data to selected experiments
    filtered_df = filter_experiments_by_group(df, selected_experiments, exp_name_col)

    if filtered_df.empty:
        print(f"No data found for selected experiments: {selected_experiments}")
        return

    # Remove rows with missing metric values
    filtered_df = filtered_df.dropna(subset=[metric, nfe_col])

    if filtered_df.empty:
        print(f"No valid data found for metric '{metric}' and NFE column '{nfe_col}'")
        return

    # Create the plot
    plt.figure(figsize=figsize)

    # Plot experiment curves
    plot_experiment_curves(
        filtered_df, metric, nfe_col, color_mapping, line_style_mapping, marker_mapping, exp_name_col, experiment_labels
    )

    # Customize plot
    metric_display_name = clean_metric_name(metric)
    plt.xlabel(xlabel)
    plt.ylabel(metric_display_name)

    # Apply x-axis scale with base-2 logarithmic support
    if xscale == "log":
        plt.xscale("log", base=2)
        # Set tick locations based on actual data values, not axis limits
        actual_nfe_values = sorted(filtered_df[nfe_col].dropna().unique())
        if actual_nfe_values:
            # Generate powers of 2 that match or bracket the actual data
            data_min, data_max = min(actual_nfe_values), max(actual_nfe_values)
            tick_powers = []
            power = 0
            while 2**power <= data_max:
                if 2**power >= data_min:
                    tick_powers.append(2**power)
                power += 1

            # Only set ticks if we have any valid powers of 2
            if tick_powers:
                ax = plt.gca()
                ax.set_xticks(tick_powers)
    else:
        plt.xscale(xscale)

    if title:
        plt.title(title)

    plt.grid(True, alpha=0.3)

    # Create a clean legend with configurable position
    if legend_bbox is not None:
        # Use fine-grained positioning with custom bbox coordinates
        plt.legend(bbox_to_anchor=legend_bbox, loc="upper left")
    elif legend_position == "outside":
        # Legend positioned outside the plot area (next to the plot)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        # Legend positioned inside the plot area using standard matplotlib locations
        plt.legend(loc=legend_position)

    plt.tight_layout()

    # Handle plot saving
    if save_path is not None:
        from pathlib import Path

        save_file = Path(save_path)

        # Create directory if it doesn't exist
        save_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and handle force_save logic
        if save_file.exists():
            if force_save:
                print(f"⚠️  Overwriting existing plot: {save_path}")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Plot saved: {save_path}")
            else:
                print(f"ℹ️  Plot already exists at {save_path}. Use force_save=True to overwrite.")
        else:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")

    plt.show()


def build_best_df(
    df: pd.DataFrame,
    bleu_col: str = "bleu_bleu",
    exp_col: str = "experiment_type",
    nfe_col: str = "nfe",
    step_num_col: str | None = None,
    step_type_col: str = "step_type",
    step_display_col: str = "step_display",
    nfe_filter: float = 1.0
) -> pd.DataFrame:
    """Return dataframe with a single (best) step per experiment across all NFEs.

    Selection rule:
      1. Filter rows where nfe == nfe_filter (highest fidelity) and metric present.
      2. For each experiment_type pick the row with max BLEU.
         Tie-breakers (in order): BLEU desc, step_type ('final' preferred),
         step_number desc (if available), step_display lexical.
      3. Using the chosen step for each experiment, collect *all* rows in the original df
         matching that experiment & step (across all nfe values).

    Fallbacks:
      - If step_number column not provided / absent, tries to infer (uses None for ordering).
      - Experiments without any row at nfe==1.0 are skipped (reported).

    Returns:
      DataFrame containing one step per experiment (multiple NFEs retained).
    """
    work = df.copy()

    # Identify step number column automatically if not specified
    if step_num_col is None:
        for cand in ("step_number", "step_num", "step"):  # ordered preference
            if cand in work.columns:
                step_num_col = cand
                break
    if step_num_col is None:
        raise ValueError("Could not find a step number column (looked for 'step_number', 'step_num', 'step').")

    required_cols = {bleu_col, exp_col, nfe_col, step_num_col}
    missing = [c for c in required_cols if c not in work.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Filter to nfe == 1.0 rows for selection
    sel = work[(work[nfe_col] == nfe_filter) & work[bleu_col].notna()].copy()
    if sel.empty:
        raise ValueError("No rows found with nfe == 1.0 and non-null BLEU values.")

    # Add helper columns for tie-breaking
    if step_type_col in sel.columns:
        sel["_is_final"] = (sel[step_type_col] == "final").astype(int)
    else:
        sel["_is_final"] = 0

    # Ensure consistent ordering for tie-breaking
    sort_cols = [exp_col, bleu_col, "_is_final", step_num_col]
    sort_asc = [True, False, False, False]

    sel = sel.sort_values(sort_cols, ascending=sort_asc)

    # Pick first (best) row per experiment_type
    best_rows = sel.groupby(exp_col, as_index=False).head(1)[[exp_col, step_num_col]]
    best_rows = best_rows.rename(columns={step_num_col: "_best_step"})

    # Merge back to get all NFEs for the chosen step
    merged = work.merge(best_rows, left_on=exp_col, right_on=exp_col, how="inner")
    df_best = merged[merged[step_num_col] == merged["_best_step"].astype(merged[step_num_col].dtype)].copy()
    df_best.drop(columns=["_best_step"], inplace=True)

    # Order for readability
    df_best.sort_values([exp_col, nfe_col], inplace=True)

    # Report skipped experiments (no nfe==1.0 rows)
    selected_exps = set(best_rows[exp_col])
    all_exps = set(work[exp_col].unique())
    skipped = sorted(all_exps - selected_exps)
    if skipped:
        print(f"Skipped {len(skipped)} experiments without nfe==1.0 rows: {skipped}")

    print(f"Selected best step for {len(selected_exps)} experiments.")
    return df_best
