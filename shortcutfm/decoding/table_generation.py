"""
LaTeX table generation utilities for experimental results.

This module provides functions to generate professional LaTeX tables from
experimental results DataFrames with proper formatting, grouping, and styling.
"""

import pandas as pd


def create_experiment_results_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    exp_col: str = "experiment_type",
    step_col: str = "step_number",
    nfe_col: str = "nfe",
    table_caption: str = "Experimental Results",
    table_label: str = "tab:results",
) -> str:
    """
    Create a LaTeX table from experimental results with individual experiment separation.

    Features:
    - Vertical format with Model | Training Steps | NFE | metrics columns
    - Merged cells for model names and training steps across NFE values
    - Individual experiment separation with \\midrule after each experiment
    - Automatic page width fitting with \\resizebox
    - Professional booktabs styling
    - Bold highlighting for best BLEU model

    Args:
        df: DataFrame with experimental results (best results per experiment)
        metric_cols: List of metric column names to include in table
        exp_col: Column name for experiment types
        step_col: Column name for training steps
        nfe_col: Column name for NFE values
        table_caption: Caption for the LaTeX table
        table_label: Label for the LaTeX table

    Returns:
        String containing the complete LaTeX table code
    """
    from shortcutfm.analysis.metrics_analysis import clean_metric_name

    # Sort data by experiment type and NFE
    df_sorted = df.sort_values([exp_col, nfe_col]).copy()

    # Find the best BLEU score for highlighting
    best_bleu_idx = df_sorted["bleu_bleu"].idxmax()
    best_bleu_exp = df_sorted.loc[best_bleu_idx, exp_col]

    # Create column headers
    metric_headers = [clean_metric_name(col) for col in metric_cols]

    # Group experiments by type (baseline vs shortcut) for sorting
    def get_experiment_group(exp_name):
        if exp_name.startswith("baseline"):
            return "baseline"
        elif exp_name.startswith("scut"):
            return "shortcut"
        else:
            return "other"

    # Add group information and sort by group then experiment
    df_sorted["exp_group"] = df_sorted[exp_col].apply(get_experiment_group)
    df_sorted = df_sorted.sort_values(["exp_group", exp_col, nfe_col])

    # Start building the LaTeX table
    latex = []

    # Table environment
    latex.append("\\begin{table*}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{" + table_caption + "}")
    latex.append("\\label{" + table_label + "}")
    latex.append("\\resizebox{\\textwidth}{!}{%")  # Ensure table fits page width

    # Column specification
    num_cols = 3 + len(metric_cols)  # Model + Training Steps + NFE + metrics
    col_spec = "l" + "c" * (num_cols - 1)
    latex.append("\\begin{tabular}{" + col_spec + "}")
    latex.append("\\toprule")

    # Header row
    header = "Model & Training Steps & NFE & " + " & ".join(metric_headers) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Process experiments (individual separation for each experiment)
    experiments = df_sorted[exp_col].unique()

    for exp_name in experiments:
        exp_data = df_sorted[df_sorted[exp_col] == exp_name].copy()
        exp_rows = len(exp_data)

        # Get step number (should be same for all rows of this experiment)
        step_num = int(exp_data[step_col].iloc[0])

        # Format experiment name for LaTeX
        exp_name_latex = exp_name.replace("_", "\\_")

        # Add bold formatting if this is the best BLEU experiment
        if exp_name == best_bleu_exp:
            exp_name_latex = "\\textbf{" + exp_name_latex + "}"

        # Process each NFE value for this experiment
        for idx, (_, row) in enumerate(exp_data.iterrows()):
            nfe_val = row[nfe_col]

            # Format metric values
            metric_values = []
            for col in metric_cols:
                val = row[col]
                if pd.isna(val):
                    metric_values.append("--")
                else:
                    metric_values.append(f"{val:.3f}")

            # Create the row
            if idx == 0:  # First row for this experiment
                model_cell = f"\\multirow{{{exp_rows}}}{{*}}{{{exp_name_latex}}}"
                step_cell = f"\\multirow{{{exp_rows}}}{{*}}{{{step_num}}}"
            else:  # Subsequent rows
                model_cell = ""
                step_cell = ""

            nfe_cell = f"{nfe_val:.0f}"

            # Combine the row
            row_data = [model_cell, step_cell, nfe_cell] + metric_values
            latex_row = " & ".join(row_data) + " \\\\"
            latex.append(latex_row)

        # Add spacing after each experiment (except the last one)
        if exp_name != experiments[-1]:
            latex.append("\\addlinespace")
            latex.append("\\midrule")  # Add midrule after every experiment for clear separation

    # Table footer
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}%")  # Close resizebox
    latex.append("\\end{table*}")

    return "\n".join(latex)


def create_experiment_mapping_table(experiment_types: list[str]) -> str:
    """
    Create a LaTeX table mapping experiment names to their hyperparameter configurations.

    Args:
        experiment_types: List of experiment type names to map

    Returns:
        String containing the complete LaTeX table code
    """

    def parse_experiment_config(exp_name):
        """Parse experiment name to extract hyperparameter configuration."""
        config = {
            "experiment_name": exp_name,
            "model_type": "",
            "classifier_free_guidance": "No",
            "self_conditioning": "No",
            "embedding_initialization": "Random",
            "transformer_initialization": "Random",
            "embedding_layers_frozen": "No",
            "embedding_dimension": "768",  # Default BERT dimension
            "parameterization": "X0",  # Default parameterization
            "sc_weight": "N/A",  # Only for shortcut models
        }

        # Determine model type
        if "scut" in exp_name.lower():
            config["model_type"] = "Shortcut"
        elif "baseline" in exp_name.lower():
            config["model_type"] = "Baseline"

        # Check for classifier-free guidance
        if "cfg" in exp_name.lower():
            config["classifier_free_guidance"] = "Yes"

        # Check for self conditioning
        if "sc" in exp_name.lower():
            config["self_conditioning"] = "Yes"

        # Parse initialization and freezing
        if "bert-pt-l" in exp_name.lower():
            config["embedding_initialization"] = "BERT"
            config["transformer_initialization"] = "BERT"
            config["embedding_layers_frozen"] = "No"
        elif "emb-pt-l" in exp_name.lower():
            config["embedding_initialization"] = "BERT"
            config["embedding_layers_frozen"] = "No"
        elif "emb-pt-frze" in exp_name.lower():
            config["embedding_initialization"] = "BERT"
            config["embedding_layers_frozen"] = "Yes"

        # Check for parameterization
        if "vel" in exp_name.lower():
            config["parameterization"] = "Velocity"

        # Parse embedding dimension
        if "dim128" in exp_name.lower():
            config["embedding_dimension"] = "128"

        # Parse self-consistency weight (only for shortcut models)
        if config["model_type"] == "Shortcut":
            if "w=" in exp_name.lower():
                # Extract weight value
                import re

                match = re.search(r"w=([0-9.]+)", exp_name.lower())
                if match:
                    weight_val = match.group(1)
                    # Format as 0.X instead of .X
                    if weight_val.startswith("."):
                        weight_val = "0" + weight_val
                    config["sc_weight"] = weight_val
            else:
                config["sc_weight"] = "1.0"  # Default weight

        return config

    # Parse all experiments
    configs = [parse_experiment_config(exp) for exp in experiment_types]

    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table*}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Experiment Configuration Mapping}")
    latex_lines.append("\\label{tab:experiment_mapping}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{%")
    latex_lines.append("\\begin{tabular}{llcccccccc}")
    latex_lines.append("\\toprule")

    # Header
    header = (
        "Experiment Name & Model Type & CFG & Self-Cond. & Emb. Init. & "
        "Trans. Init. & Emb. Frozen & Emb. Dim. & Param. & SC Weight \\\\"
    )
    latex_lines.append(header)
    latex_lines.append("\\midrule")

    # Data rows
    for config in configs:
        exp_name_latex = config["experiment_name"].replace("_", "\\_").replace("=", "=")

        row = (
            f"{exp_name_latex} & "
            f"{config['model_type']} & "
            f"{config['classifier_free_guidance']} & "
            f"{config['self_conditioning']} & "
            f"{config['embedding_initialization']} & "
            f"{config['transformer_initialization']} & "
            f"{config['embedding_layers_frozen']} & "
            f"{config['embedding_dimension']} & "
            f"{config['parameterization']} & "
            f"{config['sc_weight']} \\\\"
        )
        latex_lines.append(row)

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}%")
    latex_lines.append("\\end{table*}")

    return "\n".join(latex_lines)
