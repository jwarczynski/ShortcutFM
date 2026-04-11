"""Tests for plot_metric_over_training_steps function."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from shortcutfm.analysis.metrics_analysis import plot_metric_over_training_steps


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)  # For reproducible tests

    # Create test data with multiple experiments, steps, and shortcut sizes
    data = []
    experiments = ["exp1", "exp2", "exp3"]
    shortcut_sizes = [256, 512, 1024]
    training_steps = [1000, 2000, 3000, 4000, 5000]

    for exp in experiments:
        for size in shortcut_sizes:
            for step in training_steps:
                # Add some realistic metric values with noise
                bleu_score = 0.2 + 0.1 * (step / 5000) + np.random.normal(0, 0.02)
                bertscore_f1 = 0.5 + 0.2 * (step / 5000) + np.random.normal(0, 0.03)

                data.append(
                    {
                        "experiment_type": exp,
                        "step_number": step,
                        "shortcut_size": size,
                        "bleu_bleu": max(0, bleu_score),  # Ensure non-negative
                        "bertscore_f1": max(0, bertscore_f1),
                        "step_type": "training",
                    }
                )

    return pd.DataFrame(data)


def test_data_filtering_by_metric_and_shortcut_size(sample_data):
    """Test that function correctly filters data by metric and shortcut size."""
    baseline_exps = ["exp1"]
    shortcut_exps = ["exp2", "exp3"]

    with patch("matplotlib.pyplot.show"):
        plot_metric_over_training_steps(
            df=sample_data,
            metric="bleu_bleu",
            baseline_experiments=baseline_exps,
            shortcut_experiments=shortcut_exps,
            shortcut_size=512,
            nfe_col="shortcut_size",  # Specify the correct column name
        )

    # If no error is raised, the filtering worked correctly


def test_metric_not_in_data():
    """Test handling when requested metric is not in the data."""
    data = pd.DataFrame(
        {
            "experiment_type": ["exp1"],
            "step_number": [1000],
            "shortcut_size": [512],
            "bleu_bleu": [0.2],  # Only BLEU, no BERTScore
            "step_type": ["training"],
        }
    )

    baseline_exps = ["exp1"]
    shortcut_exps = []

    # Should handle gracefully when metric doesn't exist
    with patch("builtins.print") as mock_print:
        plot_metric_over_training_steps(
            df=data,
            metric="bertscore_f1",  # This metric doesn't exist in data
            baseline_experiments=baseline_exps,
            shortcut_experiments=shortcut_exps,
            shortcut_size=512,
            nfe_col="shortcut_size",
        )

        # Should print a message about metric not found
        assert mock_print.called
        # Check that the error message contains the expected text
        mock_print.assert_called_with(
            "Metric 'bertscore_f1' not found in DataFrame columns"
        )
