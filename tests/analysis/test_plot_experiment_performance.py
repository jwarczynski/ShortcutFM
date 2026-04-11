"""
Test module for plot_experiment_performance functionality.

This module contains comprehensive tests for the plotting function that compares
baseline vs shortcut flow matching experiments.
"""

import unittest
from unittest.mock import patch

import pandas as pd

from shortcutfm.analysis.metrics_analysis import (
    assign_colors_to_extensions,
    extract_extension_type,
    filter_experiments_by_group,
    map_line_styles,
    plot_experiment_performance,
)


class TestPlotExperimentPerformance(unittest.TestCase):
    """Test class for plot_experiment_performance functionality."""

    def setUp(self):
        """Set up test data for plotting tests."""
        # Create comprehensive test DataFrame
        self.df = pd.DataFrame({
            'experiment_type': [
                'baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg',
                'baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg',
                'baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg',
                'baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg'
            ],
            'bleu_bleu': [
                0.30, 0.28, 0.32, 0.29,  # NFE = 1.0
                0.35, 0.33, 0.37, 0.34,  # NFE = 2.0
                0.40, 0.38, 0.42, 0.39,  # NFE = 4.0
                0.45, 0.43, 0.47, 0.44   # NFE = 8.0
            ],
            'bertscore_f1': [
                0.75, 0.73, 0.77, 0.74,  # NFE = 1.0
                0.78, 0.76, 0.80, 0.77,  # NFE = 2.0
                0.82, 0.80, 0.84, 0.81,  # NFE = 4.0
                0.85, 0.83, 0.87, 0.84   # NFE = 8.0
            ],
            'nfe': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
                   4.0, 4.0, 4.0, 4.0, 8.0, 8.0, 8.0, 8.0],
            'shortcut_size': [2048, 2048, 2048, 2048, 1024, 1024, 1024, 1024,
                             512, 512, 512, 512, 256, 256, 256, 256]
        })

        self.baseline_experiments = ['baseline_dim128_tied', 'baseline_cfg']
        self.shortcut_experiments = ['scut_dim128_tied', 'scut_cfg']

    def test_assign_colors_to_extensions(self):
        """Test color assignment to extension types."""
        experiment_names = ['baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg']
        color_mapping = assign_colors_to_extensions(experiment_names)

        # Check that all extension types get colors
        self.assertIn('baseline', color_mapping)
        self.assertIn('scut', color_mapping)
        self.assertIn('cfg', color_mapping)

        # Check that colors are valid hex codes
        for color in color_mapping.values():
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)

        # Check consistency - same input should give same output
        color_mapping2 = assign_colors_to_extensions(experiment_names)
        self.assertEqual(color_mapping, color_mapping2)

    def test_map_line_styles(self):
        """Test line style mapping for baseline vs shortcut experiments."""
        experiment_names = ['baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg']
        line_style_mapping = map_line_styles(experiment_names, self.baseline_experiments, self.shortcut_experiments)

        # Check baseline experiments get solid lines
        self.assertEqual(line_style_mapping['baseline_dim128_tied'], '-')
        self.assertEqual(line_style_mapping['baseline_cfg'], '-')

        # Check shortcut experiments get dashed lines
        self.assertEqual(line_style_mapping['scut_dim128_tied'], '--')
        self.assertEqual(line_style_mapping['scut_cfg'], '--')

    def test_extract_extension_type(self):
        """Test extension type extraction from experiment names."""
        # Test baseline variants
        self.assertEqual(extract_extension_type('baseline_dim128_tied'), 'baseline')
        self.assertEqual(extract_extension_type('baseline_cfg'), 'cfg')
        self.assertEqual(extract_extension_type('baseline_sc'), 'sc')

        # Test shortcut variants
        self.assertEqual(extract_extension_type('scut_dim128_tied'), 'scut')
        self.assertEqual(extract_extension_type('scut_cfg'), 'cfg')
        self.assertEqual(extract_extension_type('scut_sc'), 'sc')

        # Test unknown experiment names
        self.assertEqual(extract_extension_type('unknown_experiment'), 'unknown_experiment')

    def test_filter_experiments_by_group(self):
        """Test filtering DataFrame by experiment groups."""
        selected_experiments = ['baseline_dim128_tied', 'scut_dim128_tied']
        filtered_df = filter_experiments_by_group(self.df, selected_experiments)

        # Check that only selected experiments remain
        expected_count = 8  # 2 experiments * 4 NFE values
        self.assertEqual(len(filtered_df), expected_count)
        self.assertSetEqual(set(filtered_df['experiment_type']), set(selected_experiments))

        # Check that data is preserved
        self.assertIn('bleu_bleu', filtered_df.columns)
        self.assertIn('nfe', filtered_df.columns)

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_basic(self, mock_plt):
        """Test basic functionality of plot_experiment_performance."""
        # Call the function - should not raise any errors
        plot_experiment_performance(
            df=self.df,
            metric='bleu_bleu',
            baseline_experiments=self.baseline_experiments,
            shortcut_experiments=self.shortcut_experiments
        )

        # Verify that plotting functions were called
        mock_plt.figure.assert_called()
        mock_plt.xlabel.assert_called_with('Number of Function Evaluations (NFE)')
        mock_plt.ylabel.assert_called_with('BLEU')
        mock_plt.grid.assert_called_with(True, alpha=0.3)
        mock_plt.show.assert_called()

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_bertscore(self, mock_plt):
        """Test plotting with BERTScore metric."""
        plot_experiment_performance(
            df=self.df,
            metric='bertscore_f1',
            baseline_experiments=self.baseline_experiments,
            shortcut_experiments=self.shortcut_experiments
        )

        # Verify that correct metric display name is used
        mock_plt.ylabel.assert_called_with('BERTScore F1')

    def test_plot_experiment_performance_validation(self):
        """Test input validation for plot_experiment_performance."""
        # Test missing metric column
        with self.assertRaises(ValueError) as context:
            plot_experiment_performance(
                df=self.df,
                metric='nonexistent_metric',
                baseline_experiments=self.baseline_experiments,
                shortcut_experiments=self.shortcut_experiments
            )
        self.assertIn("Metric 'nonexistent_metric' not found", str(context.exception))

        # Test missing NFE column
        df_no_nfe = self.df.drop(columns=['nfe'])
        with self.assertRaises(ValueError) as context:
            plot_experiment_performance(
                df=df_no_nfe,
                metric='bleu_bleu',
                baseline_experiments=self.baseline_experiments,
                shortcut_experiments=self.shortcut_experiments
            )
        self.assertIn("NFE column 'nfe' not found", str(context.exception))

        # Test missing experiment column
        df_no_exp = self.df.drop(columns=['experiment_type'])
        with self.assertRaises(ValueError) as context:
            plot_experiment_performance(
                df=df_no_exp,
                metric='bleu_bleu',
                baseline_experiments=self.baseline_experiments,
                shortcut_experiments=self.shortcut_experiments
            )
        self.assertIn("Experiment name column 'experiment_type' not found", str(context.exception))

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_empty_data(self, mock_plt):
        """Test plot_experiment_performance with empty or invalid data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()

        # Should handle empty data gracefully
        with self.assertRaises(ValueError):
            plot_experiment_performance(
                df=empty_df,
                metric='bleu_bleu',
                baseline_experiments=self.baseline_experiments,
                shortcut_experiments=self.shortcut_experiments
            )

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_no_matching_experiments(self, mock_plt):
        """Test behavior when no experiments match the selected ones."""
        # Use experiment names that don't exist in the DataFrame
        non_existent_baseline = ['nonexistent_baseline']
        non_existent_shortcut = ['nonexistent_shortcut']

        # Should print message and return without plotting
        plot_experiment_performance(
            df=self.df,
            metric='bleu_bleu',
            baseline_experiments=non_existent_baseline,
            shortcut_experiments=non_existent_shortcut
        )

        # Should not create any plots
        mock_plt.figure.assert_not_called()

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_custom_params(self, mock_plt):
        """Test plot_experiment_performance with custom parameters."""
        # Create DataFrame with custom column names
        df_custom = pd.DataFrame({
            'exp_name': ['baseline_dim128_tied', 'scut_dim128_tied'] * 4,
            'rouge_score': [0.30, 0.28, 0.35, 0.32, 0.40, 0.38, 0.45, 0.42],
            'function_evals': [1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 8.0, 8.0]
        })

        # Call with custom column names and parameters
        plot_experiment_performance(
            df=df_custom,
            metric='rouge_score',
            baseline_experiments=['baseline_dim128_tied'],
            shortcut_experiments=['scut_dim128_tied'],
            nfe_col='function_evals',
            exp_name_col='exp_name',
            title='Custom Title',
            figsize=(12, 8)
        )

        # Verify figure was created with correct size and title
        mock_plt.figure.assert_called_with(figsize=(12, 8))
        mock_plt.title.assert_called_with('Custom Title')

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_missing_values(self, mock_plt):
        """Test handling of missing values in data."""
        # Create DataFrame with some missing values
        df_with_na = self.df.copy()
        df_with_na.loc[0, 'bleu_bleu'] = None
        df_with_na.loc[1, 'nfe'] = None

        # Should handle missing values by dropping those rows
        plot_experiment_performance(
            df=df_with_na,
            metric='bleu_bleu',
            baseline_experiments=self.baseline_experiments,
            shortcut_experiments=self.shortcut_experiments
        )

        # Should still create plots with remaining valid data
        mock_plt.figure.assert_called()
        mock_plt.show.assert_called()

    @patch('shortcutfm.analysis.metrics_analysis.plt')
    def test_plot_experiment_performance_single_experiment(self, mock_plt):
        """Test plotting with only one type of experiment."""
        # Test with only baseline experiments
        plot_experiment_performance(
            df=self.df,
            metric='bleu_bleu',
            baseline_experiments=self.baseline_experiments,
            shortcut_experiments=[]
        )

        mock_plt.figure.assert_called()
        mock_plt.show.assert_called()

        # Reset mock
        mock_plt.reset_mock()

        # Test with only shortcut experiments
        plot_experiment_performance(
            df=self.df,
            metric='bleu_bleu',
            baseline_experiments=[],
            shortcut_experiments=self.shortcut_experiments
        )

        mock_plt.figure.assert_called()
        mock_plt.show.assert_called()

    def test_color_mapping_consistency(self):
        """Test that color mapping is consistent across multiple calls."""
        experiment_names = ['baseline_dim128_tied', 'scut_dim128_tied', 'baseline_cfg', 'scut_cfg']

        # Generate color mappings multiple times
        mapping1 = assign_colors_to_extensions(experiment_names)
        mapping2 = assign_colors_to_extensions(experiment_names)
        mapping3 = assign_colors_to_extensions(experiment_names)

        # All mappings should be identical
        self.assertEqual(mapping1, mapping2)
        self.assertEqual(mapping2, mapping3)

    def test_line_style_mapping_edge_cases(self):
        """Test line style mapping with edge cases."""
        experiment_names = ['exp1', 'exp2', 'exp3']
        baseline_experiments = ['exp1']
        shortcut_experiments = ['exp2']

        mapping = map_line_styles(experiment_names, baseline_experiments, shortcut_experiments)

        # exp1 should be solid (baseline)
        self.assertEqual(mapping['exp1'], '-')

        # exp2 should be dashed (shortcut)
        self.assertEqual(mapping['exp2'], '--')

        # exp3 should default to solid (not in either list)
        self.assertEqual(mapping['exp3'], '-')

    def test_extension_type_extraction_edge_cases(self):
        """Test extension type extraction with various naming patterns."""
        # Test complex names
        self.assertEqual(extract_extension_type('baseline_dim128_tied_cfg'), 'cfg')
        self.assertEqual(extract_extension_type('scut_model_sc'), 'sc')

        # Test simple names
        self.assertEqual(extract_extension_type('baseline'), 'baseline')
        self.assertEqual(extract_extension_type('scut'), 'scut')

        # Test names without underscores
        self.assertEqual(extract_extension_type('simpleexp'), 'simpleexp')


if __name__ == '__main__':
    unittest.main()
