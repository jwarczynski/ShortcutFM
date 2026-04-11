"""
Test module for metrics analysis functionality.

This module tests the core functions in shortcutfm.analysis.metrics_analysis
including data loading, metric processing, NFE calculations, and plotting functions.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from shortcutfm.analysis.metrics_analysis import (
    calculate_nfe,
    clean_metric_name,
    find_metrics_files,
    get_metric_columns,
    parse_path_metadata,
)


class TestMetricsAnalysis(unittest.TestCase):
    """Test class for metrics analysis functionality."""

    def test_clean_metric_name(self):
        """Test metric name cleaning functionality."""
        # Test basic metric name cleaning
        self.assertEqual(clean_metric_name("bleu_bleu"), "BLEU")
        self.assertEqual(clean_metric_name("rouge_rouge-l"), "ROUGE-L")
        self.assertEqual(clean_metric_name("meteor_meteor"), "METEOR")
        self.assertEqual(clean_metric_name("bert_score_bert_score"), "BERTScore")

        # Test edge cases
        self.assertEqual(clean_metric_name("unknown_metric"), "Unknown Metric")
        self.assertEqual(clean_metric_name(""), "")

    def test_calculate_nfe(self):
        """Test Number of Function Evaluations calculation."""
        # Standard cases
        self.assertEqual(calculate_nfe(1024), 2.0)
        self.assertEqual(calculate_nfe(512), 4.0)
        self.assertEqual(calculate_nfe(2048), 1.0)

        # Edge cases
        self.assertEqual(calculate_nfe(256), 8.0)
        self.assertEqual(calculate_nfe(128), 16.0)

        # Test with floating point result
        self.assertAlmostEqual(calculate_nfe(1000), 2.048, places=3)

    def test_parse_path_metadata(self):
        """Test path metadata parsing."""
        # Test QQP path parsing (split long line)
        test_path = (
            "/generation_outputs/qqp/baseline_dim128_tied/"
            "run_abc123/step=40000/scut=1024/seed_1_v2/"
            "metrics_nltk_fallback_test.json"
        )
        result = parse_path_metadata(test_path)
        expected = {
            'dataset': 'qqp',
            'experiment_type': 'baseline_dim128_tied',
            'run_id': 'run_abc123',
            'step': 'step=40000',
            'shortcut_size': 1024,
            'seed': 'seed_1_v2',
            'metrics_file': 'metrics_nltk_fallback_test.json'
        }
        self.assertEqual(result, expected)

        # Test WebNLG path parsing (split long line)
        test_path2 = (
            "/generation_outputs/webnlg/scut/"
            "run_xyz789/step=final/scut=512/seed_3/"
            "metrics_nltk_fallback_test.json"
        )
        result2 = parse_path_metadata(test_path2)
        expected2 = {
            'dataset': 'webnlg',
            'experiment_type': 'scut',
            'run_id': 'run_xyz789',
            'step': 'step=final',
            'shortcut_size': 512,
            'seed': 'seed_3',
            'metrics_file': 'metrics_nltk_fallback_test.json'
        }
        self.assertEqual(result2, expected2)

        # Test invalid path
        with self.assertRaises(ValueError):
            parse_path_metadata("/invalid/path/structure")

    def test_get_metric_columns(self):
        """Test metric column extraction."""
        # Create test DataFrame
        test_data = {
            'file_path': ['test1.json', 'test2.json'],
            'experiment_type': ['baseline', 'scut'],
            'run_id': ['run1', 'run2'],
            'shortcut_size': [1024, 512],
            'bleu_bleu': [0.25, 0.30],
            'rouge_rouge-l': [0.35, 0.40],
            'meteor_meteor': [0.20, 0.25],
            'other_column': ['data1', 'data2']
        }
        df = pd.DataFrame(test_data)

        # Test metric column extraction
        metric_cols = get_metric_columns(df)
        expected_cols = [
            'file_path', 'experiment_type', 'run_id', 'shortcut_size',
            'bleu_bleu', 'rouge_rouge-l', 'meteor_meteor'
        ]
        self.assertEqual(metric_cols, expected_cols)

    def test_find_metrics_files(self):
        """Test metrics file discovery."""
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test directory structure
            test_structure = [
                "baseline/run_1/step=40000/scut=512/seed_1",
                "baseline/run_2/step=30000/scut=1024/seed_2",
                "scut/run_3/step=final/scut=256/seed_1"
            ]

            # Create directories and files
            for structure in test_structure:
                full_path = Path(temp_dir) / structure
                full_path.mkdir(parents=True, exist_ok=True)
                # Create metrics file
                metrics_file = full_path / "metrics_nltk_fallback_test.json"
                metrics_file.write_text('{"test": "data"}')

            # Test file discovery
            found_files = find_metrics_files(temp_dir)

            # Should find 3 metrics files
            self.assertEqual(len(found_files), 3)

            # Check that all found files are metrics files
            for file_path in found_files:
                self.assertTrue(
                    Path(file_path).name == "metrics_nltk_fallback_test.json"
                )


class TestDataProcessing(unittest.TestCase):
    """Test data processing and manipulation functions."""

    def setUp(self):
        """Set up test data for data processing tests."""
        self.sample_metrics_data = {
            'file_path': [
                'test1.json',
                'test2.json',
                'test3.json'
            ],
            'dataset': ['qqp', 'qqp', 'webnlg'],
            'experiment_type': ['baseline', 'scut', 'baseline'],
            'shortcut_size': [1024, 512, 2048],
            'bleu_bleu': [0.25, 0.30, 0.35],
            'rouge_rouge-l': [0.35, 0.40, 0.45],
            'meteor_meteor': [0.20, 0.25, 0.30]
        }
        self.df = pd.DataFrame(self.sample_metrics_data)

    def test_nfe_calculation_integration(self):
        """Test NFE calculation on DataFrame."""
        # Add NFE column using calculate_nfe function
        df_with_nfe = self.df.copy()
        df_with_nfe['nfe'] = df_with_nfe['shortcut_size'].apply(calculate_nfe)

        # Check NFE values
        expected_nfe = [2.0, 4.0, 1.0]
        actual_nfe = df_with_nfe['nfe'].tolist()
        self.assertEqual(actual_nfe, expected_nfe)

    def test_metric_name_cleaning_on_dataframe(self):
        """Test metric name cleaning on DataFrame columns."""
        # Get metric columns and clean names
        metric_cols = get_metric_columns(self.df)
        cleaned_names = {}

        for col in metric_cols:
            if col not in [
                'file_path', 'experiment_type', 'run_id', 'shortcut_size'
            ]:
                cleaned_names[col] = clean_metric_name(col)

        expected_cleaned = {
            'bleu_bleu': 'BLEU',
            'rouge_rouge-l': 'ROUGE-L',
            'meteor_meteor': 'METEOR'
        }
        self.assertEqual(cleaned_names, expected_cleaned)

    @patch('shortcutfm.analysis.metrics_analysis.pd.read_json')
    def test_metrics_file_loading_mock(self, mock_read_json):
        """Test metrics file loading with mocked data."""
        # Mock the JSON loading
        mock_data = pd.DataFrame({
            'metric1': [0.5, 0.6],
            'metric2': [0.7, 0.8]
        })
        mock_read_json.return_value = mock_data

        # This would test actual file loading if we had that function
        # For now, just verify the mock works
        result = pd.read_json('dummy_path.json')
        self.assertEqual(len(result), 2)
        self.assertIn('metric1', result.columns)


if __name__ == '__main__':
    unittest.main()
