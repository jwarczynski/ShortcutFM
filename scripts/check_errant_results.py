#!/usr/bin/env python3
"""
Check and summarize ERRANT evaluation results.
"""

import json
from pathlib import Path


def check_results():
    """Check which files have been evaluated and show summary."""
    base_dir = Path('/home/inf148234/projects/ShortcutFM/generation_outputs/grammar')

    # Find all generation files
    generation_files = list(base_dir.rglob('generation_texts_nltk_fallback.json'))
    print(f"Found {len(generation_files)} generation files total")

    # Check which ones have been evaluated
    evaluated_files = []
    for gen_file in generation_files:
        result_file = gen_file.parent / 'errant_evaluation_cli.json'
        if result_file.exists():
            evaluated_files.append((gen_file, result_file))

    print(f"Found {len(evaluated_files)} evaluated files")

    # Show results for evaluated files
    if evaluated_files:
        print("\nEvaluation Results:")
        print("-" * 100)
        print(f"{'File':<70} {'Precision':<10} {'Recall':<10} {'F-Score':<10}")
        print("-" * 100)

        all_results = []
        for gen_file, result_file in evaluated_files:
            try:
                with open(result_file) as f:
                    results = json.load(f)

                # Extract relative path for display
                rel_path = str(gen_file.relative_to(base_dir))
                print(f"{rel_path:<70} {results['precision']:<10.4f} {results['recall']:<10.4f} {results['f_score']:<10.4f}")
                all_results.append(results)

            except Exception as e:
                print(f"Error reading {result_file}: {e}")

        if all_results:
            # Calculate average metrics
            avg_precision = sum(r['precision'] for r in all_results) / len(all_results)
            avg_recall = sum(r['recall'] for r in all_results) / len(all_results)
            avg_f_score = sum(r['f_score'] for r in all_results) / len(all_results)

            print("-" * 100)
            print(f"{'AVERAGE':<70} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f_score:<10.4f}")

    # Check if summary file exists
    summary_file = base_dir / 'errant_evaluation_cli_summary.json'
    if summary_file.exists():
        print(f"\nSummary file exists: {summary_file}")
        try:
            with open(summary_file) as f:
                summary = json.load(f)
            print(f"Summary contains {len(summary)} entries")
        except Exception as e:
            print(f"Error reading summary file: {e}")
    else:
        print(f"\nSummary file does not exist yet: {summary_file}")


if __name__ == '__main__':
    check_results()
