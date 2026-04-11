#!/usr/bin/env python3
"""
Evaluate grammar correction models using ERRANT CLI tools.

This script converts JSON files to text format and uses the ERRANT CLI
for efficient evaluation of grammatical error correction models.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def load_generation_data(file_path: Path) -> list[dict[str, str]]:
    """Load generation data from JSON file."""
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def write_texts_to_file(texts: list[str], file_path: Path) -> None:
    """Write texts to a file, one per line."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip() + '\n')


def run_errant_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run an ERRANT command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def parse_errant_compare_output(output: str) -> dict[str, float]:
    """Parse the output from errant_compare to extract metrics."""
    lines = output.strip().split('\n')

    # Look for the line with TP, FP, FN values in the span-based correction section
    for i, line in enumerate(lines):
        if 'TP' in line and 'FP' in line and 'FN' in line and 'Prec' in line:
            # Next line should have the actual values
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                parts = values_line.split()
                if len(parts) >= 6:
                    tp = int(parts[0])
                    fp = int(parts[1])
                    fn = int(parts[2])
                    precision = float(parts[3])
                    recall = float(parts[4])
                    f_score = float(parts[5])

                    return {
                        'precision': precision,
                        'recall': recall,
                        'f_score': f_score,
                        'tp': tp,
                        'fp': fp,
                        'fn': fn
                    }

    # Fallback: return zeros if parsing fails
    return {
        'precision': 0.0,
        'recall': 0.0,
        'f_score': 0.0,
        'tp': 0,
        'fp': 0,
        'fn': 0
    }


def evaluate_with_errant_cli(sources: list[str], references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Evaluate using ERRANT CLI tools."""

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create temporary files
        orig_file = temp_path / 'orig.txt'
        ref_file = temp_path / 'ref.txt'
        hyp_file = temp_path / 'hyp.txt'
        ref_m2 = temp_path / 'ref.m2'
        hyp_m2 = temp_path / 'hyp.m2'

        # Write texts to files
        write_texts_to_file(sources, orig_file)
        write_texts_to_file(references, ref_file)
        write_texts_to_file(hypotheses, hyp_file)

        # Annotate reference edits
        ref_cmd = ['errant_parallel', '-orig', str(orig_file), '-cor', str(ref_file), '-out', str(ref_m2)]
        exit_code, stdout, stderr = run_errant_command(ref_cmd)

        if exit_code != 0:
            print(f"Error running reference annotation: {stderr}")
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        # Annotate hypothesis edits
        hyp_cmd = ['errant_parallel', '-orig', str(orig_file), '-cor', str(hyp_file), '-out', str(hyp_m2)]
        exit_code, stdout, stderr = run_errant_command(hyp_cmd)

        if exit_code != 0:
            print(f"Error running hypothesis annotation: {stderr}")
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        # Compare annotations
        compare_cmd = ['errant_compare', '-hyp', str(hyp_m2), '-ref', str(ref_m2)]
        exit_code, stdout, stderr = run_errant_command(compare_cmd)

        if exit_code != 0:
            print(f"Error running comparison: {stderr}")
            return {'precision': 0.0, 'recall': 0.0, 'f_score': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}

        # Parse results
        results = parse_errant_compare_output(stdout)
        results['total_examples'] = len(sources)

        return results


def save_results(results: dict[str, float], output_path: Path) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def evaluate_single_file(file_path: Path) -> dict[str, float]:
    """Evaluate a single generation file with ERRANT CLI."""
    print(f"Evaluating: {file_path}")

    # Load data
    data = load_generation_data(file_path)

    # Extract texts
    sources = [item['source'] for item in data]
    references = [item['reference'] for item in data]
    hypotheses = [item['hypothesis'] for item in data]

    # Evaluate using ERRANT CLI
    results = evaluate_with_errant_cli(sources, references, hypotheses)

    # Save results in the same directory
    output_path = file_path.parent / 'errant_evaluation_cli.json'
    save_results(results, output_path)

    print(f"Results saved to: {output_path}")
    print(f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F-score: {results['f_score']:.4f}")

    return results


def find_generation_files(base_dir: Path) -> list[Path]:
    """Find all generation_texts_nltk_fallback.json files in the directory."""
    return list(base_dir.rglob('generation_texts_nltk_fallback.json'))


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate grammar correction models with ERRANT CLI')
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/home/inf148234/projects/ShortcutFM/generation_outputs/grammar',
        help='Base directory containing generation outputs'
    )
    parser.add_argument(
        '--file_path',
        type=str,
        help='Evaluate a specific file instead of all files'
    )

    args = parser.parse_args()

    # Check if ERRANT CLI tools are available
    for cmd in ['errant_parallel', 'errant_compare']:
        exit_code, _, _ = run_errant_command([cmd, '--help'])
        if exit_code != 0:
            print(f"Error: {cmd} is not available. Make sure ERRANT is properly installed.")
            return

    if args.file_path:
        # Evaluate single file
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist")
            return

        evaluate_single_file(file_path)
    else:
        # Evaluate all files
        base_dir = Path(args.base_dir)
        if not base_dir.exists():
            print(f"Error: Directory {base_dir} does not exist")
            return

        generation_files = find_generation_files(base_dir)

        if not generation_files:
            print(f"No generation_texts_nltk_fallback.json files found in {base_dir}")
            return

        print(f"Found {len(generation_files)} files to evaluate")

        all_results = []
        for i, file_path in enumerate(generation_files):
            print(f"\n--- Processing file {i+1}/{len(generation_files)} ---")
            try:
                results = evaluate_single_file(file_path)
                all_results.append({
                    'file_path': str(file_path),
                    'results': results
                })
            except Exception as e:
                print(f"Error evaluating {file_path}: {e}")
                continue

        # Save summary results
        summary_path = base_dir / 'errant_evaluation_cli_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")
        print(f"Successfully evaluated {len(all_results)}/{len(generation_files)} files")


if __name__ == '__main__':
    main()