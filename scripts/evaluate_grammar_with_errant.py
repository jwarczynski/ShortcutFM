#!/usr/bin/env python3
"""
Evaluate grammar correction models using ERRANT toolkit.

This script finds all generation_texts_nltk_fallback.json files in the grammar 
generation outputs directory and evaluates them using ERRANT to calculate
precision, recall, and F-score for grammatical error correction.
"""

import argparse
import json
import tempfile
from pathlib import Path

import errant


def load_generation_data(file_path: Path) -> list[dict[str, str]]:
    """Load generation data from JSON file."""
    with open(file_path, encoding='utf-8') as f:
        return json.load(f)


def write_temp_file(texts: list[str], suffix: str = '.txt') -> str:
    """Write texts to a temporary file and return the file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        for text in texts:
            f.write(text.strip() + '\n')
        return f.name


def extract_texts_from_data(data: list[dict[str, str]]) -> tuple[list[str], list[str], list[str]]:
    """Extract source, reference, and hypothesis texts from the data."""
    sources = [item['source'] for item in data]
    references = [item['reference'] for item in data]
    hypotheses = [item['hypothesis'] for item in data]

    return sources, references, hypotheses


def calculate_errant_scores(sources: list[str], references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Calculate ERRANT scores for the given texts."""
    # Initialize ERRANT annotator
    annotator = errant.load('en')

    # Process each sentence pair
    ref_edits = []
    hyp_edits = []

    for source, reference, hypothesis in zip(sources, references, hypotheses, strict=True):
        # Get reference edits (source -> reference)
        source_doc = annotator.parse(source)
        reference_doc = annotator.parse(reference)
        ref_alignment = annotator.align(source_doc, reference_doc)
        ref_sentence_edits = annotator.merge(ref_alignment)
        ref_edits.extend(ref_sentence_edits)

        # Get hypothesis edits (source -> hypothesis)
        hypothesis_doc = annotator.parse(hypothesis)
        hyp_alignment = annotator.align(source_doc, hypothesis_doc)
        hyp_sentence_edits = annotator.merge(hyp_alignment)
        hyp_edits.extend(hyp_sentence_edits)

    # Calculate metrics
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives

    # Convert edits to sets for comparison
    ref_edit_set = set()
    for edit in ref_edits:
        if edit.type != 'noop':  # Skip no-operation edits
            ref_edit_set.add((edit.o_start, edit.o_end, edit.c_start, edit.c_end, edit.type))

    hyp_edit_set = set()
    for edit in hyp_edits:
        if edit.type != 'noop':  # Skip no-operation edits
            hyp_edit_set.add((edit.o_start, edit.o_end, edit.c_start, edit.c_end, edit.type))

    # Calculate TP, FP, FN
    tp = len(ref_edit_set & hyp_edit_set)
    fp = len(hyp_edit_set - ref_edit_set)
    fn = len(ref_edit_set - hyp_edit_set)

    # Calculate precision, recall, F-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f_score': f_score,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_examples': len(sources)
    }


def save_errant_results(results: dict[str, float], output_path: Path) -> None:
    """Save ERRANT evaluation results to a JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)


def evaluate_single_file(file_path: Path) -> dict[str, float]:
    """Evaluate a single generation file with ERRANT."""
    print(f"Evaluating: {file_path}")

    # Load data
    data = load_generation_data(file_path)

    # Extract texts
    sources, references, hypotheses = extract_texts_from_data(data)

    # Calculate ERRANT scores
    results = calculate_errant_scores(sources, references, hypotheses)

    # Save results in the same directory
    output_path = file_path.parent / 'errant_evaluation.json'
    save_errant_results(results, output_path)

    print(f"Results saved to: {output_path}")
    print(f"Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F-score: {results['f_score']:.4f}")

    return results


def find_generation_files(base_dir: Path) -> list[Path]:
    """Find all generation_texts_nltk_fallback.json files in the directory."""
    return list(base_dir.rglob('generation_texts_nltk_fallback.json'))


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate grammar correction models with ERRANT')
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
        for file_path in generation_files:
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
        summary_path = base_dir / 'errant_evaluation_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")
        print(f"Successfully evaluated {len(all_results)}/{len(generation_files)} files")


if __name__ == '__main__':
    main()
