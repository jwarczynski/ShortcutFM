import argparse
from pathlib import Path

import numpy as np
import torch
from bert_score import score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


def load_and_merge_outputs(output_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and merge outputs from all ranks."""
    inputs, predictions = [], []

    # Load all rank files
    for file in output_dir.glob("inputs_rank*.pt"):
        rank_inputs = torch.load(file)
        rank_predictions = torch.load(file.parent / f"predictions_{file.name.split('_')[1]}")
        inputs.append(rank_inputs)
        predictions.append(rank_predictions)

    return torch.cat(inputs), torch.cat(predictions)


def process_sequence(text: str, tokenizer) -> tuple[str, str]:
    """Process a sequence by splitting on separator and removing special tokens."""
    # Split on first separator token
    parts = text.split(tokenizer.sep_token)
    if len(parts) < 2:
        return text, text

    source, target = parts[0], parts[1]

    # Remove special tokens
    special_tokens = {
        tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
        tokenizer.unk_token, tokenizer.mask_token
    }
    special_tokens = {t for t in special_tokens if t is not None}

    # Clean source and target
    source = ' '.join(token for token in source.split() if token not in special_tokens).strip()
    target = ' '.join(token for token in target.split() if token not in special_tokens).strip()

    return source, target


def process_prediction(text: str, tokenizer) -> str:
    """Process a prediction by taking text up to first separator and removing special tokens."""
    # Split on separator token and take target part
    parts = text.split(tokenizer.sep_token)
    if len(parts) < 2:
        return text

    target = parts[1]
    # Take only up to the next separator if it exists
    if tokenizer.sep_token in target:
        target = target.split(tokenizer.sep_token)[0]

    # Remove special tokens
    special_tokens = {
        tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token,
        tokenizer.unk_token, tokenizer.mask_token
    }
    special_tokens = {t for t in special_tokens if t is not None}

    return ' '.join(token for token in target.split() if token not in special_tokens).strip()


def compute_distinct_ngrams(texts: list[str], n: int) -> float:
    """Compute distinct n-grams ratio."""
    all_ngrams = []
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def evaluate_generations(
        output_dir: Path,
        tokenizer_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Evaluate generated sequences using multiple metrics."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load and merge outputs
    inputs, predictions = load_and_merge_outputs(output_dir)

    # Process inputs
    input_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in inputs]
    sources, references = zip(*[process_sequence(text, tokenizer) for text in input_texts])
    references = list(references)  # Convert tuple to list

    # Process predictions (take last step predictions)
    pred_texts = [tokenizer.decode(seq[-1], skip_special_tokens=False) for seq in predictions]
    hypotheses = [process_prediction(text, tokenizer) for text in pred_texts]

    # Compute ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {
        'rouge1': [], 'rouge2': [], 'rougeL': []
    }
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    rouge_scores = {k: np.mean(v) for k, v in rouge_scores.items()}

    # Compute BLEU scores with smoothing
    references_tokens = [[ref.split()] for ref in references]
    hypotheses_tokens = [hyp.split() for hyp in hypotheses]

    # Use different n-gram orders with smoothing
    chencherry = SmoothingFunction()
    bleu_scores = {}
    for n in range(1, 5):
        bleu_scores[f'bleu-{n}'] = corpus_bleu(
            references_tokens,
            hypotheses_tokens,
            weights=tuple([1.0 / n] * n),  # Equal weights for n-grams
            smoothing_function=chencherry.method3  # Use method3 smoothing
        )

    # Compute BERTScore
    P, R, F1 = score(hypotheses, references, lang="en", device=device)
    bert_score = {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

    # Compute distinct n-grams
    distinct_scores = {
        f'distinct-{n}': compute_distinct_ngrams(hypotheses, n)
        for n in range(1, 5)
    }

    # Combine all metrics
    metrics = {
        **rouge_scores,
        **bleu_scores,
        **{f'bertscore_{k}': v for k, v in bert_score.items()},
        **distinct_scores
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated sequences')
    parser.add_argument('output_dir', type=str, help='Directory containing generation outputs')
    parser.add_argument('--tokenizer', type=str, default='bert-base-uncased',
                        help='Tokenizer to use for decoding')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for BERTScore computation')

    args = parser.parse_args()

    metrics = evaluate_generations(Path(args.output_dir), args.tokenizer, args.device)

    # Print metrics
    print("\nEvaluation Results:")
    print("==================")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == '__main__':
    main() 