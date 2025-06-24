import argparse
import json
import logging
from pathlib import Path

import evaluate
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    # print(f"Parts: {parts}")
    if len(parts) < 3:
        raise ValueError(f"Input text does not contain a valid separator: {text}")
    if len(parts) == 3:
        # If there are exactly three parts, use them as source and target
        source, target = parts[0], parts[1]
    elif len(parts) == 4:
        # If there are four parts, there were 2 PAD tokens
        source, target = parts[0], parts[2]
    else:
        # If there are more than three parts, raise an error
        raise ValueError(f"Input text contains too many parts: {parts}")

    # Remove special tokens
    special_tokens = {
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token,
        tokenizer.unk_token,
        tokenizer.mask_token,
    }
    special_tokens = {t for t in special_tokens if t is not None}

    # Clean source and target
    source = " ".join(token for token in source.split() if token not in special_tokens).strip()
    target = " ".join(token for token in target.split() if token not in special_tokens).strip()

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
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.pad_token,
        tokenizer.unk_token,
        tokenizer.mask_token,
    }
    special_tokens = {t for t in special_tokens if t is not None}

    return " ".join(token for token in target.split() if token not in special_tokens).strip()


def compute_distinct_ngrams(texts: list[str], n: int) -> float:
    """Compute distinct n-grams ratio."""
    all_ngrams = []
    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def evaluate_generations(
    output_dir: Path,
    tokenizer_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Evaluate generated sequences using multiple metrics."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load and merge outputs
    inputs, predictions = load_and_merge_outputs(output_dir)

    # Process inputs
    input_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in inputs]
    sources, references = zip(*[process_sequence(text, tokenizer) for text in input_texts], strict=False)
    references = list(references)  # Convert tuple to list

    # Process predictions (take last step predictions)
    pred_texts = [tokenizer.decode(seq[-1], skip_special_tokens=False) for seq in predictions]
    hypotheses = [process_prediction(text, tokenizer) for text in pred_texts]

    # Calculate metrics
    metrics = {}

    # BLEU score
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=hypotheses, references=[[ref] for ref in references])
    metrics["bleu"] = bleu_score
    logger.info(f"BLEU score: {bleu_score}")

    # ROUGE scores
    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=hypotheses, references=references)
    metrics.update(rouge_scores)
    logger.info(f"ROUGE scores: {rouge_scores}")

    # BERTScore
    bert_score = evaluate.load("bertscore")
    bert_scores = bert_score.compute(
        predictions=hypotheses,
        references=references,
        model_type="microsoft/deberta-xlarge-mnli",
    )
    metrics["bertscore"] = {
        "precision": sum(bert_scores["precision"]) / len(bert_scores["precision"]),
        "recall": sum(bert_scores["recall"]) / len(bert_scores["recall"]),
        "f1": sum(bert_scores["f1"]) / len(bert_scores["f1"]),
    }
    logger.info(f"BERTScore: {metrics['bertscore']}")

    # Distinct n-grams
    for n in [1, 2, 3, 4]:
        distinct_score = compute_distinct_ngrams(hypotheses, n)
        metrics[f"distinct_{n}"] = distinct_score
        logger.info(f"Distinct-{n}: {distinct_score}")

    # Save all results
    save_evaluation_results(
        output_dir=Path(output_dir),
        sources=sources,
        references=references,
        hypotheses=hypotheses,
        metrics=metrics,
    )

    return metrics


def save_evaluation_results(
    output_dir: Path,
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    metrics: dict,
):
    """Save evaluation results and texts to files.

    Args:
        output_dir: Directory to save results
        sources: List of source texts
        references: List of reference texts
        hypotheses: List of generated texts
        metrics: Dictionary of evaluation metrics

    """
    # Save texts
    texts = [
        {"source": src, "reference": ref, "hypothesis": hyp}
        for src, ref, hyp in zip(sources, references, hypotheses, strict=False)
    ]

    texts_file = output_dir / "generation_texts.json"
    with open(texts_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved generation texts to {texts_file}")

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated sequences")
    parser.add_argument("output_dir", type=str, help="Directory containing generation outputs")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use for decoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for BERTScore computation",
    )

    args = parser.parse_args()

    metrics = evaluate_generations(Path(args.output_dir), args.tokenizer, args.device)

    # Print metrics
    print("\nEvaluation Results:")
    print("==================")
    for metric, value in metrics.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for sub_metric, sub_value in value.items():
                if isinstance(sub_value, list):  # Handle lists (like 'precisions')
                    print(f"  {sub_metric}: {[round(x, 4) for x in sub_value]}")
                else:  # Handle single floats
                    print(f"  {sub_metric}: {sub_value:.4f}")
        else:
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
