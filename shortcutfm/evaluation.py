"""
Evaluation module for generated sequences.
Extracted from scripts/evaluate_seq.py for integration with generation pipeline.

Uses NLTK corpus_bleu with smoothing for robust BLEU calculation that handles empty predictions.
"""

import json
import logging
from pathlib import Path
from typing import Any

import evaluate
import nltk
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from transformers import AutoTokenizer

from shortcutfm.decoding.text_processing import (
    extract_clean_predictions,
    extract_sources_and_references_from_input_texts,
    get_separator_token,
    process_batch_predictions,
)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

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
    sep_token = get_separator_token(tokenizer)

    # Split on separator token
    parts = text.split(sep_token)

    # Handle different formats based on number of parts
    if len(parts) < 2:
        raise ValueError(f"Input text does not contain a valid separator: {text}")

    # For Helsinki-NLP format: "source</s></s>target</s><pad>..."
    # For BERT format: "[CLS]source[SEP]target[SEP]<pad>..."
    if len(parts) >= 3:
        # Take first non-empty part as source and find target
        source = parts[0].strip()

        # Find the target - skip empty parts (from double separators)
        target = None
        for i in range(1, len(parts)):
            candidate = parts[i].strip()
            if candidate and not candidate.startswith('<pad>'):
                target = candidate
                break

        if target is None:
            target = parts[1].strip()  # Fallback to second part
    else:
        # Simple case with just source and target
        source, target = parts[0].strip(), parts[1].strip()

    # Remove special tokens
    special_tokens = {
        tokenizer.cls_token if hasattr(tokenizer, 'cls_token') else None,
        tokenizer.sep_token if hasattr(tokenizer, 'sep_token') else None,
        tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else None,
        tokenizer.bos_token if hasattr(tokenizer, 'bos_token') else None,
        tokenizer.pad_token if hasattr(tokenizer, 'pad_token') else None,
        tokenizer.unk_token if hasattr(tokenizer, 'unk_token') else None,
        tokenizer.mask_token if hasattr(tokenizer, 'mask_token') else None,
        '<pad>', '</s>', '[SEP]', '[CLS]', '[PAD]', '[UNK]', '[MASK]'
    }
    special_tokens = {t for t in special_tokens if t is not None}

    # Clean source and target
    source_tokens = []
    target_tokens = []

    for token in source.split():
        if token not in special_tokens and not token.startswith('<pad>') and token.strip():
            source_tokens.append(token)

    for token in target.split():
        if token not in special_tokens and not token.startswith('<pad>') and token.strip():
            target_tokens.append(token)

    source = " ".join(source_tokens).strip()
    target = " ".join(target_tokens).strip()

    return source, target


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


def compute_bleu_score(
    hypotheses: list[str],
    references: list[str],
    smoothing_method: int = 4
) -> float:
    """Compute BLEU score using NLTK corpus_bleu with smoothing.

    This function provides a robust BLEU computation that handles edge cases
    like empty predictions gracefully using NLTK's smoothing functions.

    Args:
        hypotheses: List of predicted texts
        references: List of reference texts
        smoothing_method: NLTK smoothing method to use (1-7, default: 4 - Chen & Cherry)

    Returns:
        BLEU score as a float
    """
    if not hypotheses or not references:
        logger.warning("Empty hypotheses or references provided to BLEU computation")
        return 0.0

    if len(hypotheses) != len(references):
        logger.warning(f"Mismatched lengths: {len(hypotheses)} hypotheses vs {len(references)} references")
        return 0.0

    # Tokenize for NLTK
    tokenized_hypotheses = [hyp.split() for hyp in hypotheses]
    tokenized_references = [[ref.split()] for ref in references]

    try:
        # Get smoothing function
        smoothing_function = getattr(SmoothingFunction(), f'method{smoothing_method}')

        bleu_score = corpus_bleu(
            tokenized_references,
            tokenized_hypotheses,
            smoothing_function=smoothing_function
        )

        logger.debug(f"NLTK BLEU score (method {smoothing_method}): {bleu_score}")
        return bleu_score

    except Exception as e:
        logger.error(f"Error computing NLTK BLEU score: {e}")
        return 0.0


def compute_bleu_from_batch(
    batch,
    predicted_tokens,
    tokenizer,
    use_fallback_processing: bool = False,
    smoothing_method: int = 4
) -> float:
    """Compute BLEU score directly from a batch and predicted tokens.

    This function is designed for use during training/validation where you have
    a batch object and predicted tokens.

    Args:
        batch: EncoderBatch containing input sequences and masks
        predicted_tokens: Predicted token IDs [batch_size, seq_len] or [batch_size, num_steps, seq_len]
        tokenizer: Tokenizer for decoding
        use_fallback_processing: Whether to use fallback processing for edge cases
        smoothing_method: NLTK smoothing method to use

    Returns:
        BLEU score as a float
    """
    try:
        # Extract texts using shared processing functions
        source_texts, reference_texts, predicted_texts = process_batch_predictions(
            batch, predicted_tokens, tokenizer, use_fallback_processing
        )

        # Compute BLEU score
        return compute_bleu_score(predicted_texts, reference_texts, smoothing_method)

    except Exception as e:
        logger.error(f"Error computing BLEU from batch: {e}")
        return 0.0


def compute_bleu_from_saved_outputs(
    inputs,
    predictions,
    tokenizer,
    use_fallback_processing: bool = False,
    smoothing_method: int = 4
) -> tuple[float, list[str], list[str], list[str]]:
    """Compute BLEU score from saved test outputs (as used in evaluate_generations).

    This function handles the multi-step prediction format saved by SaveTestOutputsCallback.

    Args:
        inputs: Input token sequences [batch_size, seq_len]
        predictions: Model predictions [batch_size, num_steps, seq_len]
        tokenizer: Tokenizer for decoding
        use_fallback_processing: Whether to use fallback processing for edge cases
        smoothing_method: NLTK smoothing method to use

    Returns:
        Tuple of (bleu_score, sources, references, hypotheses)
    """
    try:
        # Process inputs to get sources and references
        input_texts = [tokenizer.decode(seq, skip_special_tokens=False) for seq in inputs]
        sources, references = extract_sources_and_references_from_input_texts(input_texts, tokenizer)

        # Process predictions (take last step)
        hypotheses = extract_clean_predictions(
            torch.stack([seq[-1] for seq in predictions]),
            tokenizer,
            use_fallback_processing
        )

        # Compute BLEU score
        bleu_score = compute_bleu_score(hypotheses, references, smoothing_method)

        return bleu_score, sources, references, hypotheses

    except Exception as e:
        logger.error(f"Error computing BLEU from saved outputs: {e}")
        return 0.0, [], [], []


def save_evaluation_results(
    output_dir: Path,
    sources: list[str],
    references: list[str],
    hypotheses: list[str],
    metrics: dict[str, Any],
    suffix: str = "",
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

    texts_file = output_dir / f"generation_texts_{suffix}.json"
    with open(texts_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved generation texts to {texts_file}")

    # Save metrics
    metrics_file = output_dir / f"metrics_{suffix}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")


def evaluate_generations(
    output_dir: Path,
    tokenizer_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_fallback_processing: bool = False,
    suffix: str = "",
) -> dict[str, Any]:
    """Evaluate generated sequences using multiple metrics.

    Args:
        output_dir: Directory containing generation outputs
        tokenizer_name: Name of the tokenizer to use
        device: Device for evaluation computations
        use_fallback_processing: Whether to use fallback processing for empty predictions

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load and merge outputs
    inputs, predictions = load_and_merge_outputs(output_dir)

    # Compute BLEU score and extract texts using shared function
    bleu_score, sources, references, hypotheses = compute_bleu_from_saved_outputs(
        inputs, predictions, tokenizer, use_fallback_processing
    )

    # Debug information
    logger.info(f"Number of references: {len(references)}")
    logger.info(f"Number of hypotheses: {len(hypotheses)}")

    # Additional debugging for empty hypotheses
    non_empty_hyp = [h for h in hypotheses if h.strip()]
    logger.info(f"Non-empty hypotheses: {len(non_empty_hyp)}")

    # Report on fallback processing effectiveness if enabled
    if use_fallback_processing:
        original_empty = sum(1 for h in hypotheses if not h.strip())
        logger.info(f"Empty hypotheses after processing: {original_empty}")

    # Calculate metrics
    metrics = {}

    # BLEU score (already computed)
    metrics["bleu"] = {"bleu": bleu_score}
    logger.info(f"NLTK BLEU score (with smoothing): {bleu_score}")

    # ROUGE scores - use original string format
    try:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(predictions=hypotheses, references=references)
        if rouge_scores:
            metrics.update({
                "rouge1": rouge_scores.get("rouge1", 0.0),
                "rouge2": rouge_scores.get("rouge2", 0.0),
                "rougeL": rouge_scores.get("rougeL", 0.0),
            })
        logger.info(f"ROUGE scores: {rouge_scores}")
    except Exception as e:
        logger.error(f"Error computing ROUGE scores: {e}")
        metrics.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})

    # BERTScore - use original string format
    try:
        bert_score = evaluate.load("bertscore")
        bert_scores = bert_score.compute(
            predictions=hypotheses,
            references=references,
            model_type="microsoft/deberta-xlarge-mnli",
        )
        if bert_scores and "precision" in bert_scores:
            metrics["bertscore"] = {
                "precision": sum(bert_scores["precision"]) / len(bert_scores["precision"]),
                "recall": sum(bert_scores["recall"]) / len(bert_scores["recall"]),
                "f1": sum(bert_scores["f1"]) / len(bert_scores["f1"]),
            }
        logger.info(f"BERTScore: {metrics.get('bertscore', 'Failed')}")
    except Exception as e:
        logger.error(f"Error computing BERTScore: {e}")
        metrics["bertscore"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

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
        suffix=suffix
    )

    return metrics
