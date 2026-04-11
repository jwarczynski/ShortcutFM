"""
Text processing functions for model predictions and input sequences.

This module provides shared utilities for processing raw model outputs and input sequences
across training validation and evaluation pipelines.
"""

import logging

from torch import Tensor
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def get_separator_token(tokenizer: PreTrainedTokenizer) -> str:
    """Get the appropriate separator token for the tokenizer."""
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        return tokenizer.sep_token
    elif hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        return tokenizer.eos_token
    else:
        # Fallback to common separators
        return "</s>"


def get_cls_token(tokenizer: PreTrainedTokenizer) -> str | None:
    """Get the appropriate CLS token for the tokenizer."""
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        return tokenizer.cls_token
    elif hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        return tokenizer.bos_token
    else:
        return None


def get_special_tokens_set(tokenizer: PreTrainedTokenizer) -> set[str]:
    """Get a comprehensive set of special tokens for the tokenizer."""
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
    return {t for t in special_tokens if t is not None}


def extract_sources_and_references_from_batch(
    batch,
    tokenizer: PreTrainedTokenizer
) -> tuple[list[str], list[str]]:
    """Extract source and reference texts from a batch.

    Args:
        batch: EncoderBatch containing sequences and masks
        tokenizer: Tokenizer for decoding

    Returns:
        Tuple of (source_texts, reference_texts)
    """
    # Split sequences into source and reference using input_mask
    source_tokens = batch.seqs.clone()
    reference_tokens = batch.seqs.clone()

    # Zero out reference/source parts based on input_mask
    source_tokens[batch.input_ids_mask == 1] = tokenizer.pad_token_id
    reference_tokens[batch.input_ids_mask == 0] = tokenizer.pad_token_id

    # Decode each part separately
    source_texts = tokenizer.batch_decode(source_tokens, skip_special_tokens=True)
    reference_texts = tokenizer.batch_decode(reference_tokens, skip_special_tokens=True)

    return source_texts, reference_texts


def extract_sources_and_references_from_input_texts(
    input_texts: list[str],
    tokenizer: PreTrainedTokenizer
) -> tuple[list[str], list[str]]:
    """Extract source and reference texts from decoded input texts.

    This function processes input texts that contain both source and reference
    separated by special tokens (as used in evaluate_generations).

    Args:
        input_texts: List of decoded input texts containing source and reference
        tokenizer: Tokenizer for special token handling

    Returns:
        Tuple of (source_texts, reference_texts)
    """
    sources, references = [], []

    for text in input_texts:
        source, reference = _process_input_sequence(text, tokenizer)
        sources.append(source)
        references.append(reference)

    return sources, references


def _process_input_sequence(text: str, tokenizer: PreTrainedTokenizer) -> tuple[str, str]:
    """Process a single input sequence by splitting on separator and removing special tokens."""
    sep_token = get_separator_token(tokenizer)
    special_tokens = get_special_tokens_set(tokenizer)

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

    # Clean source and target
    source = _clean_text_tokens(source, special_tokens)
    target = _clean_text_tokens(target, special_tokens)

    return source, target


def _clean_text_tokens(text: str, special_tokens: set[str]) -> str:
    """Clean text by removing special tokens and padding."""
    tokens = []
    for token in text.split():
        if token not in special_tokens and not token.startswith('<pad>') and token.strip():
            tokens.append(token)
    return " ".join(tokens).strip()


def extract_clean_predictions(
    predicted_tokens: Tensor,
    tokenizer: PreTrainedTokenizer,
    use_fallback_processing: bool = False
) -> list[str]:
    """Extract clean predicted text from predicted tokens.

    Args:
        predicted_tokens: Tensor of predicted token IDs [batch_size, seq_len] or [batch_size, num_steps, seq_len]
        tokenizer: Tokenizer for decoding
        use_fallback_processing: Whether to use fallback processing for edge cases

    Returns:
        List of clean predicted texts
    """
    # Handle multi-step predictions (take last step)
    if predicted_tokens.dim() == 3:
        predicted_tokens = predicted_tokens[:, -1, :]

    # Decode predictions
    predicted_texts = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=False)

    # Process each prediction
    clean_predictions = []
    for text in predicted_texts:
        clean_text = _process_prediction_text(text, tokenizer, use_fallback_processing)
        clean_predictions.append(clean_text)

    return clean_predictions


def _process_prediction_text(
    text: str,
    tokenizer: PreTrainedTokenizer,
    use_fallback_processing: bool = False
) -> str:
    """Process a single prediction text by extracting the generated part."""
    sep_token = get_separator_token(tokenizer)
    cls_token = get_cls_token(tokenizer)
    special_tokens = get_special_tokens_set(tokenizer)

    # Try different formats based on tokenizer type
    if cls_token and cls_token in text:
        # BERT-style format: [CLS] input [SEP] [SEP] [CLS] output [SEP] [PAD]...
        cls_parts = text.split(cls_token)
        if len(cls_parts) >= 3:
            output_part = cls_parts[2].split(sep_token)[0]  # Get part before first SEP after second CLS
        else:
            output_part = cls_parts[-1].split(sep_token)[0]  # Fallback
    else:
        # Helsinki-NLP style format: source</s></s>target</s><pad>...
        parts = text.split(sep_token)

        if len(parts) >= 3:
            output_part = parts[2]  # Take the third part as the generated output
        else:
            output_part = parts[-1]  # Fallback to last part

    # Clean and filter tokens
    cleaned_tokens = []
    for token in output_part.split():
        # Skip various forms of padding and special tokens
        if (token not in special_tokens and
            not token.startswith('<pad>') and
            not token.endswith('<pad>') and
            token not in ['<pad>', '</s>', '[PAD]', '[SEP]', '[CLS]'] and
            token.strip()
        ):
            cleaned_tokens.append(token)

    result = " ".join(cleaned_tokens).strip()

    # Apply fallback processing if result is empty and fallback is enabled
    if not result and use_fallback_processing:
        result = _fallback_prediction_processing(text, tokenizer, sep_token, cls_token, special_tokens)

    return result


def _fallback_prediction_processing(
    text: str,
    tokenizer: PreTrainedTokenizer,
    sep_token: str,
    cls_token: str | None,
    special_tokens: set
) -> str:
    """Fallback processing for predictions that result in empty strings.

    This function implements smart patterns based on analysis of failed predictions:
    1. Try extracting from different separator positions
    2. Look for content between any special tokens
    3. Extract the longest non-special token sequence
    4. Use less aggressive token filtering

    Args:
        text: Raw prediction text
        tokenizer: Tokenizer instance
        sep_token: Separator token
        cls_token: CLS token (if available)
        special_tokens: Set of special tokens to filter

    Returns:
        Best effort extracted text or empty string if nothing found
    """
    logger.debug(f"Applying fallback processing to: {repr(text)}")

    # Strategy 1: Try separator-split parts, but skip likely source parts
    parts = text.split(sep_token)
    if len(parts) > 1:
        # For BERT-style: [CLS] input [SEP] [SEP] [CLS] output [SEP] [PAD]
        # parts[0] = "[CLS] input", parts[1] = "", parts[2] = "[CLS] output", parts[3] = "[PAD]..."
        # We want to skip the first part (source) and start from index 1 or 2
        start_idx = 1 if len(parts) > 2 else 1  # Skip the source part

        for i in range(start_idx, len(parts)):
            part = parts[i]
            candidate = _clean_tokens_lenient(part, special_tokens)
            if candidate:
                logger.debug(f"Fallback strategy 1 success: {repr(candidate)}")
                return candidate

    # Strategy 2: If using CLS tokens, try CLS-split parts (but skip likely source parts)
    if cls_token and cls_token in text:
        cls_parts = text.split(cls_token)
        # For BERT: parts would be ["", " input [SEP] [SEP] ", " output [SEP] [PAD]..."]
        # Skip first empty part and potential source part
        start_idx = 2 if len(cls_parts) > 2 else 1

        for i in range(start_idx, len(cls_parts)):
            part = cls_parts[i]
            # Try the part directly
            candidate = _clean_tokens_lenient(part, special_tokens)
            if candidate and not _looks_like_source_content(candidate, cls_parts[1] if len(cls_parts) > 1 else ""):
                logger.debug(f"Fallback strategy 2a success: {repr(candidate)}")
                return candidate

            # Also try sub-parts after splitting by separator
            sub_parts = part.split(sep_token)
            for _, sub_part in enumerate(sub_parts):
                candidate = _clean_tokens_lenient(sub_part, special_tokens)
                if candidate and not _looks_like_source_content(candidate, cls_parts[1] if len(cls_parts) > 1 else ""):
                    logger.debug(f"Fallback strategy 2b success: {repr(candidate)}")
                    return candidate

    # Strategy 3: Extract longest sequence of non-special tokens
    all_tokens = text.split()
    longest_sequence = []
    current_sequence = []

    for token in all_tokens:
        if (token not in special_tokens and
            not token.startswith('<pad>') and
            not token.endswith('<pad>') and
            token.strip()):
            current_sequence.append(token)
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = []

    # Check final sequence
    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence

    if longest_sequence:
        result = " ".join(longest_sequence).strip()
        logger.debug(f"Fallback strategy 3 success: {repr(result)}")
        return result

    # Strategy 4: Very lenient filtering - keep anything that looks like real text
    lenient_tokens = []
    for token in all_tokens:
        # Only filter out obvious special tokens and padding
        if (token not in ['<pad>', '[PAD]', '</s>', '[SEP]', '[CLS]', '[UNK]', '[MASK]'] and
            not token.startswith('<pad>') and
            not token.endswith('<pad>') and
            token.strip() and
            len(token.strip()) > 0):
            lenient_tokens.append(token)

    if lenient_tokens:
        result = " ".join(lenient_tokens).strip()
        logger.debug(f"Fallback strategy 4 success: {repr(result)}")
        return result

    logger.debug("All fallback strategies failed")
    return ""


def _clean_tokens_lenient(text: str, special_tokens: set) -> str:
    """Clean tokens with more lenient filtering than the main processing."""
    tokens = []
    for token in text.split():
        # More lenient - only filter obvious special tokens
        if (token not in special_tokens and
            token not in ['<pad>', '[PAD]', '</s>', '[SEP]', '[CLS]'] and
            not token.startswith('<pad>') and
            token.strip()):
            tokens.append(token)
    return " ".join(tokens).strip()


def _looks_like_source_content(candidate: str, potential_source: str) -> bool:
    """Check if candidate text looks like it might be source content rather than generated output.
    
    Args:
        candidate: The text candidate we're considering as prediction
        potential_source: The potential source text to compare against
        
    Returns:
        True if candidate appears to be source content, False otherwise
    """
    if not candidate or not potential_source:
        return False

    # Clean both texts for comparison
    candidate_clean = candidate.lower().strip()
    source_clean = potential_source.lower().strip()

    # Remove special tokens from source for cleaner comparison
    for token in ['[cls]', '[sep]', '[pad]', '<pad>', '</s>']:
        source_clean = source_clean.replace(token, ' ')
    source_clean = ' '.join(source_clean.split())  # Normalize whitespace

    if not source_clean:
        return False

    # Check for high overlap with source
    candidate_words = set(candidate_clean.split())
    source_words = set(source_clean.split())

    if len(candidate_words) == 0:
        return False

    # If more than 70% of candidate words appear in source, it's likely source content
    overlap = len(candidate_words & source_words)
    overlap_ratio = overlap / len(candidate_words)

    return overlap_ratio > 0.7


def process_batch_predictions(
    batch,
    predicted_tokens: Tensor,
    tokenizer: PreTrainedTokenizer,
    use_fallback_processing: bool = False
) -> tuple[list[str], list[str], list[str]]:
    """Process a batch and extract source, reference, and prediction texts.

    Args:
        batch: EncoderBatch containing input sequences and masks
        predicted_tokens: Predicted token IDs [batch_size, seq_len] or [batch_size, num_steps, seq_len]
        tokenizer: Tokenizer for decoding
        use_fallback_processing: Whether to use fallback processing for edge cases

    Returns:
        Tuple of (source_texts, reference_texts, predicted_texts)
    """
    # Extract sources and references from batch
    source_texts, reference_texts = extract_sources_and_references_from_batch(batch, tokenizer)

    # Extract clean predictions
    predicted_texts = extract_clean_predictions(predicted_tokens, tokenizer, use_fallback_processing)

    return source_texts, reference_texts, predicted_texts


# Compatibility wrapper for notebooks
def process_prediction(text: str, tokenizer, use_fallback_processing: bool = False) -> str:
    """Compatibility wrapper for _process_prediction_text.

    This function maintains backward compatibility for notebooks and other code
    that previously imported process_prediction from evaluation.py.
    """
    return _process_prediction_text(text, tokenizer, use_fallback_processing)
