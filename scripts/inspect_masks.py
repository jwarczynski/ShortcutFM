"""Inspect the mask composition: how much is source, target, and padding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
ds = Dataset.load_from_disk("datasets/tokenized/bert-base-uncased/QQP-Official/test")

print(f"Seq length: {len(ds[0]['input_ids'])}")
print(f"Pad token id: {tok.pad_token_id}")  # should be 0
print(f"Sep token id: {tok.sep_token_id}")  # should be 102
print(f"CLS token id: {tok.cls_token_id}")  # should be 101

for i in range(min(5, len(ds))):
    ex = ds[i]
    ids = ex["input_ids"]
    mask = ex["input_mask"]
    pad_mask = ex["padding_mask"]
    seq_len = len(ids)

    # Count actual content vs padding in input_ids
    num_pad_tokens = sum(1 for t in ids if t == tok.pad_token_id)
    num_content_tokens = seq_len - num_pad_tokens

    # Count mask values
    mask_zeros = sum(1 for m in mask if m == 0)  # source part
    mask_ones = sum(1 for m in mask if m == 1)   # "target" part (includes real target + padding)

    # Find where actual target ends and padding begins
    # The sequence is: [CLS] src [SEP] [CLS] trg [SEP] [PAD]...
    # Find last non-pad token
    last_content_idx = seq_len - 1
    while last_content_idx >= 0 and ids[last_content_idx] == tok.pad_token_id:
        last_content_idx -= 1

    real_target_len = last_content_idx + 1 - mask_zeros  # content after source
    padding_len = seq_len - (last_content_idx + 1)

    # What the model sees as "to denoise" (mask=1)
    # Of those mask=1 positions, how many are real target vs padding?
    target_positions_that_are_pad = padding_len  # all padding has mask=1
    target_positions_that_are_real = mask_ones - padding_len

    print(f"\n=== Example {i} ===")
    print(f"  Total seq: {seq_len}")
    print(f"  Source (mask=0): {mask_zeros} tokens")
    print(f"  mask=1 total: {mask_ones} tokens")
    print(f"    Real target: {target_positions_that_are_real} tokens")
    print(f"    Padding:     {target_positions_that_are_pad} tokens")
    print(f"  Ratio real_target/mask1: {target_positions_that_are_real/mask_ones:.1%}")
    print(f"  Ratio padding/mask1:     {target_positions_that_are_pad/mask_ones:.1%}")

    # Decode the parts
    src_text = tok.decode(ids[:mask_zeros], skip_special_tokens=True)
    trg_text = tok.decode(ids[mask_zeros:last_content_idx+1], skip_special_tokens=True)
    print(f"  Source: {src_text[:80]}")
    print(f"  Target: {trg_text[:80]}")

# Aggregate stats
print("\n=== Aggregate over full dataset ===")
total_mask1 = 0
total_real_target = 0
total_padding = 0
for i in range(len(ds)):
    ex = ds[i]
    ids = ex["input_ids"]
    mask = ex["input_mask"]
    seq_len = len(ids)
    mask_zeros = sum(1 for m in mask if m == 0)
    mask_ones = sum(1 for m in mask if m == 1)
    last_content_idx = seq_len - 1
    while last_content_idx >= 0 and ids[last_content_idx] == tok.pad_token_id:
        last_content_idx -= 1
    padding_len = seq_len - (last_content_idx + 1)
    total_mask1 += mask_ones
    total_real_target += (mask_ones - padding_len)
    total_padding += padding_len

print(f"  Avg mask=1 positions: {total_mask1/len(ds):.1f}")
print(f"  Avg real target:      {total_real_target/len(ds):.1f}")
print(f"  Avg padding:          {total_padding/len(ds):.1f}")
print(f"  Ratio real_target/mask1: {total_real_target/total_mask1:.1%}")
print(f"  Ratio padding/mask1:     {total_padding/total_mask1:.1%}")
