"""Decode raw model outputs without any postprocessing.

Loads inputs_rank0.pt and predictions_rank0.pt, decodes them with the tokenizer
using skip_special_tokens=False, and saves the raw decoded strings.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer

gen_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "generation_outputs/qqp/scut_768/run_ll9cnmi5/scut=2048/seed_44"
)

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

inputs = torch.load(gen_dir / "inputs_rank0.pt", map_location="cpu")
preds = torch.load(gen_dir / "predictions_rank0.pt", map_location="cpu")

print(f"Inputs shape: {inputs.shape}")
print(f"Predictions shape: {preds.shape}")

results = []
for i in range(len(inputs)):
    input_raw = tok.decode(inputs[i], skip_special_tokens=False)
    # predictions may be [num_steps, seq_len] — take last step
    if preds[i].dim() == 2:
        pred_tokens = preds[i][-1]
    else:
        pred_tokens = preds[i]
    pred_raw = tok.decode(pred_tokens, skip_special_tokens=False)

    results.append({
        "idx": i,
        "input_raw": input_raw,
        "prediction_raw": pred_raw,
        "input_tokens": inputs[i].tolist(),
        "pred_tokens": pred_tokens.tolist(),
    })

    if i < 5:
        print(f"\n=== Example {i} ===")
        print(f"INPUT RAW:      {input_raw[:200]}")
        print(f"PREDICTION RAW: {pred_raw[:200]}")
        # Check if they're identical
        print(f"IDENTICAL: {input_raw == pred_raw}")
        print(f"TOKEN MATCH: {(inputs[i] == pred_tokens).all().item()}")

out_path = gen_dir / "raw_decoded_outputs.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved {len(results)} raw outputs to {out_path}")

# Summary stats
identical_count = sum(1 for r in results if r["input_raw"] == r["prediction_raw"])
token_match = sum(1 for i in range(len(inputs)) for _ in [1]
               if (inputs[i] == (preds[i][-1] if preds[i].dim() == 2 else preds[i])).all().item())
print(f"Identical raw strings: {identical_count}/{len(results)}")
print(f"Identical token IDs: {token_match}/{len(results)}")
