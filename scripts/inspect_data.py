"""Quick inspection of tokenized dataset to check source/target/mask alignment."""
import sys
sys.path.insert(0, ".")

from datasets import Dataset
from transformers import AutoTokenizer

ds = Dataset.load_from_disk("datasets/tokenized/bert-base-uncased/QQP-Official/test")
print("Columns:", ds.column_names)
print("Num examples:", len(ds))

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

for idx in [0, 1, 2]:
    ex = ds[idx]
    ids = ex["input_ids"]
    mask = ex["input_mask"]

    # Find transition from 0 to 1
    tr = -1
    for i in range(len(mask) - 1):
        if mask[i] == 0 and mask[i + 1] == 1:
            tr = i + 1
            break

    print(f"\n=== Example {idx} ===")
    print(f"Mask transition at: {tr}")
    print(f"Source (mask=0): {tok.decode(ids[:tr], skip_special_tokens=False)}")
    print(f"Target (mask=1): {tok.decode(ids[tr:], skip_special_tokens=False)}")
    print(f"Mask 0s: {sum(1 for m in mask if m == 0)}, 1s: {sum(1 for m in mask if m == 1)}")

    # Also show the raw token IDs around the transition
    print(f"IDs around transition [{tr-3}:{tr+3}]: {ids[max(0,tr-3):tr+3]}")
    print(f"Mask around transition: {mask[max(0,tr-3):tr+3]}")
    print(f"Tokens around transition: {tok.convert_ids_to_tokens(ids[max(0,tr-3):tr+3])}")
