"""Check if training data has src != trg, and check the raw columns."""
import sys
sys.path.insert(0, ".")

from datasets import Dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

for split_name, path in [
    ("train", "datasets/tokenized/bert-base-uncased/QQP-Official/train"),
    ("test", "datasets/tokenized/bert-base-uncased/QQP-Official/test"),
]:
    ds = Dataset.load_from_disk(path)
    print(f"\n=== {split_name} ({len(ds)} examples) ===")
    print(f"Columns: {ds.column_names}")

    # Check if input_id_x == input_id_y for any examples
    same_count = 0
    for i in range(min(len(ds), 1000)):
        ex = ds[i]
        if ex["input_id_x"] == ex["input_id_y"]:
            same_count += 1
            if same_count <= 3:
                print(f"  SAME at idx {i}: {tok.decode(ex['input_id_x'], skip_special_tokens=True)}")

    print(f"  src == trg count: {same_count}/{min(len(ds), 1000)}")

    # Show a few examples of src vs trg
    for i in [0, 1, 2]:
        ex = ds[i]
        src = tok.decode(ex["input_id_x"], skip_special_tokens=True)
        trg = tok.decode(ex["input_id_y"], skip_special_tokens=True)
        print(f"  [{i}] src: {src}")
        print(f"  [{i}] trg: {trg}")
        print(f"  [{i}] same? {ex['input_id_x'] == ex['input_id_y']}")
