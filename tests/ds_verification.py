# %%
import os
from functools import partial

from transformers import AutoTokenizer

from shortcutfm.__main__ import parse_config

# %%
os.chdir("..")
# %%
cfg = parse_config("configs/training/qqp.yaml", [])
# %%
from torch.utils.data import DataLoader

from datasets import Dataset
from shortcutfm.batch import collate
from shortcutfm.text_datasets import TextDataset

if __name__ == "__main__":
    configured_collate = partial(
        collate,
        mark_first_padding=cfg.padding_strategy.mark_first_padding,
        mark_second_padding=cfg.padding_strategy.mark_second_padding,
    )
    train_ds = Dataset.load_from_disk(cfg.training_data_path)
    train_text_ds = TextDataset(train_ds)
    train = DataLoader(
        train_text_ds,
        batch_size=cfg.batch_size,
        collate_fn=configured_collate,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
    )
    # %%
    len(train)
    # %%
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # %%
    for batch in train:
        texts = tokenizer.batch_decode(batch.seqs, skip_special_tokens=False)

        for idx, (text, seq, input_mask, pad_mask) in enumerate(
            zip(texts, batch.seqs, batch.input_ids_mask, batch.padding_mask, strict=False)
        ):
            print(f"\nExample {idx + 1}:")
            print(f"Decoded full sequence:\n{text}\n")

            # Compute loss mask
            loss_mask = pad_mask * input_mask

            # Token IDs and words belonging to the source sequence (input part)
            src_token_ids = seq[(input_mask == 0).bool()].tolist()
            decoded_src_tokens = tokenizer.batch_decode(src_token_ids, skip_special_tokens=False)

            print(f"Token IDs (Source Sequence):\n{src_token_ids}\n")
            print(f"Decoded words (Source Sequence):\n{decoded_src_tokens}\n")

            # Token IDs and words contributing to loss (target part)
            loss_token_ids = seq[loss_mask.bool()].tolist()
            decoded_loss_tokens = tokenizer.batch_decode(loss_token_ids, skip_special_tokens=False)

            print(f"Token IDs contributing to loss:\n{loss_token_ids}\n")
            print(f"Decoded words contributing to loss:\n{decoded_loss_tokens}\n")

        break  # Process only the first batch for verification
    # %%
