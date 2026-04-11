import argparse
import sys
from pathlib import Path

import datasets
from datasets import DatasetDict

# Ensure project root (parent of scripts/) is on sys.path before importing project modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from shortcutfm.text_datasets import get_corpus, get_webnlg_tokenize_fn, helper_tokenize  # noqa: E402
from shortcutfm.tokenizer import MyTokenizer  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize dataset and save to disk.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--vocab", type=str, default="bert", help="Vocabulary type")
    parser.add_argument("--config_name", type=str, default="bert-base-uncased", help="Model config name")
    parser.add_argument("--max_seq_length", type=int, default=128)

    args = parser.parse_args()

    config_name = args.config_name.split("/")[-1] if "/" in args.config_name else args.config_name
    args.data_dir = f"datasets/raw/{args.dataset}"
    save_path = f"datasets/tokenized/{config_name}/{args.dataset}"

    # Initialize tokenizer
    args.tokenizer_config_name = args.config_name
    tokenizer = MyTokenizer(args, True)

    if (args.dataset).lower() == "webnlg":
        data = datasets.load_dataset("GEM/web_nlg", "en")
        # drop all features except 'input' and 'target'
        data = data.map(
            lambda x: {"input": x["input"], "target": x["target"]},
            remove_columns=[col for col in data["train"].column_names if col not in ["input", "target"]],
        )

        train_corpus = helper_tokenize(
            data["train"],
            tokenizer,
            args.max_seq_length,
            from_dict=False,
            tokenize_function=get_webnlg_tokenize_fn(tokenizer.tokenizer),
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask'],
        train_corpus = train_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in train_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        val_corpus = helper_tokenize(
            data["validation"],
            tokenizer,
            args.max_seq_length,
            from_dict=False,
            tokenize_function=get_webnlg_tokenize_fn(tokenizer.tokenizer),
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask']
        val_corpus = val_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in val_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        test_corpus = helper_tokenize(
            data["test"],
            tokenizer,
            args.max_seq_length,
            from_dict=False,
            tokenize_function=get_webnlg_tokenize_fn(tokenizer.tokenizer),
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask']
        test_corpus = test_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in test_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
    elif args.dataset.lower() == "parasci":
        # Load ParaSCI dataset from Hugging Face datasets
        data = datasets.load_dataset("HHousen/ParaSCI")
        # rename columns to 'src' and 'trg' to match the expected format of helper_tokenize
        data = data.map(lambda x: {"src": x["sentence1"], "trg": x["sentence2"]})

        # Tokenize the dataset
        train_corpus = helper_tokenize(
            data["train"],
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        train_corpus = train_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in train_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        val_corpus = helper_tokenize(
            data["validation"],  # Use 'validation' split directly
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        val_corpus = val_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in val_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        test_corpus = helper_tokenize(
            data["test"],
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        test_corpus = test_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in test_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
z        )
    elif args.dataset.lower() == "wmt":
        # Load WMT19 en-de dataset from Hugging Face datasets
        data = datasets.load_dataset("wmt19", "de-en")
        # rename columns to 'src' and 'trg' to match the expected format of helper_tokenize
        data = data.map(lambda x: {"src": x["translation"]["de"], "trg": x["translation"]["en"]})
        # split train split into train and validation
        data["train"], data["val"] = data["train"].train_test_split(test_size=0.1).values()
        # Tokenize the dataset
        train_corpus = helper_tokenize(
            data["train"],  # Limit to 1000 samples for testing
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask']
        train_corpus = train_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in train_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        val_corpus = helper_tokenize(
            data["val"],  # Limit to 1000 samples for testing
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask']
        val_corpus = val_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in val_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        test_corpus = helper_tokenize(
            data["validation"],  # Limit to 1000 samples for testing
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_id_y', 'input_ids', 'input_mask', 'padding_mask']
        test_corpus = test_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in test_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
        # test_corpus = val_corpus  # For simplicity, use validation set as test set
    elif args.dataset.lower() == "grammar_correction":
        # Load Grammar Correction dataset from Hugging Face datasets
        data = datasets.load_dataset("agentlans/grammar-correction")
        # rename columns to 'src' and 'trg' to match the expected format of helper_tokenize
        data = data.map(lambda x: {"src": x["input"], "trg": x["output"]})

        # Create val split from train split (following ParaSCI pattern)
        train_val_split = data["train"].train_test_split(test_size=0.1, seed=42)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]  # val split from original train

        # Use original validation split as test split
        test_data = data["validation"]

        print("Grammar Correction dataset splits:")
        print(f"  Train: {len(train_data):,}")
        print(f"  Val (from train): {len(val_data):,}")
        print(f"  Test (original validation): {len(test_data):,}")

        # Tokenize the dataset
        train_corpus = helper_tokenize(
            train_data,
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        train_corpus = train_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in train_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )

        val_corpus = helper_tokenize(
            val_data,
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        val_corpus = val_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in val_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )

        test_corpus = helper_tokenize(
            test_data,
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        test_corpus = test_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in test_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
    elif args.dataset.lower() == "paws_wiki":
        # Load PAWS-Wiki dataset from local TSV files
        import pandas as pd

        from datasets import Dataset as Dataset2

        # Load TSV files
        train_df = pd.read_csv(f"{args.data_dir}/train.tsv", sep='\t')
        dev_df = pd.read_csv(f"{args.data_dir}/dev.tsv", sep='\t')
        test_df = pd.read_csv(f"{args.data_dir}/test.tsv", sep='\t')

        # Filter for paraphrases only (label=1)
        train_df = train_df[train_df['label'] == 1]
        dev_df = dev_df[dev_df['label'] == 1]
        test_df = test_df[test_df['label'] == 1]

        print("Filtered PAWS-Wiki dataset:")
        print(f"  Train paraphrases: {len(train_df):,}")
        print(f"  Dev paraphrases: {len(dev_df):,}")
        print(f"  Test paraphrases: {len(test_df):,}")

        # Convert to datasets format and rename columns to match expected format
        train_data = Dataset2.from_pandas(train_df)
        dev_data = Dataset2.from_pandas(dev_df)
        test_data = Dataset2.from_pandas(test_df)

        # Rename columns: sentence1→src, sentence2→trg (following ParaSCI pattern)
        train_data = train_data.map(lambda x: {"src": x["sentence1"], "trg": x["sentence2"]})
        dev_data = dev_data.map(lambda x: {"src": x["sentence1"], "trg": x["sentence2"]})
        test_data = test_data.map(lambda x: {"src": x["sentence1"], "trg": x["sentence2"]})

        # Tokenize using helper_tokenize function
        train_corpus = helper_tokenize(
            train_data,
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        train_corpus = train_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in train_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )

        val_corpus = helper_tokenize(
            dev_data,  # dev.tsv maps to valid split
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        val_corpus = val_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in val_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )

        test_corpus = helper_tokenize(
            test_data,
            tokenizer,
            args.max_seq_length,
            from_dict=False,
        )
        # drop all features except ['input_ids', 'input_mask', 'padding_mask']
        test_corpus = test_corpus.map(
            lambda x: {"input_ids": x["input_ids"], "input_mask": x["input_mask"], "padding_mask": x["padding_mask"]},
            remove_columns=[
                col for col in test_corpus.column_names if col not in ["input_ids", "input_mask", "padding_mask"]
            ],
        )
    else:
        # Load datasets
        train_corpus = get_corpus(args, args.max_seq_length, split="train", loaded_vocab=tokenizer)
        val_corpus = get_corpus(args, args.max_seq_length, split="valid", loaded_vocab=tokenizer)
        test_corpus = get_corpus(args, args.max_seq_length, split="test", loaded_vocab=tokenizer)

    # Create DatasetDict
    ds_dict = DatasetDict({"train": train_corpus, "valid": val_corpus, "test": test_corpus})

    # Save dataset to disk
    print(f"Saving dataset to {save_path}")
    ds_dict.save_to_disk(save_path)
