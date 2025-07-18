import argparse

import datasets
from datasets import DatasetDict
from shortcutfm.text_datasets import get_corpus, get_webnlg_tokenize_fn, helper_tokenize
from shortcutfm.tokenizer import MyTokenizer

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
    elif args.dataset.lower() == "wmt":
        # Load WMT19 en-de dataset from Hugging Face datasets
        data = datasets.load_dataset("wmt19", "de-en")
        # rename columns to 'src' and 'trg' to match the expected format of helper_tokenize
        data = data.map(lambda x: {"src": x["translation"]["de"], "trg": x["translation"]["en"]})
        # Tokenize the dataset
        train_corpus = helper_tokenize(
            data["train"],
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
            data["validation"],
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
            data["test"],
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
