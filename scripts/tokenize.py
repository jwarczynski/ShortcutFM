import argparse

from datasets import DatasetDict

from shortcutfm.text_datasets import get_corpus
from shortcutfm.tokenizer import MyTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize dataset and save to disk.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--vocab", type=str, default="bert", help="Vocabulary type")
    parser.add_argument("--config_name", type=str, default="bert-base-uncased", help="Model config name")
    parser.add_argument("--max_seq_length", type=int, default=128)

    args = parser.parse_args()

    args.data_dir = f"../datasets/raw/{args.dataset}"
    save_path = f"../datasets/tokenized/{args.dataset}"

    # Initialize tokenizer
    tokenizer = MyTokenizer(args, True)

    # Load datasets
    train_corpus = get_corpus(args, args.max_seq_length, split="train", loaded_vocab=tokenizer)
    val_corpus = get_corpus(args, args.max_seq_length, split="valid", loaded_vocab=tokenizer)
    test_corpus = get_corpus(args, args.max_seq_length, split="test", loaded_vocab=tokenizer)

    # Create DatasetDict
    ds_dict = DatasetDict(
        {
            "train": train_corpus,
            "valid": val_corpus,
            "test": test_corpus
        }
    )

    # Save dataset to disk
    print(f"Saving dataset to {save_path}")
    ds_dict.save_to_disk(save_path)
