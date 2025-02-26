from argparse import Namespace

from shortcutfm.text_datasets import get_corpus
from shortcutfm.tokenizer import MyTokenizer

if __name__ == '__main__':
    args = Namespace(
        data_dir="../datasets/QQP-Official",
        dataset="QQP-Official",
        vocab="bert",
        config_name="bert-base-uncased",
        merge_strategy="nonequal",
    )
    tokenizer = MyTokenizer(args, True)

    corpus = get_corpus(args, 128, split="valid", loaded_vocab=tokenizer)["train"]

    save_path = "../datasets/tokenized/QQP-Official/val_subset_16"

    print(corpus)
    corpus.save_to_disk(save_path)
