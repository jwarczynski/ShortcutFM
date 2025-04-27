import json

import torch
from torch.utils.data import Dataset

from datasets import Dataset as Dataset2


def get_corpus(data_args, seq_len, split="train", loaded_vocab=None, max_examples=None):
    print(
        "#" * 30,
        f"\nLoading dataset {data_args.dataset} from {data_args.data_dir}...",
    )

    sentence_lst = {"src": [], "trg": []}

    if split == "train":
        print("### Loading form the TRAIN set...")
        path = f"{data_args.data_dir}/train.jsonl"
    elif split == "valid":
        print("### Loading form the VALID set...")
        path = f"{data_args.data_dir}/valid.jsonl"
    elif split == "test":
        print("### Loading form the TEST set...")
        path = f"{data_args.data_dir}/test.jsonl"
    else:
        raise ValueError("invalid split for dataset")

    with open(path) as f_reader:
        for i, row in enumerate(f_reader):
            if max_examples is not None and i >= max_examples:
                break
            content = json.loads(row)
            sentence_lst["src"].append(content["src"].strip())
            sentence_lst["trg"].append(content["trg"].strip())

    print("### Data samples...\n", sentence_lst["src"][:2], sentence_lst["trg"][:2])

    # get tokenizer.
    vocab_dict = loaded_vocab
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples["src"])
        input_id_y = vocab_dict.encode_token(examples["trg"])
        result_dict = {"input_id_x": input_id_x, "input_id_y": input_id_y}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["src", "trg"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print("### tokenized_datasets", tokenized_datasets)
    print("### tokenized_datasets...example", tokenized_datasets["input_id_x"][0])

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        src_len = trg_len = 0
        for i in range(len(group_lst["input_id_x"])):
            end_token = group_lst["input_id_x"][i][-1]
            src = group_lst["input_id_x"][i][:-1]
            trg = group_lst["input_id_y"][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src) > len(trg):
                    src.pop()
                elif len(src) < len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)
            src_len += len(src)
            trg_len += len(trg)

            # 2 consecutive sep tokens (copied from flowseq)
            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0] * (len(src) + 1))
        group_lst["input_ids"] = lst
        group_lst["input_mask"] = mask
        return group_lst

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc="merge and mask",
    )

    print("### tokenized_datasets", tokenized_datasets)
    print(
        "### tokenized_datasets...example",
        tokenized_datasets["input_ids"][0],
        tokenized_datasets["input_mask"][0],
    )

    def pad_function(group_lst):
        max_length = seq_len
        seqs, padding_mask = _collate_batch_helper(
            group_lst["input_ids"],
            vocab_dict.pad_token_id,
            max_length,
            return_mask=True,
        )
        group_lst["input_ids"] = seqs
        group_lst["padding_mask"] = padding_mask
        group_lst["input_mask"] = _collate_batch_helper(group_lst["input_mask"], 1, max_length)
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc="padding",
    )

    print(lm_datasets, "padded dataset")
    print(
        "### padded dataset...example",
        lm_datasets["input_ids"][0],
        lm_datasets["input_mask"][0],
    )

    return lm_datasets


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], 0, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


class TextDataset(Dataset):
    def __init__(self, text_dataset):
        super().__init__()
        self.text_dataset = text_dataset
        self.length = len(self.text_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "seqs": torch.tensor(self.text_dataset[idx]["input_ids"]),
            "padding_mask": torch.tensor(self.text_dataset[idx]["padding_mask"]),
            "input_ids_mask": torch.tensor(self.text_dataset[idx]["input_mask"]),
        }
