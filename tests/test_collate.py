import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Assuming EncoderBatch is a simple class that takes the dictionary as kwargs
class EncoderBatch:
    def __init__(self, seqs: Tensor, padding_mask: Tensor, input_ids_mask: Tensor):
        self.seqs = seqs
        self.padding_mask = padding_mask
        self.input_ids_mask = input_ids_mask


# Your TextDataset class (unchanged)
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


# Your modified collate function
def collate(
    batch: list[dict[str, Tensor]],
    mark_first_padding: bool = False,
    mark_second_padding: bool = False,
) -> EncoderBatch:
    transposed_batch = {k: [item[k] for item in batch] for k in batch[0]}
    collated_batch = {k: torch.stack(v) for k, v in transposed_batch.items()}

    if mark_first_padding or mark_second_padding:
        padding_mask = collated_batch["padding_mask"]
        batch_size, seq_len = padding_mask.shape

        for i in range(batch_size):
            padding_indices = (padding_mask[i] == 0).nonzero(as_tuple=True)[0]

            if len(padding_indices) > 0 and mark_first_padding:
                padding_mask[i, padding_indices[0]] = 1

            if len(padding_indices) > 1 and mark_second_padding:
                padding_mask[i, padding_indices[1]] = 1

        collated_batch["padding_mask"] = padding_mask

    return EncoderBatch(**collated_batch)


# Create sample data
sample_data = [
    # Sequence 1: length 5, 2 padding tokens
    {
        "input_ids": [1, 2, 3, 0, 0],
        "padding_mask": [1, 1, 1, 0, 0],
        "input_mask": [1, 0, 1, 0, 0],
    },
    # Sequence 2: length 5, 1 padding token
    {
        "input_ids": [4, 5, 6, 7, 0],
        "padding_mask": [1, 1, 1, 1, 0],
        "input_mask": [1, 1, 0, 1, 0],
    },
    # Sequence 3: length 5, 3 padding tokens
    {
        "input_ids": [8, 9, 0, 0, 0],
        "padding_mask": [1, 1, 0, 0, 0],
        "input_mask": [0, 1, 0, 0, 0],
    },
]

# Create dataset and dataloader
dataset = TextDataset(sample_data)
dataloader = DataLoader(dataset, batch_size=3, collate_fn=collate)


# Test function with different settings
def test_collate(dataloader: DataLoader):
    print("Testing with default settings (no padding marking):")
    for batch in dataloader:
        print("Padding mask:\n", batch.padding_mask)

    print("\nTesting with mark_first_padding=True:")
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=lambda x: collate(x, mark_first_padding=True))
    for batch in dataloader:
        print("Padding mask:\n", batch.padding_mask)

    print("\nTesting with mark_first_padding=True and mark_second_padding=True:")
    dataloader = DataLoader(
        dataset,
        batch_size=3,
        collate_fn=lambda x: collate(x, mark_first_padding=True, mark_second_padding=True),
    )
    for batch in dataloader:
        print("Padding mask:\n", batch.padding_mask)


# Run the test
if __name__ == "__main__":
    test_collate(dataloader)
