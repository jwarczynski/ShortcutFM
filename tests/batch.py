import pytest
import torch
from torch import tensor

from shortcutfm.batch import EncoderBatch, FMEncoderBatch


class TestEncoderBatch:
    def setup_method(self):
        self.seqs = tensor([[1, 2, 3], [4, 5, 6]])
        self.input_mask = tensor([[1, 1, 0], [1, 0, 0]])
        self.batch = EncoderBatch(self.seqs, self.input_mask)

    def test_numel(self):
        assert self.batch.numel() == 6

    def test_size(self):
        assert self.batch.size() == 2


class TestFMEncoderBatch:
    def setup_method(self):
        self.seqs = tensor([[1, 2, 3], [4, 5, 6]])
        self.input_mask = tensor([[1, 1, 0], [1, 0, 0]])
        self.embeddings = tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]])
        self.fm_batch = FMEncoderBatch(self.seqs, self.input_mask, self.embeddings)

    def test_fm_encoder_batch_creation(self):
        assert isinstance(self.fm_batch, FMEncoderBatch)
        assert isinstance(self.fm_batch, EncoderBatch)  # FMEncoderBatch inherits from EncoderBatch
        assert torch.equal(self.fm_batch.seqs, self.seqs)
        assert torch.equal(self.fm_batch.input_mask, self.input_mask)
        assert torch.equal(self.fm_batch.emebddings, self.embeddings)

    def test_numel(self):
        assert self.fm_batch.numel() == 6

    def test_size(self):
        assert self.fm_batch.size() == 2


if __name__ == '__main__':
    pytest.main()
