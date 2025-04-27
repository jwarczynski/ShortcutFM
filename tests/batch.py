import pytest
import torch

from shortcutfm.batch import EncoderBatch, FlowMatchingBatch, ShortcutFMBatch


class TestEncoderBatch:
    def setup_method(self):
        self.seqs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.padding_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.input_ids_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.batch = EncoderBatch(self.seqs, self.padding_mask, self.input_ids_mask)

    def test_numel(self):
        assert self.batch.numel() == 3

    def test_size(self):
        assert self.batch.size() == 2

    def test_encoder_batch_creation(self):
        assert torch.equal(self.batch.seqs, self.seqs)
        assert torch.equal(self.batch.padding_mask, self.padding_mask)
        assert torch.equal(self.batch.input_ids_mask, self.input_ids_mask)


class TestFlowMatchingBatch:
    def setup_method(self):
        self.seqs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.padding_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.input_ids_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.x_start = torch.tensor([[7, 8, 9], [10, 11, 12]])
        self.x_t = torch.tensor([[13, 14, 15], [16, 17, 18]])
        self.noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.t = torch.tensor([100, 200])

        self.fm_batch = FlowMatchingBatch(
            self.seqs,
            self.padding_mask,
            self.input_ids_mask,
            self.x_start,
            self.x_t,
            self.noise,
            self.t,
        )

    def test_flow_matching_batch_creation(self):
        assert isinstance(self.fm_batch, FlowMatchingBatch)
        assert isinstance(self.fm_batch, EncoderBatch)
        assert torch.equal(self.fm_batch.seqs, self.seqs)
        assert torch.equal(self.fm_batch.padding_mask, self.padding_mask)
        assert torch.equal(self.fm_batch.input_ids_mask, self.input_ids_mask)
        assert torch.equal(self.fm_batch.x_start, self.x_start)
        assert torch.equal(self.fm_batch.x_t, self.x_t)
        assert torch.equal(self.fm_batch.noise, self.noise)
        assert torch.equal(self.fm_batch.t, self.t)

    def test_numel(self):
        assert self.fm_batch.numel() == 3

    def test_size(self):
        assert self.fm_batch.size() == 2

    def test_split(self):
        fm_batch1, fm_batch2 = self.fm_batch.split(1)
        assert fm_batch1.size() == 1
        assert fm_batch2.size() == 1
        assert torch.equal(fm_batch1.seqs, self.seqs[:1])
        assert torch.equal(fm_batch2.seqs, self.seqs[1:])


class TestShortcutFMBatch:
    def setup_method(self):
        self.seqs = torch.tensor([[1, 2, 3], [4, 5, 6]])
        self.padding_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.input_ids_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        self.x_start = torch.tensor([[7, 8, 9], [10, 11, 12]])
        self.x_t = torch.tensor([[13, 14, 15], [16, 17, 18]])
        self.noise = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        self.t = torch.tensor([100, 200])
        self.shortcut_size = torch.tensor([32, 64])
        self.fm_batch = FlowMatchingBatch(
            self.seqs,
            self.padding_mask,
            self.input_ids_mask,
            self.x_start,
            self.x_t,
            self.noise,
            self.t,
        )
        self.shortcut_fm_batch = ShortcutFMBatch.from_flow_matching_batch(self.fm_batch, self.shortcut_size)

    def test_shortcut_fm_batch_creation(self):
        assert isinstance(self.shortcut_fm_batch, ShortcutFMBatch)
        assert isinstance(self.shortcut_fm_batch, FlowMatchingBatch)
        assert isinstance(self.shortcut_fm_batch, EncoderBatch)
        assert torch.equal(self.shortcut_fm_batch.seqs, self.seqs)
        assert torch.equal(self.shortcut_fm_batch.padding_mask, self.padding_mask)
        assert torch.equal(self.shortcut_fm_batch.input_ids_mask, self.input_ids_mask)
        assert torch.equal(self.shortcut_fm_batch.x_start, self.x_start)
        assert torch.equal(self.shortcut_fm_batch.x_t, self.x_t)
        assert torch.equal(self.shortcut_fm_batch.noise, self.noise)
        assert torch.equal(self.shortcut_fm_batch.t, self.t)
        assert torch.equal(self.shortcut_fm_batch.shortcut_size, self.shortcut_size)

    def test_numel(self):
        assert self.shortcut_fm_batch.numel() == 3

    def test_size(self):
        assert self.shortcut_fm_batch.size() == 2


if __name__ == "__main__":
    pytest.main()
