from dataclasses import dataclass

from torch import Tensor


@dataclass
class EncoderBatch:
    seqs: Tensor
    padding_mask: Tensor
    input_ids_mask: Tensor

    def numel(self) -> int:
        return self.padding_mask.sum().item()

    def size(self):
        return self.seqs.size(0)


@dataclass
class FlowMatchingBatch(EncoderBatch):
    x_start: Tensor
    x_t: Tensor
    noise: Tensor
    t: Tensor

    def split(self, index: int) -> tuple["FlowMatchingBatch", "FlowMatchingBatch"]:
        return FlowMatchingBatch(
            self.seqs[:index],
            self.padding_mask[:index],
            self.input_ids_mask[:index],
            self.x_start[:index],
            self.x_t[:index],
            self.noise[:index],
            self.t[:index]
        ), FlowMatchingBatch(
            self.seqs[index:],
            self.padding_mask[index:],
            self.input_ids_mask[index:],
            self.x_start[index:],
            self.x_t[index:],
            self.noise[index:],
            self.t[index:]
        )


@dataclass
class ShortcutFMBatch(FlowMatchingBatch):
    shortcut_size: Tensor

    @classmethod
    def from_flow_matching_batch(cls, fm_batch: FlowMatchingBatch, shortcut_size: Tensor) -> "ShortcutFMBatch":
        return cls(
            fm_batch.seqs,
            fm_batch.padding_mask,
            fm_batch.input_ids_mask,
            fm_batch.x_start,
            fm_batch.x_t,
            fm_batch.noise,
            fm_batch.t,
            shortcut_size
        )
