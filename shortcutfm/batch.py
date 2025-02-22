from dataclasses import dataclass

from torch import Tensor


@dataclass
class EncoderBatch:
    seqs: Tensor
    input_mask: Tensor

    def numel(self) -> int:
        return self.seqs.numel()

    def size(self):
        return self.seqs.size(0)

@dataclass
class FMEncoderBatch(EncoderBatch):
    emebddings: Tensor