from typing import Any, Union, Sequence, Optional

from composer.models import ComposerModel
from torch import Tensor
from torchmetrics import Metric

from shortcutfm.batch import EncoderBatch
from shortcutfm.criteria import Criterion


class TrainUnit(ComposerModel):

    def __init__(self, criterion: Criterion) -> None:
        super().__init__()
        self.criterion = criterion

    def forward(self, batch: EncoderBatch) -> Any:
        return self.criterion(batch)

    def loss(self, outputs: dict[str, Tensor], batch: EncoderBatch, *args, **kwargs) -> Union[Tensor, Sequence[Tensor]]:
        return outputs["loss"]

    def eval_forward(self, batch: EncoderBatch, outputs: Optional[Any] = None, ) -> Any:
        return outputs["loss"]

    def get_metrics(self, is_train=False) -> dict[str, Metric]:
        ...

    def update_metric(self, batch, outputs, metric) -> None:
        ...
