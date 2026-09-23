"""Implementation of AnomalyScoreThreshold based on TorchMetrics."""

from __future__ import annotations

import warnings
import torch
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve


class AnomalyScoreThreshold(BinaryPrecisionRecallCurve):
    """Anomaly Score Threshold based on BinaryPrecisionRecallCurve."""

    def __init__(self, default_value: float = 0.5, thresholds: int | None = 1000, **kwargs) -> None:
        super().__init__(thresholds=thresholds, **kwargs)
        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

    def compute(self) -> Tensor:
        """Compute the threshold that yields the optimal F1 score."""
        precision: Tensor
        recall: Tensor
        thresholds: Tensor

        if not any(1 in batch for batch in self.target):
            warnings.warn(
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value
