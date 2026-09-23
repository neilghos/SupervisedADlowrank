"""Implementation of AUPR metric based on modern TorchMetrics."""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve


class AUPR(BinaryPrecisionRecallCurve):
    """Area under the PR curve based on BinaryPrecisionRecallCurve."""

    def __init__(self, thresholds: int | None = 1000, **kwargs):
        super().__init__(thresholds=thresholds, **kwargs)

    def compute(self) -> Tensor:
        """Compute PR curve, then compute area under the curve."""
        prec, rec = self._compute()
        return torch.trapezoid(prec, rec)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with flattened predictions and targets."""
        super().update(preds.flatten(), target.flatten().long())

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute prec/rec value pairs."""
        prec, rec, _ = super().compute()
        return (prec, rec)

    def generate_figure(self):
        """Plotting disabled for headless evaluation."""
        return None
