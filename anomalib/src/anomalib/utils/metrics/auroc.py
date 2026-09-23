from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics.classification import BinaryAUROC


class AUROC(BinaryAUROC):
    """Fast area under the ROC curve based on native BinaryAUROC."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with flattened predictions and targets."""
        super().update(preds.flatten(), target.flatten().long())

    def generate_figure(self):
        """Plotting disabled for headless evaluation."""
        return None
