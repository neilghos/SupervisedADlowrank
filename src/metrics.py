"""Classification metrics used by the reconstruction-free SpatialAD pipeline."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torchmetrics.classification import BinaryAUROC, BinaryROC


def paper_metrics(pred_scores: Tensor, labels: Tensor) -> dict[str, float]:
    """Return image-level AUROC and FPR at 95% TPR."""

    scores = pred_scores.detach().flatten().float().cpu()
    targets = labels.detach().flatten().long().cpu()
    if targets.unique().numel() < 2:
        return {"classification_auroc": float("nan"), "fpr_at_95_tpr": float("nan")}

    auroc = float(BinaryAUROC()(scores, targets).item())
    fpr, tpr, _ = BinaryROC()(scores, targets)
    valid = torch.where(tpr >= 0.95)[0]
    index = valid[0] if len(valid) else torch.argmin(torch.abs(tpr - 0.95))
    return {
        "classification_auroc": auroc,
        "fpr_at_95_tpr": float(fpr[index].item() * 100.0),
    }


def test_group_masks(groups: list[str]) -> dict[str, Tensor]:
    """Masks for all, seen-defect, and unseen-defect test groups."""

    values = np.asarray(groups, dtype=object)
    return {
        "all": torch.ones(len(groups), dtype=torch.bool),
        "seen_defects": torch.from_numpy(np.isin(values, ["good", "bad"])),
        "unseen_defects": torch.from_numpy(
            np.isin(values, ["good", "bad_unseen_defects"])
        ),
    }
