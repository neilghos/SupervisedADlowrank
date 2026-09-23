"""Paper-matched image-level metrics for the VAD benchmark."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torchmetrics.classification import BinaryAUROC, BinaryROC


def _as_flat_tensor(values: Tensor | Iterable[Tensor], dtype: torch.dtype) -> Tensor:
    """Convert batched tensors or tensor iterables to one CPU vector."""

    if isinstance(values, Tensor):
        return values.detach().flatten().to(dtype=dtype, device="cpu")
    return torch.cat([value.detach().flatten().to(dtype=dtype, device="cpu") for value in values])


def classification_auroc(pred_scores: Tensor, labels: Tensor) -> float:
    """Compute image-level Classification AUROC in [0, 1]."""

    scores = _as_flat_tensor(pred_scores, torch.float32)
    targets = _as_flat_tensor(labels, torch.long)
    return float(BinaryAUROC()(scores, targets).item())


def fpr_at_95_tpr(pred_scores: Tensor, labels: Tensor, target_tpr: float = 0.95) -> float:
    """Compute FPR at the first ROC operating point reaching target TPR.

    Returns a percentage, matching the paper's table convention. For example,
    ``36.8`` means that 36.8 percent of good parts are false positives.
    """

    scores = _as_flat_tensor(pred_scores, torch.float32)
    targets = _as_flat_tensor(labels, torch.long)
    fpr, tpr, _ = BinaryROC()(scores, targets)
    valid = torch.where(tpr >= target_tpr)[0]
    index = valid[0] if len(valid) else torch.argmin(torch.abs(tpr - target_tpr))
    return float(fpr[index].item() * 100.0)


def paper_metrics(pred_scores: Tensor, labels: Tensor) -> dict[str, float]:
    """Return the two metrics reported by the VAD paper."""

    return {
        "classification_auroc": classification_auroc(pred_scores, labels),
        "fpr_at_95_tpr": fpr_at_95_tpr(pred_scores, labels),
    }


@torch.inference_mode()
def evaluate_model(model, datamodule) -> dict[str, float]:
    """Run the model over the complete test set and compute paper metrics."""

    model.eval()
    scores: list[Tensor] = []
    labels: list[Tensor] = []
    for batch in datamodule.test_dataloader():
        predictions = model.model(batch.image.to(model.device))
        scores.append(predictions.pred_score.detach().cpu())
        labels.append(batch.gt_label.detach().cpu())
    return paper_metrics(torch.cat(scores), torch.cat(labels))
