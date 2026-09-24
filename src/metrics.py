"""Image-level and seen/unseen metrics for spatial MIL."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryROC


def _as_flat_tensor(values: Tensor | Iterable[Tensor], dtype: torch.dtype) -> Tensor:
    if isinstance(values, Tensor):
        return values.detach().flatten().to(dtype=dtype, device="cpu")
    return torch.cat([value.detach().flatten().to(dtype=dtype, device="cpu") for value in values])


def paper_metrics(pred_scores: Tensor, labels: Tensor) -> dict[str, float]:
    scores = _as_flat_tensor(pred_scores, torch.float32)
    targets = _as_flat_tensor(labels, torch.long)
    fpr, tpr, _ = BinaryROC()(scores, targets)
    valid = torch.where(tpr >= 0.95)[0]
    index = valid[0] if len(valid) else torch.argmin(torch.abs(tpr - 0.95))
    return {
        "classification_auroc": float(BinaryAUROC()(scores, targets).item()),
        "fpr_at_95_tpr": float(fpr[index].item() * 100.0),
    }


def _group_masks(groups: list[str]) -> dict[str, Tensor]:
    group_tensor = torch.tensor(
        [0 if group == "good" else 1 if group == "bad" else 2 for group in groups],
        dtype=torch.long,
    )
    return {
        "all": torch.ones_like(group_tensor, dtype=torch.bool),
        "seen_defects": (group_tensor == 0) | (group_tensor == 1),
        "unseen_defects": (group_tensor == 0) | (group_tensor == 2),
    }


@torch.inference_mode()
def _test_predictions(model, datamodule) -> tuple[Tensor, Tensor, list[str], Tensor]:
    model.eval()
    loader = DataLoader(
        dataset=datamodule.test_data,
        shuffle=False,
        batch_size=datamodule.eval_batch_size,
        num_workers=datamodule.num_workers,
        collate_fn=datamodule.test_data.collate_fn,
    )
    groups = datamodule.test_data.samples["label"].astype(str).tolist()
    scores: list[Tensor] = []
    labels: list[Tensor] = []
    maps: list[Tensor] = []
    ordered_groups: list[str] = []
    cursor = 0
    for batch in loader:
        predictions = model.model(batch.image.to(model.device))
        batch_size = len(batch.gt_label)
        scores.append(predictions.pred_score.detach().cpu())
        labels.append(batch.gt_label.detach().cpu())
        maps.append(predictions.anomaly_map.detach().cpu())
        ordered_groups.extend(groups[cursor : cursor + batch_size])
        cursor += batch_size
    return torch.cat(scores), torch.cat(labels), ordered_groups, torch.cat(maps)


@torch.inference_mode()
def evaluate_model(model, datamodule) -> dict[str, float]:
    scores, labels, _, _ = _test_predictions(model, datamodule)
    return paper_metrics(scores, labels)


@torch.inference_mode()
def evaluate_model_by_test_group(model, datamodule) -> dict[str, dict[str, float]]:
    scores, labels, groups, _ = _test_predictions(model, datamodule)
    return {
        name: paper_metrics(scores[mask], labels[mask])
        for name, mask in _group_masks(groups).items()
    }


@torch.inference_mode()
def evaluate_attention_maps(model, datamodule) -> dict[str, dict[str, float]]:
    _, labels, groups, maps = _test_predictions(model, datamodule)
    flat = maps.flatten(1)
    scores = {
        "mean": flat.mean(1),
        "max": flat.amax(1),
        "p95": torch.quantile(flat, 0.95, dim=1),
        "top1pct": flat.topk(max(1, int(flat.shape[1] * 0.01)), dim=1).values.mean(1),
        "top5pct": flat.topk(max(1, int(flat.shape[1] * 0.05)), dim=1).values.mean(1),
    }
    masks = _group_masks(groups)
    return {
        aggregation: {
            group: paper_metrics(score[mask], labels[mask])
            for group, mask in masks.items()
        }
        for aggregation, score in scores.items()
    }
