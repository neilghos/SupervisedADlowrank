"""Memory-safe supervised low-rank residual model for Anomalib 2.x."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from anomalib import LearningType
from anomalib.data import InferenceBatch
from anomalib.models.components import AnomalibModule


class _SupervisedLowRankCore(nn.Module):
    """Healthy subspace plus a supervised classifier over residual statistics."""

    def __init__(self, rank: int = 50, variance_threshold: float = 1e-4, ridge: float = 1e-2) -> None:
        super().__init__()
        self.rank = rank
        self.variance_threshold = variance_threshold
        self.ridge = ridge
        self.register_buffer("active_mask", torch.empty(0, dtype=torch.bool))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("basis", torch.empty(0))
        self.register_buffer("feature_mean", torch.zeros(5))
        self.register_buffer("feature_std", torch.ones(5))
        self.register_buffer("classifier_weight", torch.zeros(5))
        self.register_buffer("classifier_bias", torch.zeros(()))
        self.register_buffer("fitted", torch.tensor(False))

    @property
    def is_fitted(self) -> bool:
        return bool(self.fitted.item())

    @staticmethod
    def _gray(images: Tensor) -> Tensor:
        images = images.float()
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B,C,H,W), got {tuple(images.shape)}")
        if images.shape[1] == 1:
            return images[:, 0]
        if images.shape[1] >= 3:
            return 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
        return images.mean(dim=1)

    def _project_residual(self, images: Tensor) -> Tensor:
        active = self._gray(images).flatten(1)[:, self.active_mask]
        sample_mean = active.mean(1, keepdim=True)
        sample_std = active.std(1, keepdim=True).clamp_min(1e-6)
        centered = (active - sample_mean) / sample_std - self.mean
        projection = (centered @ self.basis) @ self.basis.T
        return centered - projection

    @staticmethod
    def _features(residual: Tensor) -> Tensor:
        absolute = residual.abs()
        top_count = max(1, min(100, absolute.shape[1]))
        top_mean = absolute.topk(top_count, dim=1).values.mean(dim=1)
        percentile = torch.quantile(absolute, 0.95, dim=1)
        return torch.stack(
            [absolute.mean(1), absolute.std(1), absolute.amax(1), top_mean, percentile],
            dim=1,
        )

    @staticmethod
    def _normal_images(batch, device: torch.device) -> Tensor:
        labels = batch.gt_label.to(device)
        return batch.image.to(device)[labels == 0]

    def fit_basis_from_loader(self, loader_factory: Callable[[], Iterable], device: torch.device) -> None:
        """Fit the healthy basis in batches using a randomized SVD."""

        pixel_sum = None
        pixel_sq_sum = None
        count = 0
        for batch in loader_factory():
            normal = self._gray(self._normal_images(batch, device))
            if not len(normal):
                continue
            flat = normal.flatten(1)
            batch_sum = flat.sum(0)
            batch_sq_sum = (flat * flat).sum(0)
            pixel_sum = batch_sum if pixel_sum is None else pixel_sum + batch_sum
            pixel_sq_sum = batch_sq_sum if pixel_sq_sum is None else pixel_sq_sum + batch_sq_sum
            count += len(flat)
        if count < 2:
            raise ValueError("At least two normal images are required for the basis.")

        variance = (pixel_sq_sum / count - (pixel_sum / count).square()).clamp_min(0)
        active_mask = variance > self.variance_threshold
        if int(active_mask.sum()) < 2:
            active_mask = torch.ones_like(active_mask, dtype=torch.bool)
        active_count = int(active_mask.sum())
        random_q = min(self.rank + 8, count, active_count)
        omega = torch.randn((active_count, random_q), device=device)

        mean_sum = torch.zeros(active_count, device=device)
        for batch in loader_factory():
            normal = self._gray(self._normal_images(batch, device))
            if not len(normal):
                continue
            flat = normal.flatten(1)[:, active_mask]
            normalized = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-6)
            mean_sum += normalized.sum(0)
        mean = (mean_sum / count).unsqueeze(0)

        projected_chunks: list[Tensor] = []
        for batch in loader_factory():
            normal = self._gray(self._normal_images(batch, device))
            if not len(normal):
                continue
            flat = normal.flatten(1)[:, active_mask]
            normalized = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-6)
            projected_chunks.append((normalized - mean) @ omega)
        q_basis = torch.linalg.qr(torch.cat(projected_chunks, dim=0), mode="reduced").Q

        right_matrix = torch.zeros((active_count, random_q), device=device)
        row_start = 0
        for batch in loader_factory():
            normal = self._gray(self._normal_images(batch, device))
            if not len(normal):
                continue
            flat = normal.flatten(1)[:, active_mask]
            normalized = (flat - flat.mean(1, keepdim=True)) / flat.std(1, keepdim=True).clamp_min(1e-6)
            centered = normalized - mean
            row_end = row_start + len(centered)
            right_matrix += centered.T @ q_basis[row_start:row_end]
            row_start = row_end

        left, _, _ = torch.linalg.svd(right_matrix, full_matrices=False)
        self.active_mask = active_mask
        self.mean = mean
        self.basis = left[:, : min(self.rank, left.shape[1])].contiguous()

    def fit_classifier_features(self, features: Tensor, labels: Tensor) -> None:
        labels = labels.long().flatten()
        if int((labels == 0).sum()) < 2 or int((labels == 1).sum()) < 1:
            raise ValueError("The supervised stage needs normal and bad labelled images.")
        feature_mean = features.mean(0)
        feature_std = features.std(0).clamp_min(1e-6)
        standardized = (features - feature_mean) / feature_std
        design = torch.cat([torch.ones((len(features), 1), device=features.device), standardized], dim=1)
        regularizer = self.ridge * torch.eye(design.shape[1], device=features.device)
        regularizer[0, 0] = 0.0
        coefficients = torch.linalg.solve(
            design.T @ design + regularizer,
            design.T @ labels.float().unsqueeze(1),
        ).squeeze(1)
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.classifier_bias = coefficients[0]
        self.classifier_weight = coefficients[1:]
        self.fitted.fill_(True)

    def fit_classifier_from_loader(self, loader_factory: Callable[[], Iterable], device: torch.device) -> None:
        feature_batches: list[Tensor] = []
        label_batches: list[Tensor] = []
        for batch in loader_factory():
            images = batch.image.to(device)
            feature_batches.append(self._features(self._project_residual(images)))
            label_batches.append(batch.gt_label.to(device))
        self.fit_classifier_features(torch.cat(feature_batches), torch.cat(label_batches))

    def forward(self, images: Tensor) -> InferenceBatch:
        if not self.is_fitted:
            raise RuntimeError("SupervisedLowRank has not been fitted yet.")
        residual = self._project_residual(images)
        absolute = residual.abs()
        height, width = images.shape[-2:]
        full_map = torch.zeros((images.shape[0], height * width), device=images.device, dtype=absolute.dtype)
        full_map[:, self.active_mask] = absolute
        anomaly_map = F.avg_pool2d(full_map.view(images.shape[0], 1, height, width), 7, 1, 3)
        features = self._features(residual)
        standardized = (features - self.feature_mean) / self.feature_std
        score = torch.sigmoid(self.classifier_bias + standardized @ self.classifier_weight)
        return InferenceBatch(pred_score=score, anomaly_map=anomaly_map)


class SupervisedLowRank(AnomalibModule):
    """Anomalib wrapper for memory-safe supervised low-rank decomposition."""

    def __init__(self, rank: int = 50, variance_threshold: float = 1e-4, ridge: float = 1e-2) -> None:
        super().__init__(pre_processor=False, post_processor=False, visualizer=False)
        self.model = _SupervisedLowRankCore(rank, variance_threshold, ridge)

    @property
    def learning_type(self) -> LearningType:
        return LearningType.FEW_SHOT

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"max_epochs": 1, "num_sanity_val_steps": 0}

    @staticmethod
    def configure_optimizers() -> None:
        return None

    def _fit_from_datamodule(self) -> None:
        if self.model.is_fitted or getattr(self.trainer, "datamodule", None) is None:
            return
        pass_names = iter(("basis variance", "basis mean", "basis range", "basis projection", "supervised head"))

        def loader_factory():
            description = next(pass_names, "model pass")
            return tqdm(
                self.trainer.datamodule.train_dataloader(),
                desc=description,
                unit="batch",
                leave=True,
            )

        self.model.fit_basis_from_loader(loader_factory, self.device)
        self.model.fit_classifier_from_loader(loader_factory, self.device)

    def on_validation_start(self) -> None:
        self._fit_from_datamodule()

    def on_test_start(self) -> None:
        self._fit_from_datamodule()

    def training_step(self, batch, batch_idx: int):
        del batch, batch_idx
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def validation_step(self, batch, batch_idx: int):
        del batch_idx
        if not self.model.is_fitted:
            return None
        return batch.update(**self.model(batch.image)._asdict())

    def test_step(self, batch, batch_idx: int):
        del batch_idx
        return batch.update(**self.model(batch.image)._asdict())

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        del batch_idx, dataloader_idx
        return batch.update(**self.model(batch.image)._asdict())
