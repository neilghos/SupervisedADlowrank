"""Torch model for Low-Rank Residual Decomposition."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor


class CommonSpatialPruner:
    """Stage 1: Variance pruner to collapse static background columns (P -> P')."""

    def __init__(self, var_threshold: float = 1e-4):
        self.var_threshold = var_threshold
        self.active_mask: Optional[Tensor] = None
        self.H = self.W = self.P_total = self.P_active = None

    def fit(self, X: Tensor) -> "CommonSpatialPruner":
        """Fit active spatial coordinates on N scans of shape (N, C, H, W) or (N, H, W)."""
        if X.ndim == 4:
            if X.shape[1] == 3:
                X = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
            else:
                X = X.squeeze(1)
        N, self.H, self.W = X.shape
        self.P_total = self.H * self.W

        col_vars = torch.var(X.view(N, -1), dim=0, unbiased=True)
        self.active_mask = col_vars > self.var_threshold
        self.P_active = int(self.active_mask.sum().item())
        return self

    def transform(self, X: Tensor) -> Tensor:
        """Prune invariant coordinates: (B, C, H, W) -> (B, P_active)."""
        if X.ndim == 4:
            if X.shape[1] == 3:
                X = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
            else:
                X = X.squeeze(1)
        B = X.shape[0] if X.ndim == 3 else 1
        mask = self.active_mask.to(X.device)
        return X.view(B, -1)[:, mask]

    def map_to_2d(self, X_pruned: Tensor) -> Tensor:
        """Reconstruct 2D spatial grid (B, H, W) from pruned vectors."""
        is_single = X_pruned.ndim == 1
        X_p = X_pruned.unsqueeze(0) if is_single else X_pruned
        B = X_p.shape[0]

        grid = torch.zeros((B, self.P_total), device=X_p.device, dtype=X_p.dtype)
        grid[:, self.active_mask.to(X_p.device)] = X_p
        grid = grid.view(B, self.H, self.W)
        return grid.squeeze(0) if is_single else grid


from anomalib.models.components.filters.blur import GaussianBlur2d


class LowRankModel(nn.Module):
    """
    Closed-Form Low-Rank Subspace & Sparse Residual Decomposition Core.
    """

    def __init__(self, rank: int = 50, var_threshold: float = 1e-4, score_type: str = "l1"):
        super().__init__()
        self.rank = rank
        self.score_type = score_type.lower()
        self.pruner = CommonSpatialPruner(var_threshold=var_threshold)
        self.blur = GaussianBlur2d(sigma=4, channels=1)

        self.register_buffer("mean", None)
        self.register_buffer("basis", None)
        self.is_fitted = False

    def fit(self, X_train: Tensor) -> "LowRankModel":
        """
        Fits the single healthy anatomy subspace V_H strictly on healthy training scans.
        """
        self.pruner.fit(X_train)
        X_pruned = self.pruner.transform(X_train)

        # Per-sample affine normalization: cancels out patient-to-patient scale and brightness
        mu_sample = torch.mean(X_pruned, dim=1, keepdim=True)
        std_sample = torch.std(X_pruned, dim=1, keepdim=True) + 1e-6
        X_norm = (X_pruned - mu_sample) / std_sample

        N, P = X_norm.shape
        actual_rank = min(self.rank, N - 1, P)
        mean = torch.mean(X_norm, dim=0, keepdim=True)
        X_centered = X_norm - mean

        # Closed-form SVD gives healthy basis V_H
        _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        self.mean = mean
        self.basis = Vh[:actual_rank, :].T

        self.is_fitted = True
        return self

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes anomaly heatmaps and anomaly scores.
        Returns:
            anomaly_maps: Tensor of shape (B, 1, H, W)
            pred_scores: Tensor of shape (B,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted! Run fit() on healthy scans first.")

        images = images.to(self.basis.device)
        X_p = self.pruner.transform(images)

        # Per-sample affine normalization
        mu_sample = torch.mean(X_p, dim=1, keepdim=True)
        std_sample = torch.std(X_p, dim=1, keepdim=True) + 1e-6
        X_norm = (X_p - mu_sample) / std_sample

        # Orthogonal complement projection: r = (X_norm - mean) (I - V_k V_k^T)
        X_c = X_norm - self.mean
        X_proj = (X_c @ self.basis) @ self.basis.T
        residual = X_c - X_proj

        # Pixel-level anomaly maps (B, 1, H, W): Full-resolution residual preserves sharp tumor margins
        abs_res = torch.abs(residual)
        anomaly_maps = self.pruner.map_to_2d(abs_res).unsqueeze(1)
        smoothed_maps = self.blur(anomaly_maps)

        # Image-level anomaly score: top-100 brightest lesion pixels average (BMAD benchmark standard)
        B = smoothed_maps.shape[0]
        flat_maps = smoothed_maps.view(B, -1)
        k = min(100, flat_maps.shape[1])
        pred_scores = torch.topk(flat_maps, k=k, dim=1).values.mean(dim=1)

        return smoothed_maps, pred_scores
