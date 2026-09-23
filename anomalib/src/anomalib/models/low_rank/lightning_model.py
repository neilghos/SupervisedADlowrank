"""Low-Rank Residual Decomposition Lightning Model."""

from __future__ import annotations

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Any
import torch
from torch import Tensor
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from anomalib.models.components import AnomalyModule
from .torch_model import LowRankModel

logger = logging.getLogger(__name__)


class LowRank(AnomalyModule):
    """
    Low-Rank Residual Decomposition Lightning Module.
    """

    def __init__(
        self,
        rank: int = 50,
        var_threshold: float = 1e-4,
        score_type: str = "l1",
    ) -> None:
        super().__init__()
        self.model = LowRankModel(rank=rank, var_threshold=var_threshold, score_type=score_type)
        self.train_images: list[Tensor] = []

    def configure_optimizers(self) -> None:
        """Closed-form decomposition requires no optimizer."""
        return None

    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> None:
        """Accumulate healthy normal scans during training."""
        del args, kwargs
        self.train_images.append(batch["image"])

    def on_validation_start(self) -> None:
        """Fit spatial pruner and closed-form SVD basis on healthy training scans."""
        if len(self.train_images) > 0:
            logger.info("Aggregating healthy scans for closed-form subspace fit.")
            stacked = torch.vstack(self.train_images).to(self.device)
            self.train_images.clear()  # Free memory
            self.model.fit(stacked)
            logger.info(f"Basis V_H extracted with shape: {self.model.basis.shape}")

    def on_test_start(self) -> None:
        """Fit single healthy subspace basis strictly on healthy training scans."""
        if not self.model.is_fitted:
            logger.info("Fitting LowRank basis on healthy training scans...")
            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is not None:
                datamodule.setup(stage="fit")
                train_loader = datamodule.train_dataloader()
                train_batches = []
                for batch in train_loader:
                    train_batches.append(batch["image"])
                    if sum(b.shape[0] for b in train_batches) >= 800:
                        break
                stacked = torch.vstack(train_batches).to(self.device)
                self.model.fit(stacked)
                logger.info(f"Basis V_H fitted: {self.model.basis.shape}")

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        """Compute anomaly maps and scores for incoming test/validation batch."""
        del args, kwargs
        anomaly_maps, pred_scores = self.model(batch["image"])
        batch["anomaly_maps"] = anomaly_maps
        batch["pred_scores"] = pred_scores
        batch["pred_boxes"] = [torch.empty((0, 4), device=batch["image"].device)] * len(pred_scores)
        batch["box_scores"] = [torch.empty((0,), device=batch["image"].device)] * len(pred_scores)
        batch["box_labels"] = [torch.empty((0,), device=batch["image"].device)] * len(pred_scores)
        return batch


class LowRankLightning(LowRank):
    """Entry point for BMAD config parser."""

    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        super().__init__(
            rank=getattr(hparams.model, "rank", 50),
            var_threshold=getattr(hparams.model, "var_threshold", 1e-4),
            score_type=getattr(hparams.model, "score_type", "l1"),
        )
        self.hparams: DictConfig | ListConfig
        self.save_hyperparameters(hparams)
