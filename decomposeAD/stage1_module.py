from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from dataloader.datamodule import (
    DATASETS_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESIZE_SIZE,
    DECOMPOSITION_CACHE_ROOT,
    MVTecTrainDataset,
    _ensure_decomposition_cache,
)
from loss import stage1loss
from models.restorationnet import AnalyticalRestorationNet


COMPONENT_NAMES = [
    "cb_detail",
    "cb_fill",
    "cr_detail",
    "cr_fill",
    "y_coarse_fill",
    "y_edge_detail",
    "y_mid_fill",
    "y_texture_detail",
]


class Stage1DataModule(pl.LightningDataModule):
    def __init__(
        self,
        mvtec_class: str,
        batch_size: int,
        num_workers: int,
        val_split: float,
        data_root: str = DATASETS_PATH,
        cache_root: str = DECOMPOSITION_CACHE_ROOT,
        seed: int = 42,
    ):
        super().__init__()
        self.mvtec_class = mvtec_class
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_size = DEFAULT_RESIZE_SIZE
        self.image_size = DEFAULT_IMAGE_SIZE
        self.val_split = val_split
        self.data_root = data_root
        self.cache_root = cache_root
        self.seed = seed
        self.full_train_size = 0
        self.inner_train_size = 0
        self.inner_val_size = 0

    def setup(self, stage: str | None = None):
        _ensure_decomposition_cache(
            cls=self.mvtec_class,
            source=self.data_root,
            cache_root=self.cache_root,
            resize_size=self.resize_size,
            image_size=self.image_size,
        )
        train_full = MVTecTrainDataset(
            cls=self.mvtec_class,
            source=self.data_root,
            cache_root=self.cache_root,
            resize=self.resize_size,
            imagesize=self.image_size,
        )
        self.full_train_size = len(train_full)

        val_size = max(1, int(self.full_train_size * self.val_split))
        train_size = self.full_train_size - val_size
        if train_size < 1:
            raise ValueError("Training split is empty. Reduce --val-split or use a larger dataset.")

        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(train_full, [train_size, val_size], generator=generator)
        self.inner_train_size = len(self.train_ds)
        self.inner_val_size = len(self.val_ds)

    def split_summary(self) -> dict:
        return {
            "mvtec_class": self.mvtec_class,
            "full_train_size": self.full_train_size,
            "inner_train_size": self.inner_train_size,
            "inner_val_size": self.inner_val_size,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class Stage1LightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        decomposition_weight: float = 1.0,
        fusion_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AnalyticalRestorationNet(in_channels=3, num_decomp_maps=len(COMPONENT_NAMES))
        self._train_running = {"loss": 0.0, "decomposition_loss": 0.0, "fusion_loss": 0.0, "count": 0}

    def forward(self, image: torch.Tensor):
        pred_maps, pred_rgb = self.model(image)
        predicted_components = OrderedDict(
            (name, pred_maps[:, idx : idx + 1]) for idx, name in enumerate(COMPONENT_NAMES)
        )
        return predicted_components, pred_rgb

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str):
        image = batch["image_raw"]
        target_components = batch["components"]
        batch_size = image.size(0)
        predicted_components, predicted_rgb = self(image)

        losses = stage1loss(
            predicted_components=predicted_components,
            target_components=target_components,
            predicted_rgb=predicted_rgb,
            target_rgb=image,
            decomposition_weight=self.hparams.decomposition_weight,
            fusion_weight=self.hparams.fusion_weight,
        )

        if stage == "train":
            self._train_running["count"] += 1
            self._train_running["loss"] += losses["loss"].detach().item()
            self._train_running["decomposition_loss"] += losses["decomposition_loss"].detach().item()
            self._train_running["fusion_loss"] += losses["fusion_loss"].detach().item()
            count = self._train_running["count"]
            self.log(
                "train/loss",
                self._train_running["loss"] / count,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                "train/decomposition_loss",
                self._train_running["decomposition_loss"] / count,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                "train/fusion_loss",
                self._train_running["fusion_loss"] / count,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        else:
            self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log(
                "val/decomposition_loss",
                losses["decomposition_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log(
                "val/fusion_loss",
                losses["fusion_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
            self.log("val_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return losses["loss"]

    def on_train_epoch_start(self):
        self._train_running = {"loss": 0.0, "decomposition_loss": 0.0, "fusion_loss": 0.0, "count": 0}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
