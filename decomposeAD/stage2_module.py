from __future__ import annotations

from collections import OrderedDict
import csv
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from dataloader.datamodule import (
    DATASETS_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESIZE_SIZE,
    DECOMPOSITION_CACHE_ROOT,
    MVTecTestDataset,
    MVTecTrainDataset,
    _ensure_decomposition_cache,
)
from loss import stage2loss
from mvtec_metric import compute_imagewise_retrieval_metrics
from models.restorationnet import AnalyticalRestorationNet
from perturbation.precompute_stage2_cache import (
    STAGE2_CACHE_ROOT,
    STAGE2_COMBINATIONS,
    stage2_cache_path,
    stage2_class_cache_root,
)


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

PRECOMPUTE_STAGE2_SCRIPT = "/data/stepdown vision/perturbation/precompute_stage2_cache.py"


def _stage2_cache_complete(cache_root: str, subset: Dataset, resize_size: int, image_size: int) -> bool:
    if not hasattr(subset, "indices") or not hasattr(subset, "dataset"):
        return False
    base_dataset = subset.dataset
    class_root = Path(base_dataset.class_root)
    cache_root_path = Path(cache_root)
    expected_count = len(base_dataset) * len(STAGE2_COMBINATIONS)
    count = sum(1 for _ in cache_root_path.rglob("*.pt")) if cache_root_path.exists() else 0
    if count != expected_count:
        return False
    probe_dataset_index = subset.indices[0]
    probe_image_path = Path(base_dataset.samples[probe_dataset_index][0])
    probe_path = stage2_cache_path(class_root, probe_image_path, cache_root_path, 0)
    if not probe_path.exists():
        return False
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    return (
        probe.get("resize_size") == resize_size
        and probe.get("image_size") == image_size
        and probe["perturbed_image"].shape[-2:] == (image_size, image_size)
        and next(iter(probe["perturbed_components"].values())).shape[-2:] == (image_size, image_size)
    )


def _ensure_stage2_cache(
    mvtec_class: str,
    data_root: str,
    clean_cache_root: str,
    stage2_cache_root: str,
    resize_size: int,
    image_size: int,
    train_subset: Dataset,
):
    cache_root = stage2_class_cache_root(stage2_cache_root, mvtec_class)
    if _stage2_cache_complete(str(cache_root), train_subset, resize_size, image_size):
        return

    command = [
        sys.executable,
        PRECOMPUTE_STAGE2_SCRIPT,
        "--mvtec-class",
        mvtec_class,
        "--data-root",
        data_root,
        "--clean-cache-root",
        clean_cache_root,
        "--stage2-cache-root",
        stage2_cache_root,
    ]
    if cache_root.exists():
        command.append("--overwrite")
    subprocess.run(command, check=True)


class Stage2CachedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, mvtec_class: str, stage2_cache_root: str):
        self.base_dataset = base_dataset
        self.mvtec_class = mvtec_class
        self.class_root = Path(base_dataset.dataset.class_root)
        self.cache_root = stage2_class_cache_root(stage2_cache_root, mvtec_class)
        self.combinations = STAGE2_COMBINATIONS

    def __len__(self) -> int:
        return len(self.base_dataset) * len(self.combinations)

    def __getitem__(self, index: int) -> dict[str, Any]:
        base_index = index // len(self.combinations)
        combo_index = index % len(self.combinations)
        dataset_index = self.base_dataset.indices[base_index]
        image_path = Path(self.base_dataset.dataset.samples[dataset_index][0])
        cache_file = stage2_cache_path(self.class_root, image_path, self.cache_root, combo_index)
        record = torch.load(cache_file, map_location="cpu", weights_only=False)

        return {
            "image_raw": record["image_raw"].float(),
            "perturbed_image": record["perturbed_image"].float(),
            "components": {name: value.float() for name, value in record["components"].items()},
            "perturbed_components": {name: value.float() for name, value in record["perturbed_components"].items()},
            "label": int(record["label"]),
            "path": record["source_path"],
            "surface_perturbations": record["surface_perturbations"],
            "color_perturbations": record["color_perturbations"],
            "combo_index": int(record["combo_index"]),
        }


class Stage2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        mvtec_class: str,
        batch_size: int,
        num_workers: int,
        val_split: float,
        data_root: str = DATASETS_PATH,
        clean_cache_root: str = DECOMPOSITION_CACHE_ROOT,
        stage2_cache_root: str = str(STAGE2_CACHE_ROOT),
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
        self.clean_cache_root = clean_cache_root
        self.stage2_cache_root = stage2_cache_root
        self.seed = seed
        self.full_train_size = 0
        self.inner_train_size = 0
        self.inner_val_size = 0
        self.augmented_train_size = 0
        self.augmented_val_size = 0

    def setup(self, stage: str | None = None):
        _ensure_decomposition_cache(
            cls=self.mvtec_class,
            source=self.data_root,
            cache_root=self.clean_cache_root,
            resize_size=self.resize_size,
            image_size=self.image_size,
        )

        if stage in (None, "fit"):
            train_full = MVTecTrainDataset(
                cls=self.mvtec_class,
                source=self.data_root,
                cache_root=self.clean_cache_root,
                resize=self.resize_size,
                imagesize=self.image_size,
            )
            self.full_train_size = len(train_full)

            val_size = max(1, int(self.full_train_size * self.val_split))
            train_size = self.full_train_size - val_size
            if train_size < 1:
                raise ValueError("Training split is empty. Reduce --val-split or use a larger dataset.")

            generator = torch.Generator().manual_seed(self.seed)
            train_subset, val_subset = random_split(train_full, [train_size, val_size], generator=generator)
            _ensure_stage2_cache(
                mvtec_class=self.mvtec_class,
                data_root=self.data_root,
                clean_cache_root=self.clean_cache_root,
                stage2_cache_root=self.stage2_cache_root,
                resize_size=self.resize_size,
                image_size=self.image_size,
                train_subset=train_subset,
            )
            self.train_ds = Stage2CachedDataset(train_subset, mvtec_class=self.mvtec_class, stage2_cache_root=self.stage2_cache_root)
            self.val_ds = Stage2CachedDataset(val_subset, mvtec_class=self.mvtec_class, stage2_cache_root=self.stage2_cache_root)
            self.inner_train_size = len(train_subset)
            self.inner_val_size = len(val_subset)
            self.augmented_train_size = len(self.train_ds)
            self.augmented_val_size = len(self.val_ds)

        if stage in (None, "test"):
            self.test_ds = MVTecTestDataset(
                cls=self.mvtec_class,
                source=self.data_root,
                cache_root=self.clean_cache_root,
                resize=self.resize_size,
                imagesize=self.image_size,
            )

    def split_summary(self) -> dict[str, int | str]:
        return {
            "mvtec_class": self.mvtec_class,
            "full_train_size": self.full_train_size,
            "inner_train_size": self.inner_train_size,
            "inner_val_size": self.inner_val_size,
            "augmented_train_size": self.augmented_train_size,
            "augmented_val_size": self.augmented_val_size,
            "perturbation_combinations": len(STAGE2_COMBINATIONS),
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

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class Stage2LightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        decomposition_weight: float = 1.0,
        fusion_weight: float = 1.0,
        stage1_checkpoint: str | None = None,
        artifact_dir: str | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AnalyticalRestorationNet(in_channels=3, num_decomp_maps=len(COMPONENT_NAMES))
        self._train_running = {"loss": 0.0, "decomposition_loss": 0.0, "fusion_loss": 0.0, "count": 0}
        self.stage1_checkpoint = stage1_checkpoint
        self._stage1_loaded = False
        self.test_rows: list[dict[str, Any]] = []

    def setup(self, stage: str | None = None):
        if self.stage1_checkpoint and not self._stage1_loaded:
            self._load_stage1_weights(self.stage1_checkpoint)
            self._stage1_loaded = True

    def _load_stage1_weights(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"[stage2 init] missing model keys from stage1 checkpoint: {missing}")
        if unexpected:
            print(f"[stage2 init] unexpected model keys from stage1 checkpoint: {unexpected}")
        print(f"[stage2 init] loaded stage1 weights from {checkpoint_path}")

    def forward(self, image: torch.Tensor):
        pred_maps, pred_rgb = self.model(image)
        predicted_components = OrderedDict(
            (name, pred_maps[:, idx : idx + 1]) for idx, name in enumerate(COMPONENT_NAMES)
        )
        return predicted_components, pred_rgb

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str):
        perturbed_image = batch["perturbed_image"]
        perturbed_components = batch["perturbed_components"]
        target_rgb = batch["image_raw"]
        target_components = batch["components"]
        batch_size = perturbed_image.size(0)
        _ = perturbed_components

        predicted_components, predicted_rgb = self(perturbed_image)
        losses = stage2loss(
            predicted_components=predicted_components,
            target_components=target_components,
            predicted_rgb=predicted_rgb,
            target_rgb=target_rgb,
            decomposition_weight=self.hparams.decomposition_weight,
            fusion_weight=self.hparams.fusion_weight,
        )

        if stage == "train":
            self._train_running["count"] += 1
            self._train_running["loss"] += losses["loss"].detach().item()
            self._train_running["decomposition_loss"] += losses["decomposition_loss"].detach().item()
            self._train_running["fusion_loss"] += losses["fusion_loss"].detach().item()
            count = self._train_running["count"]
            self.log("train/loss", self._train_running["loss"] / count, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("train/decomposition_loss", self._train_running["decomposition_loss"] / count, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("train/fusion_loss", self._train_running["fusion_loss"] / count, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        else:
            self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("val/decomposition_loss", losses["decomposition_loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("val/fusion_loss", losses["fusion_loss"], on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
            self.log("val_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return losses["loss"]

    def on_train_epoch_start(self):
        self._train_running = {"loss": 0.0, "decomposition_loss": 0.0, "fusion_loss": 0.0, "count": 0}

    def on_test_epoch_start(self):
        self.test_rows = []

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int):
        input_image = batch["image_raw"]
        _, restored_rgb = self(input_image)
        per_sample_mse = F.mse_loss(restored_rgb, input_image, reduction="none").flatten(1).mean(dim=1)

        labels = batch["label"].detach().cpu().tolist()
        paths = batch["path"]
        for index, score in enumerate(per_sample_mse.detach().cpu().tolist()):
            self.test_rows.append(
                {
                    "path": paths[index],
                    "label": int(labels[index]),
                    "input_restore_mse": float(score),
                }
            )

        self.log("test/input_restore_mse", per_sample_mse.mean(), on_step=False, on_epoch=True, prog_bar=True, batch_size=input_image.size(0))
        return per_sample_mse.mean()

    def on_test_epoch_end(self):
        if not self.test_rows:
            return

        artifact_dir = Path(self.hparams.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        normal_scores = [row["input_restore_mse"] for row in self.test_rows if row["label"] == 0]
        anomaly_scores = [row["input_restore_mse"] for row in self.test_rows if row["label"] == 1]
        all_scores = [row["input_restore_mse"] for row in self.test_rows]
        all_labels = [row["label"] for row in self.test_rows]

        csv_path = artifact_dir / "test_input_restore_scores.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["path", "label", "input_restore_mse"])
            writer.writeheader()
            writer.writerows(self.test_rows)

        auroc = None
        if len(set(all_labels)) > 1:
            auroc = float(compute_imagewise_retrieval_metrics(all_scores, all_labels)["auroc"])

        summary_path = artifact_dir / "test_input_restore_summary.txt"
        with summary_path.open("w") as handle:
            handle.write(f"total_samples: {len(self.test_rows)}\n")
            handle.write(f"normal_count: {len(normal_scores)}\n")
            handle.write(f"anomaly_count: {len(anomaly_scores)}\n")
            if normal_scores:
                handle.write(f"normal_mean_mse: {float(np.mean(normal_scores)):.8f}\n")
                handle.write(f"normal_std_mse: {float(np.std(normal_scores)):.8f}\n")
            if anomaly_scores:
                handle.write(f"anomaly_mean_mse: {float(np.mean(anomaly_scores)):.8f}\n")
                handle.write(f"anomaly_std_mse: {float(np.std(anomaly_scores)):.8f}\n")
            if auroc is not None:
                handle.write(f"image_auroc: {auroc:.8f}\n")

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            if normal_scores:
                plt.hist(normal_scores, bins=40, alpha=0.6, label="normal", color="tab:blue")
            if anomaly_scores:
                plt.hist(anomaly_scores, bins=40, alpha=0.6, label="anomaly", color="tab:red")
            plt.xlabel("MSE(input, corrected)")
            plt.ylabel("Count")
            plt.title("Stage 2 test-score separation")
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(artifact_dir / "test_input_restore_hist.png", dpi=200)
            plt.close()
        except Exception as exc:
            print(f"[stage2 test] failed to save histogram: {exc}")

        if normal_scores:
            self.log("test/normal_mean_mse", float(np.mean(normal_scores)), prog_bar=False)
        if anomaly_scores:
            self.log("test/anomaly_mean_mse", float(np.mean(anomaly_scores)), prog_bar=False)
        if auroc is not None:
            self.log("test/image_auroc", auroc, prog_bar=True)

        print(f"[stage2 test] wrote scores to {csv_path}")
        print(f"[stage2 test] wrote summary to {summary_path}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    dm = Stage2DataModule(
        mvtec_class="bottle",
        batch_size=2,
        num_workers=0,
        val_split=0.1,
    )
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    print("perturbed_image:", tuple(batch["perturbed_image"].shape))
    print("image_raw:", tuple(batch["image_raw"].shape))
    print("component keys:", sorted(batch["components"].keys()))
    print("surface perturbations:", batch["surface_perturbations"])
    print("color perturbations:", batch["color_perturbations"])
    print("combo count:", len(STAGE2_COMBINATIONS))
