"""Torch time-series structure for VAD coordinate sensors.

The representation is feature/sensor-first:

    one item:  [sensors, timestamps]
    one batch: [batch, sensors, timestamps]
    labels:    [batch, timestamps]

For a 255x255 image and a timestamp window of 100, one item is therefore
``[65025, 100]`` and its labels are ``[100]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pandas import DataFrame, concat
from torch.utils.data import DataLoader, Dataset

from .data import make_supervised_vad_dataset


def resolve_dataset_root(root: str | Path) -> Path:
    root = Path(root)
    if (root / "train").is_dir():
        return root
    nested = root / "VAD"
    if (nested / "train").is_dir():
        return nested
    raise FileNotFoundError(f"Could not find VAD train split below {root}")


def select_training_samples(samples: DataFrame, regime: str, seed: int) -> DataFrame:
    """Select 2,000 normal plus 100/1,000 supervised bad images."""

    if regime not in {"low", "high"}:
        raise ValueError("regime must be 'low' or 'high'")

    normal = samples[samples["label"].astype(str) == "good"]
    bad = samples[samples["label"].astype(str) != "good"]
    bad_count = 100 if regime == "low" else 1000
    if len(normal) < 2000:
        raise ValueError(f"Expected at least 2,000 normal images, found {len(normal)}")
    if len(bad) < bad_count:
        raise ValueError(f"Requested {bad_count} bad images, found {len(bad)}")

    rng = np.random.default_rng(seed)
    selected_bad = bad.iloc[rng.choice(len(bad), size=bad_count, replace=False)]
    return concat([normal.iloc[:2000], selected_bad], ignore_index=True)


def load_sensor_vector(path: str | Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("L").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    return np.array(image, dtype=np.float32, copy=True).reshape(-1) / 255.0


class VADTimeSeriesDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Group VAD images into feature-by-timestamp windows."""

    def __init__(
        self,
        samples: DataFrame,
        image_size: int = 255,
        timestamps: int = 100,
        stride: int | None = None,
        drop_last: bool = True,
    ) -> None:
        if image_size < 1 or timestamps < 1:
            raise ValueError("image_size and timestamps must be positive")

        self.samples = samples.reset_index(drop=True)
        self.image_size = image_size
        self.timestamps = timestamps
        self.stride = stride or timestamps
        if self.stride < 1:
            raise ValueError("stride must be positive")

        n_samples = len(self.samples)
        if drop_last:
            self.starts = list(range(0, n_samples - timestamps + 1, self.stride))
        else:
            self.starts = list(range(0, n_samples, self.stride))
        if not self.starts:
            raise ValueError(f"Need at least {timestamps} samples, found {n_samples}")

        self.paths = self.samples["image_path"].astype(str).tolist()
        self.labels = self.samples["label_index"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        stop = min(start + self.timestamps, len(self.paths))
        vectors = [load_sensor_vector(path, self.image_size) for path in self.paths[start:stop]]
        features = np.stack(vectors, axis=1)
        labels = self.labels[start:stop]

        if features.shape[1] != self.timestamps:
            raise RuntimeError("Incomplete timestamp window encountered with drop_last=True")

        return torch.from_numpy(features), torch.from_numpy(labels.copy()).long()

    @property
    def n_sensors(self) -> int:
        return self.image_size * self.image_size


def make_vad_dataset(
    root: str | Path = "D:/lowrank/VAD/VAD",
    split: str = "train",
    regime: str = "high",
    seed: int = 0,
    image_size: int = 255,
    timestamps: int = 100,
    stride: int | None = None,
) -> VADTimeSeriesDataset:
    """Build a direct image-backed VAD time-series dataset."""

    dataset_root = resolve_dataset_root(root)
    samples = make_supervised_vad_dataset(dataset_root, split=split)
    if split == "train":
        samples = select_training_samples(samples, regime, seed)
    return VADTimeSeriesDataset(
        samples,
        image_size=image_size,
        timestamps=timestamps,
        stride=stride,
    )


def make_vad_dataloader(
    root: str | Path = "D:/lowrank/VAD/VAD",
    split: str = "train",
    regime: str = "high",
    seed: int = 0,
    image_size: int = 255,
    timestamps: int = 100,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a loader returning ``[B, sensors, timestamps]`` and ``[B, timestamps]``."""

    dataset = make_vad_dataset(
        root=root,
        split=split,
        regime=regime,
        seed=seed,
        image_size=image_size,
        timestamps=timestamps,
    )
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
