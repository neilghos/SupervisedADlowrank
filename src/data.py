"""Supervised VAD datamodule with train/bad support."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pandas import DataFrame
from torch.utils.data import DataLoader

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.base import AnomalibDataset
from anomalib.data.datamodules.image.vad import DOWNLOAD_INFO
from anomalib.data.utils import (
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    download_and_extract,
    validate_path,
)
from anomalib.utils.path import resolve_dataset_root


IMG_EXTENSIONS = (".png", ".PNG")


def make_supervised_vad_dataset(
    root: str | Path,
    split: str | Split | None = None,
) -> DataFrame:
    """Build a VAD sample table while retaining supervised train anomalies."""
    root = validate_path(root)
    files = [
        file_path
        for file_path in root.glob("**/*")
        if file_path.is_file() and file_path.suffix in IMG_EXTENSIONS
    ]
    if not files:
        raise RuntimeError(f"Found 0 VAD images in {root}")

    rows = [(str(root), *file_path.parts[-3:]) for file_path in files]
    samples = DataFrame(rows, columns=["path", "split", "label", "image_path"])
    samples["image_path"] = (
        samples["path"]
        + "/"
        + samples["split"]
        + "/"
        + samples["label"]
        + "/"
        + samples["image_path"]
    )
    samples["label_index"] = (samples["label"] != "good").astype(int)

    masks = samples[samples["split"] == "ground_truth"]
    mask_paths = masks["image_path"].tolist()
    samples = samples[samples["split"] != "ground_truth"].copy()
    samples["mask_path"] = None

    def find_mask(image_path: str) -> str | None:
        image_stem = Path(image_path).stem
        for mask_path in mask_paths:
            if image_stem in Path(mask_path).stem:
                return mask_path
        return None

    test_abnormal = (samples["split"] == "test") & (
        samples["label_index"] == int(LabelName.ABNORMAL)
    )
    samples.loc[test_abnormal, "mask_path"] = samples.loc[
        test_abnormal, "image_path"
    ].map(find_mask)
    samples.attrs["task"] = "classification"

    if split is not None:
        split_value = split.value if isinstance(split, Split) else split
        samples = samples[samples["split"] == split_value].reset_index(drop=True)
        samples.attrs["task"] = "classification"
    return samples


class SupervisedVADDataset(AnomalibDataset):
    """Dataset wrapper for VAD's normal and supervised-bad training images."""

    def __init__(
        self,
        root: Path | str,
        category: str = "vad",
        augmentations=None,
        split=None,
    ) -> None:
        super().__init__(augmentations=augmentations)
        self.root_category = Path(root) / category
        self.category = category
        self.split = split
        self.samples = make_supervised_vad_dataset(self.root_category, split=split)


class SupervisedVAD(AnomalibDataModule):
    """VAD datamodule using 2,000 normal images plus a bad-image regime."""

    def __init__(
        self,
        root: Path | str = "./datasets/VAD",
        category: str = "vad",
        regime: str = "low",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if regime not in {"low", "high"}:
            raise ValueError("regime must be 'low' or 'high'")

        super().__init__(
            train_batch_size=kwargs.pop("train_batch_size", 32),
            eval_batch_size=kwargs.pop("eval_batch_size", 32),
            num_workers=kwargs.pop("num_workers", 8),
            train_augmentations=kwargs.pop("train_augmentations", None),
            val_augmentations=kwargs.pop("val_augmentations", None),
            test_augmentations=kwargs.pop("test_augmentations", None),
            augmentations=kwargs.pop("augmentations", None),
            test_split_mode=kwargs.pop("test_split_mode", TestSplitMode.FROM_DIR),
            val_split_mode=kwargs.pop("val_split_mode", ValSplitMode.SAME_AS_TEST),
            seed=seed,
        )
        if kwargs:
            raise TypeError(f"Unexpected SupervisedVAD arguments: {sorted(kwargs)}")

        self.root = Path(resolve_dataset_root(root, "VAD"))
        self.category = category
        self.regime = regime
        self.seed = seed

    @property
    def train_bad_count(self) -> int:
        return 100 if self.regime == "low" else 1000

    def prepare_data(self) -> None:
        if not (self.root / self.category).is_dir():
            download_and_extract(self.root, DOWNLOAD_INFO)

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = SupervisedVADDataset(
            self.root,
            self.category,
            self.train_augmentations,
            Split.TRAIN,
        )
        self.test_data = SupervisedVADDataset(
            self.root,
            self.category,
            self.test_augmentations,
            Split.TEST,
        )

        samples = self.train_data.samples
        normal_indices = np.flatnonzero(samples["label_index"].to_numpy() == 0)
        bad_indices = np.flatnonzero(samples["label_index"].to_numpy() == 1)
        if len(normal_indices) < 2000:
            raise ValueError(
                f"VAD requires 2,000 normal train images, found {len(normal_indices)}"
            )
        if len(bad_indices) < self.train_bad_count:
            raise ValueError(
                f"Requested {self.train_bad_count} bad images, found {len(bad_indices)}"
            )

        rng = np.random.default_rng(self.seed)
        chosen_bad = rng.choice(bad_indices, size=self.train_bad_count, replace=False)
        self.train_data = self.train_data.subsample(
            np.concatenate([normal_indices, chosen_bad]).tolist()
        )

    def train_dataloader(self):
        """Return a stable order because the model makes multiple training passes."""
        return DataLoader(
            dataset=self.train_data,
            shuffle=False,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.train_data.collate_fn,
        )


def describe_vad_test_split(datamodule: SupervisedVAD) -> dict[str, int]:
    """Return counts for the three VAD test groups."""
    if not hasattr(datamodule, "test_data"):
        datamodule.setup()
    labels = datamodule.test_data.samples["label"].astype(str)
    return {
        "normal": int((labels == "good").sum()),
        "seen_defects": int((labels == "bad").sum()),
        "unseen_defects": int((labels == "bad_unseen_defects").sum()),
    }
