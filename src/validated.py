"""Validation split for spatial MIL training."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from lowrank_vad.data import SupervisedVAD


class ValidatedSupervisedVAD(SupervisedVAD):
    """Use a deterministic stratified validation split from the train pool."""

    def __init__(self, *args, val_fraction: float = 0.20, **kwargs) -> None:
        if not 0.0 < val_fraction < 0.5:
            raise ValueError("val_fraction must be between 0 and 0.5")
        super().__init__(*args, **kwargs)
        self.val_fraction = val_fraction
        self._validated_train_data = None
        self._validated_val_data = None

    def _setup(self, stage: str | None = None) -> None:
        super()._setup(stage)
        selected = self.train_data
        labels = selected.samples["label_index"].to_numpy()
        rng = np.random.default_rng(self.seed + 17)
        validation_indices: list[int] = []
        training_indices: list[int] = []

        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            validation_count = max(1, int(round(len(indices) * self.val_fraction)))
            validation_indices.extend(indices[:validation_count].tolist())
            training_indices.extend(indices[validation_count:].tolist())

        self._validated_val_data = selected.subsample(validation_indices)
        self._validated_train_data = selected.subsample(training_indices)
        self.train_data = self._validated_train_data
        self.val_data = self._validated_val_data

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        self.train_data = self._validated_train_data
        self.val_data = self._validated_val_data

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            shuffle=False,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.val_data.collate_fn,
        )
