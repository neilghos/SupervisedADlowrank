"""Anomalib Datasets - Cleaned for Medical Benchmark."""

from __future__ import annotations

import logging
from omegaconf import DictConfig, ListConfig

from .base import AnomalibDataModule, AnomalibDataset
from .folder import Folder
from .task_type import TaskType

logger = logging.getLogger(__name__)

__all__ = [
    "AnomalibDataModule",
    "AnomalibDataset",
    "Folder",
    "TaskType",
    "get_datamodule",
]


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule for Medical Datasets."""
    logger.info("Loading the datamodule")

    # convert center crop to tuple if specified
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    image_size = config.dataset.image_size
    if isinstance(image_size, (int, float)):
        image_size = (int(image_size), int(image_size))
    else:
        image_size = (image_size[0], image_size[1])

    root = config.dataset.get("root") or config.dataset.get("path")
    mask_dir = config.dataset.get("mask_dir") or config.dataset.get("mask")

    transform_train = None
    transform_eval = None
    if "transform_config" in config.dataset and config.dataset.transform_config:
        transform_train = config.dataset.transform_config.get("train")
        transform_eval = config.dataset.transform_config.get("eval")

    datamodule = Folder(
        root=root,
        normal_dir=config.dataset.normal_dir,
        abnormal_dir=config.dataset.abnormal_dir,
        task=config.dataset.task,
        normal_test_dir=config.dataset.get("normal_test_dir"),
        mask_dir=mask_dir,
        extensions=config.dataset.get("extensions"),
        image_size=image_size,
        center_crop=center_crop,
        normalization=config.dataset.normalization,
        train_batch_size=config.dataset.train_batch_size,
        eval_batch_size=config.dataset.eval_batch_size,
        num_workers=config.dataset.num_workers,
        transform_config_train=transform_train,
        transform_config_eval=transform_eval,
        test_split_mode=config.dataset.test_split_mode,
        test_split_ratio=config.dataset.test_split_ratio,
        val_split_mode=config.dataset.val_split_mode,
        val_split_ratio=config.dataset.val_split_ratio,
        seed=config.dataset.get("seed"),
    )

    return datamodule
