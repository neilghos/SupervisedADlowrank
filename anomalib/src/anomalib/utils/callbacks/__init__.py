"""Callbacks for Anomalib models."""

from __future__ import annotations

import logging
import os
import warnings

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .cdf_normalization import CdfNormalizationCallback
from .metrics_configuration import MetricsConfigurationCallback
from .min_max_normalization import MinMaxNormalizationCallback
from .model_loader import LoadModelCallback
from .post_processing_configuration import PostProcessingConfigurationCallback

__all__ = [
    "CdfNormalizationCallback",
    "LoadModelCallback",
    "MetricsConfigurationCallback",
    "MinMaxNormalizationCallback",
    "PostProcessingConfigurationCallback",
    "get_callbacks",
]

logger = logging.getLogger(__name__)


def get_callbacks(config: DictConfig | ListConfig) -> list[Callback]:
    """Return base callbacks for lightning models."""
    logger.info("Loading the callbacks")

    callbacks: list[Callback] = []

    monitor_metric = None if "early_stopping" not in config.model.keys() else config.model.early_stopping.metric
    monitor_mode = "max" if "early_stopping" not in config.model.keys() else config.model.early_stopping.mode

    enable_checkpointing = config.trainer.get("enable_checkpointing", True)
    if enable_checkpointing:
        checkpoint = ModelCheckpoint(
            dirpath=os.path.join(config.project.path, "weights"),
            filename="model",
            monitor=monitor_metric,
            mode=monitor_mode,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint)



    if "resume_from_checkpoint" in config.trainer.keys() and config.trainer.resume_from_checkpoint is not None:
        if os.path.exists(config.trainer.resume_from_checkpoint):
            load_model = LoadModelCallback(config.trainer.resume_from_checkpoint)
            callbacks.append(load_model)

    # Add post-processing configurations to AnomalyModule.
    image_threshold = (
        config.metrics.threshold.manual_image if "manual_image" in config.metrics.threshold.keys() else None
    )
    pixel_threshold = (
        config.metrics.threshold.manual_pixel if "manual_pixel" in config.metrics.threshold.keys() else None
    )
    post_processing_callback = PostProcessingConfigurationCallback(
        threshold_method=config.metrics.threshold.method,
        manual_image_threshold=image_threshold,
        manual_pixel_threshold=pixel_threshold,
    )
    callbacks.append(post_processing_callback)

    # Add metric configuration to the model via MetricsConfigurationCallback
    metrics_callback = MetricsConfigurationCallback(
        config.dataset.task,
        config.metrics.get("image", None),
        config.metrics.get("pixel", None),
    )
    callbacks.append(metrics_callback)

    if "normalization_method" in config.model.keys() and not config.model.normalization_method == "none":
        if config.model.normalization_method == "cdf":
            callbacks.append(CdfNormalizationCallback())
        elif config.model.normalization_method == "min_max":
            callbacks.append(MinMaxNormalizationCallback())

    return callbacks
