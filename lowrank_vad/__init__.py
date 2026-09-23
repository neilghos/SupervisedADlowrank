"""Supervised low-rank anomaly detection on the VAD benchmark."""

from .data import SupervisedVAD, describe_vad_test_split
from .model import SupervisedLowRank

__all__ = ["SupervisedVAD", "SupervisedLowRank", "describe_vad_test_split"]
