"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace
from pytorch_lightning.profiler import SimpleProfiler, PyTorchProfiler
import os
import sys
import types
import warnings
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

if "simsimd" not in sys.modules:
    sys.modules["simsimd"] = types.ModuleType("simsimd")

# Prevent broken tensorboard imports
for mod in ["tensorboard", "tensorboard.compat", "tensorboard.compat.notf"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of test batches (e.g. 1)")

    args = parser.parse_args()
    return args


def test():
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    args = get_args()
    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        weight_file=args.weight_file,
    )

    if config.project.seed:
        seed_everything(config.project.seed)
    
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project=config.dataset.name,
    #     name = config.model.name + '_' + config.model.backbone,
    # )

    datamodule = get_datamodule(config)
    model = get_model(config)
    #profiler = PyTorchProfiler(record_functions={"test_step"})
    #profiler = SimpleProfiler
    callbacks = get_callbacks(config)
    trainer_kwargs = dict(config.trainer)
    if trainer_kwargs.get("resume_from_checkpoint") is None:
        trainer_kwargs.pop("resume_from_checkpoint", None)
    trainer_kwargs["logger"] = False
    if args.limit_batches is not None:
        trainer_kwargs["limit_test_batches"] = args.limit_batches
    trainer = Trainer(callbacks=callbacks, **trainer_kwargs)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    test()
