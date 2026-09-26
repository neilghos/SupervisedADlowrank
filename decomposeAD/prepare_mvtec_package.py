from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from dataloader.datamodule import DATASETS_PATH, DECOMPOSITION_CACHE_ROOT, _CLASSNAMES
from perturbation.precompute_stage2_cache import STAGE2_CACHE_ROOT


PRECOMPUTE_STAGE2_SCRIPT = "/data/stepdown vision/perturbation/precompute_stage2_cache.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-class", type=str, default="bottle")
    parser.add_argument("--data-root", type=str, default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", type=str, default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", type=str, default=str(STAGE2_CACHE_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_single_class(args, mvtec_class: str):
    command = [
        sys.executable,
        PRECOMPUTE_STAGE2_SCRIPT,
        "--mvtec-class",
        mvtec_class,
        "--data-root",
        args.data_root,
        "--clean-cache-root",
        args.clean_cache_root,
        "--stage2-cache-root",
        args.stage2_cache_root,
        "--seed",
        str(args.seed),
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    classes = _CLASSNAMES if args.mvtec_class == "all" else [args.mvtec_class]
    for index, mvtec_class in enumerate(classes, start=1):
        print(f"[prepare] class {index}/{len(classes)}: {mvtec_class}")
        prepare_single_class(args, mvtec_class)


if __name__ == "__main__":
    main()
