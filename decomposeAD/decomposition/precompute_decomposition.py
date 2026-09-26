import argparse
from pathlib import Path
import sys

import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import decomposition.lab_stack as lab_stack
import decomposition.ycbcr_stack as ycbcr_stack
from dataloader.datamodule import DEFAULT_IMAGE_SIZE, DEFAULT_RESIZE_SIZE


DATA_ROOT = Path("/data/stepdown vision/data/imagenette/imagenette2-320")
CACHE_ROOT = Path("/data/stepdown vision/cache/decompositions/imagenette")


def output_path(cache_root: Path, data_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(data_root).with_suffix(".pt")
    return cache_root / relative


def resolve_stack(args):
    if args.lab:
        return "lab", lab_stack
    return "ycbcr", ycbcr_stack


def precompute(args):
    stack_name, stack_module = resolve_stack(args)
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root) / stack_name
    if args.split == "all":
        splits = [split for split in ("train", "val", "test") if (data_root / split).exists()]
    else:
        splits = [args.split]

    for split in splits:
        samples = stack_module.iter_samples(data_root, split)
        progress = tqdm(samples, desc=f"{stack_name} {split}", unit="image")
        for image_path, class_name, label in progress:
            destination = output_path(cache_root, data_root, image_path)
            if destination.exists() and not args.overwrite:
                continue

            image = stack_module.load_image(image_path, DEFAULT_RESIZE_SIZE, DEFAULT_IMAGE_SIZE)
            components = stack_module.decompose(
                image=image,
                mid_blur_kernel=args.mid_blur_kernel,
                mid_blur_sigma=args.mid_blur_sigma,
                coarse_blur_kernel=args.coarse_blur_kernel,
                coarse_blur_sigma=args.coarse_blur_sigma,
                edge_gamma=args.edge_gamma,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "stack": stack_name,
                    "resize_size": DEFAULT_RESIZE_SIZE,
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "image": image,
                    "components": components,
                    "label": label,
                    "class_name": class_name,
                    "source_path": str(image_path),
                },
                destination,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute additive decomposition tensors.")
    stack_group = parser.add_mutually_exclusive_group()
    stack_group.add_argument("--ycbcr", "--yrcrb", "--ycrcb", dest="ycbcr", action="store_true")
    stack_group.add_argument("--lab", action="store_true")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--mid-blur-kernel", type=int, default=17)
    parser.add_argument("--mid-blur-sigma", type=float, default=4.0)
    parser.add_argument("--coarse-blur-kernel", type=int, default=41)
    parser.add_argument("--coarse-blur-sigma", type=float, default=10.0)
    parser.add_argument("--edge-gamma", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    precompute(parse_args())
