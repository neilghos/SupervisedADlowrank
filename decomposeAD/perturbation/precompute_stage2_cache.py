from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path
import sys

import torch
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from dataloader.datamodule import (
    DATASETS_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_RESIZE_SIZE,
    DECOMPOSITION_CACHE_ROOT,
    _ensure_decomposition_cache,
)
from determinism import make_deterministic
from decomposition import ycbcr_stack
from perturbation.colorperturber import ColorPerturber
from perturbation.surfaceperturber import SurfaceDamagePerturber


STAGE2_CACHE_ROOT = Path("/data/cache/stage2/mvtec")

SURFACE_PERTURBATIONS = [
    "scratches",
    "cuts",
    "cracks",
    "holes",
    "dents",
    "thread_damage",
]

COLOR_PERTURBATIONS = [
    "stain_chroma_blob",
    "local_discoloration",
    "burn_patch",
    "contamination_speckle",
]

STAGE2_COMBINATIONS = [
    {
        "surface": surface_combo,
        "color": color_combo,
    }
    for surface_combo in combinations(SURFACE_PERTURBATIONS, 2)
    for color_combo in combinations(COLOR_PERTURBATIONS, 2)
]


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:12], 16) % (10**9)


def stage2_class_cache_root(cache_root: str | Path, cls: str) -> Path:
    return Path(cache_root) / cls


def clean_cache_path(class_root: Path, image_path: Path, clean_cache_root: str) -> Path:
    relative = image_path.relative_to(class_root).with_suffix(".pt")
    return Path(clean_cache_root) / relative


def stage2_cache_path(class_root: Path, image_path: Path, stage2_cache_root: str | Path, combo_index: int) -> Path:
    relative = image_path.relative_to(class_root)
    return Path(stage2_cache_root) / relative.parent / f"{relative.stem}__combo_{combo_index:03d}.pt"


def tensor_to_uint8_image(image: torch.Tensor):
    import numpy as np

    image = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)


def uint8_image_to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0


def _cache_complete(
    cache_root: Path,
    class_root: Path,
    samples: list[tuple[Path, str, int]],
    resize_size: int,
    image_size: int,
) -> bool:
    expected_count = len(samples) * len(STAGE2_COMBINATIONS)
    if not cache_root.exists():
        return False
    count = sum(1 for _ in cache_root.rglob("*.pt"))
    if count != expected_count:
        return False

    probe_path = stage2_cache_path(class_root, samples[0][0], cache_root, 0)
    if not probe_path.exists():
        return False
    sample = torch.load(probe_path, map_location="cpu", weights_only=False)
    return (
        "perturbed_image" in sample
        and "perturbed_components" in sample
        and sample.get("resize_size") == resize_size
        and sample.get("image_size") == image_size
        and sample["perturbed_image"].shape[-2:] == (image_size, image_size)
        and next(iter(sample["perturbed_components"].values())).shape[-2:] == (image_size, image_size)
    )


def precompute(args):
    make_deterministic(args.seed)
    class_root = Path(args.data_root) / args.mvtec_class
    clean_cache_root = Path(args.clean_cache_root).parent / args.mvtec_class / Path(args.clean_cache_root).name
    stage2_cache_root = stage2_class_cache_root(args.stage2_cache_root, args.mvtec_class)

    _ensure_decomposition_cache(
        cls=args.mvtec_class,
        source=args.data_root,
        cache_root=args.clean_cache_root,
    )

    samples = ycbcr_stack.iter_samples(class_root, "train")
    if _cache_complete(stage2_cache_root, class_root, samples, DEFAULT_RESIZE_SIZE, DEFAULT_IMAGE_SIZE) and not args.overwrite:
        print(f"[stage2 cache] already complete: {stage2_cache_root}")
        return

    progress = tqdm(samples, desc=f"stage2 {args.mvtec_class}", unit="image")
    for image_path, class_name, label in progress:
        clean_record = torch.load(
            clean_cache_path(class_root, image_path, str(clean_cache_root)),
            map_location="cpu",
            weights_only=False,
        )
        clean_image = clean_record["image"].float()
        clean_components = {name: value.float() for name, value in clean_record["components"].items()}
        image_np = tensor_to_uint8_image(clean_image)

        for combo_index, combo in enumerate(STAGE2_COMBINATIONS):
            destination = stage2_cache_path(class_root, image_path, stage2_cache_root, combo_index)
            if destination.exists() and not args.overwrite:
                continue

            combo_image = image_np.copy()
            seed_prefix = f"{args.seed}:{args.mvtec_class}:{image_path}:{combo_index}"

            for stage_index, perturbation_name in enumerate(combo["surface"]):
                surface_perturber = SurfaceDamagePerturber(
                    seed=stable_seed(seed_prefix, "surface", stage_index, perturbation_name),
                    domain="industrial",
                )
                combo_image = getattr(surface_perturber, perturbation_name)(combo_image)

            for stage_index, perturbation_name in enumerate(combo["color"]):
                color_perturber = ColorPerturber(
                    seed=stable_seed(seed_prefix, "color", stage_index, perturbation_name),
                    domain="industrial",
                    class_hint=args.mvtec_class,
                )
                combo_image = getattr(color_perturber, perturbation_name)(combo_image)

            perturbed_image = uint8_image_to_tensor(combo_image)
            perturbed_components = ycbcr_stack.decompose(
                image=perturbed_image,
                mid_blur_kernel=args.mid_blur_kernel,
                mid_blur_sigma=args.mid_blur_sigma,
                coarse_blur_kernel=args.coarse_blur_kernel,
                coarse_blur_sigma=args.coarse_blur_sigma,
                edge_gamma=args.edge_gamma,
            )

            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "resize_size": DEFAULT_RESIZE_SIZE,
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "image_raw": clean_image,
                    "perturbed_image": perturbed_image,
                    "components": clean_components,
                    "perturbed_components": perturbed_components,
                    "label": label,
                    "class_name": class_name,
                    "source_path": str(image_path),
                    "surface_perturbations": list(combo["surface"]),
                    "color_perturbations": list(combo["color"]),
                    "combo_index": combo_index,
                },
                destination,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute stage-2 perturbed image and decomposition cache.")
    parser.add_argument("--mvtec-class", required=True)
    parser.add_argument("--data-root", default=DATASETS_PATH)
    parser.add_argument("--clean-cache-root", default=DECOMPOSITION_CACHE_ROOT)
    parser.add_argument("--stage2-cache-root", default=str(STAGE2_CACHE_ROOT))
    parser.add_argument("--mid-blur-kernel", type=int, default=17)
    parser.add_argument("--mid-blur-sigma", type=float, default=4.0)
    parser.add_argument("--coarse-blur-kernel", type=int, default=41)
    parser.add_argument("--coarse-blur-sigma", type=float, default=10.0)
    parser.add_argument("--edge-gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    precompute(parse_args())
