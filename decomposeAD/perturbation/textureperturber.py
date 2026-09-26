from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from dataloader.datamodule import DATASETS_PATH as MVTEC_ROOT, _CLASSNAMES as MVTEC_CLASSES


DEFAULT_IMAGE = "/data/imageaddatasets/mvtec_anomaly_detection/carpet/train/good/000.png"
DEFAULT_PLOT = "/data/stepdown vision/plots/perturbation/texture_damage_preview.png"
RSNA_ROOT = "/data/imageaddatasets/rsna-pneumonia-detection-challenge"


def load_rgb_image(path: str | Path, image_size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    return np.array(image, dtype=np.uint8)


def save_preview_grid(images: list[np.ndarray], titles: list[str], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes.flatten()[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class TexturePerturber:
    def __init__(self, seed: int = 42, domain: str = "industrial"):
        self.rng = random.Random(seed)
        self.domain = domain

    def apply_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "original": image.copy(),
            "patch_shuffle": self.patch_shuffle(image),
            "local_blur_patch": self.local_blur_patch(image),
            "texture_noise_patch": self.texture_noise_patch(image),
            "elastic_texture_warp": self.elastic_texture_warp(image),
        }

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "medical":
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = gray > 10
            return mask.astype(bool)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if thresh[0, 0] == 255 and thresh[-1, -1] == 255:
            thresh = cv2.bitwise_not(thresh)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if mask.mean() < 0.15 or mask.mean() > 0.9:
            return np.ones_like(mask, dtype=bool)
        return mask.astype(bool)

    def _sample_patch_bounds(self, mask: np.ndarray, patch_scale: tuple[float, float]) -> tuple[int, int, int, int]:
        h, w = mask.shape
        patch_h = self.rng.randint(max(12, int(h * patch_scale[0])), max(18, int(h * patch_scale[1])))
        patch_w = self.rng.randint(max(12, int(w * patch_scale[0])), max(18, int(w * patch_scale[1])))

        ys, xs = np.where(mask)
        if len(xs) == 0:
            cx, cy = w // 2, h // 2
        else:
            idx = self.rng.randrange(len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])

        x1 = int(np.clip(cx - patch_w // 2, 0, max(0, w - patch_w)))
        y1 = int(np.clip(cy - patch_h // 2, 0, max(0, h - patch_h)))
        x2 = x1 + patch_w
        y2 = y1 + patch_h
        return x1, y1, x2, y2

    def _soft_patch_mask(self, h: int, w: int, blur: float = 7.0) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        mask[2:-2, 2:-2] = 1.0
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=blur, sigmaY=blur)

    def patch_shuffle(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        mask = self._foreground_mask(out)
        x1, y1, x2, y2 = self._sample_patch_bounds(mask, patch_scale=(0.12, 0.22))
        patch = out[y1:y2, x1:x2].copy()

        tile = self.rng.randint(8, 16) if self.domain == "industrial" else self.rng.randint(12, 24)
        patch_h, patch_w = patch.shape[:2]
        cell_h = max(4, patch_h // max(2, patch_h // tile))
        cell_w = max(4, patch_w // max(2, patch_w // tile))
        grid_h = patch_h // cell_h
        grid_w = patch_w // cell_w
        usable_h = grid_h * cell_h
        usable_w = grid_w * cell_w
        if grid_h < 2 or grid_w < 2:
            return out

        core = patch[:usable_h, :usable_w].copy()
        blocks = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                yy = gy * cell_h
                xx = gx * cell_w
                blocks.append(core[yy:yy + cell_h, xx:xx + cell_w].copy())
        self.rng.shuffle(blocks)

        shuffled = patch.copy()
        idx = 0
        for gy in range(grid_h):
            for gx in range(grid_w):
                yy = gy * cell_h
                xx = gx * cell_w
                shuffled[yy:yy + cell_h, xx:xx + cell_w] = blocks[idx]
                idx += 1

        alpha = self._soft_patch_mask(y2 - y1, x2 - x1, blur=5.0)
        roi = out[y1:y2, x1:x2].astype(np.float32)
        mixed = shuffled.astype(np.float32) * alpha[..., None] + roi * (1.0 - alpha[..., None])
        out[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        return out

    def local_blur_patch(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        mask = self._foreground_mask(out)
        x1, y1, x2, y2 = self._sample_patch_bounds(mask, patch_scale=(0.14, 0.26))
        patch = out[y1:y2, x1:x2]

        k = self.rng.choice([9, 13, 17]) if self.domain == "industrial" else self.rng.choice([13, 17, 21])
        sigma = self.rng.uniform(2.0, 4.0) if self.domain == "industrial" else self.rng.uniform(3.0, 6.0)
        blurred = cv2.GaussianBlur(patch, (k, k), sigmaX=sigma, sigmaY=sigma)

        alpha = self.rng.uniform(0.7, 0.95) * self._soft_patch_mask(y2 - y1, x2 - x1, blur=6.0)
        roi = out[y1:y2, x1:x2].astype(np.float32)
        mixed = blurred.astype(np.float32) * alpha[..., None] + roi * (1.0 - alpha[..., None])
        out[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        return out

    def texture_noise_patch(self, image: np.ndarray) -> np.ndarray:
        out = image.copy().astype(np.float32)
        mask = self._foreground_mask(image)
        x1, y1, x2, y2 = self._sample_patch_bounds(mask, patch_scale=(0.12, 0.24))
        patch = out[y1:y2, x1:x2]
        ph, pw = patch.shape[:2]

        noise = np.random.default_rng(self.rng.randint(0, 10**9)).normal(0.0, 1.0, size=(ph, pw)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=2.0, sigmaY=2.0)
        noise /= max(noise.std(), 1e-6)

        amplitude = self.rng.uniform(18.0, 35.0) if self.domain == "industrial" else self.rng.uniform(10.0, 24.0)
        alpha = self._soft_patch_mask(ph, pw, blur=6.0)
        delta = noise[..., None] * amplitude * alpha[..., None]
        if self.domain == "medical":
            patch = patch + delta
        else:
            patch = patch + delta + self.rng.uniform(-8.0, 8.0)
        out[y1:y2, x1:x2] = np.clip(patch, 0, 255)
        return out.astype(np.uint8)

    def elastic_texture_warp(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        mask = self._foreground_mask(out)
        x1, y1, x2, y2 = self._sample_patch_bounds(mask, patch_scale=(0.16, 0.28))
        patch = out[y1:y2, x1:x2]
        ph, pw = patch.shape[:2]

        rng = np.random.default_rng(self.rng.randint(0, 10**9))
        dx = rng.normal(0, 1, size=(ph, pw)).astype(np.float32)
        dy = rng.normal(0, 1, size=(ph, pw)).astype(np.float32)
        sigma = self.rng.uniform(5.0, 8.0) if self.domain == "industrial" else self.rng.uniform(7.0, 12.0)
        alpha_scale = self.rng.uniform(4.0, 8.0) if self.domain == "industrial" else self.rng.uniform(3.0, 6.0)
        dx = cv2.GaussianBlur(dx, (0, 0), sigmaX=sigma, sigmaY=sigma) * alpha_scale
        dy = cv2.GaussianBlur(dy, (0, 0), sigmaX=sigma, sigmaY=sigma) * alpha_scale

        xx, yy = np.meshgrid(np.arange(pw), np.arange(ph))
        map_x = np.clip((xx + dx).astype(np.float32), 0, pw - 1)
        map_y = np.clip((yy + dy).astype(np.float32), 0, ph - 1)
        warped = cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

        alpha = self.rng.uniform(0.75, 1.0) * self._soft_patch_mask(ph, pw, blur=7.0)
        roi = out[y1:y2, x1:x2].astype(np.float32)
        mixed = warped.astype(np.float32) * alpha[..., None] + roi * (1.0 - alpha[..., None])
        out[y1:y2, x1:x2] = np.clip(mixed, 0, 255).astype(np.uint8)
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="Preview texture perturbations on one RGB image.")
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default=DEFAULT_PLOT)
    parser.add_argument("--preview-suite", action="store_true")
    parser.add_argument("--domain", choices=["industrial", "medical"], default="industrial")
    return parser.parse_args()


def load_rsna_preview_image(path: str | Path, image_size: int) -> np.ndarray:
    try:
        from dataloader.imagedatamodule import load_rsna_image
    except Exception as exc:
        raise RuntimeError("RSNA preview requires the medical dataloader and DICOM dependencies.") from exc

    image = load_rsna_image(path, image_size=image_size)
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def _first_mvtec_train_image(cls: str) -> Path:
    train_good = Path(MVTEC_ROOT) / cls / "train" / "good"
    return sorted(train_good.glob("*"))[0]


def _sample_rsna_paths(limit: int = 6) -> list[Path]:
    image_dir = Path(RSNA_ROOT) / "stage_2_train_images"
    return sorted(image_dir.glob("*.dcm"))[:limit]


def build_preview_suite(image_size: int, seed: int):
    mvtec_perturber = TexturePerturber(seed=seed, domain="industrial")
    for cls in MVTEC_CLASSES:
        image_path = _first_mvtec_train_image(cls)
        image = load_rgb_image(image_path, image_size=image_size)
        outputs = mvtec_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/mvtec") / cls / "texture_damage_preview.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved mvtec texture preview for {cls}: {output_path}")

    rsna_perturber = TexturePerturber(seed=seed, domain="medical")
    for idx, image_path in enumerate(_sample_rsna_paths(limit=6), start=1):
        image = load_rsna_preview_image(image_path, image_size=image_size)
        outputs = rsna_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/rsna") / f"texture_sample_{idx}_{image_path.stem}.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved rsna texture preview {idx}: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.preview_suite:
        build_preview_suite(image_size=args.image_size, seed=args.seed)
    else:
        image = load_rgb_image(args.image_path, image_size=args.image_size)
        perturber = TexturePerturber(seed=args.seed, domain=args.domain)
        outputs = perturber.apply_all(image)
        titles = list(outputs.keys())
        images = [outputs[name] for name in titles]
        save_preview_grid(images, titles, args.output_path)
        print(f"saved preview to {args.output_path}")
