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


DEFAULT_IMAGE = "/data/imageaddatasets/mvtec_anomaly_detection/bottle/train/good/000.png"
DEFAULT_PLOT = "/data/stepdown vision/plots/perturbation/color_damage_preview.png"
RSNA_ROOT = "/data/imageaddatasets/rsna-pneumonia-detection-challenge"
TEXTURE_CLASSES = {"carpet", "grid", "leather", "tile", "wood"}


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
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes.flatten()[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class ColorPerturber:
    def __init__(self, seed: int = 42, domain: str = "industrial", class_hint: str | None = None):
        self.rng = random.Random(seed)
        self.domain = domain
        self.class_hint = class_hint

    def apply_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "original": image.copy(),
            "stain_chroma_blob": self.stain_chroma_blob(image),
            "local_discoloration": self.local_discoloration(image),
            "burn_patch": self.burn_patch(image),
            "contamination_speckle": self.contamination_speckle(image),
        }

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "medical":
            return np.ones(image.shape[:2], dtype=bool)

        if self.class_hint in TEXTURE_CLASSES:
            return np.ones(image.shape[:2], dtype=bool)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if thresh[0, 0] == 255 and thresh[-1, -1] == 255:
            thresh = cv2.bitwise_not(thresh)

        mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask.astype(bool)

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = int(component_areas.argmax()) + 1
        largest_mask = labels == largest_idx
        if largest_mask.mean() < 0.01:
            return mask.astype(bool)
        return largest_mask

    def _sample_point_in_mask(self, mask: np.ndarray) -> tuple[int, int]:
        ys, xs = np.where(mask)
        idx = self.rng.randrange(len(xs))
        return int(xs[idx]), int(ys[idx])

    def _soft_blob_mask(self, h: int, w: int, cx: int, cy: int, rx: float, ry: float) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - cx) ** 2) / (2 * rx * rx) + ((yy - cy) ** 2) / (2 * ry * ry)))
        return blob.astype(np.float32)

    def _random_blob(self, image: np.ndarray, size_scale: tuple[float, float]):
        h, w = image.shape[:2]
        mask = self._foreground_mask(image)
        cx, cy = self._sample_point_in_mask(mask)
        rx = self.rng.uniform(max(4.0, w * size_scale[0]), max(6.0, w * size_scale[1]))
        ry = self.rng.uniform(max(4.0, h * size_scale[0]), max(6.0, h * size_scale[1]))
        blob = self._soft_blob_mask(h, w, cx, cy, rx, ry)
        blob *= mask.astype(np.float32)
        blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=max(1.5, rx / 3), sigmaY=max(1.5, ry / 3))
        blob *= mask.astype(np.float32)
        return blob, mask

    def stain_chroma_blob(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "medical":
            return self._medical_lesion_blob(image, mode="bright")

        out = image.astype(np.float32).copy()
        blob, _ = self._random_blob(out.astype(np.uint8), size_scale=(0.03, 0.08))
        hsv = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        hue_shift = self.rng.uniform(-18.0, 18.0)
        sat_boost = self.rng.uniform(25.0, 70.0)
        val_shift = self.rng.uniform(-20.0, 20.0)
        
        hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift * blob, 180.0)
        hsv[..., 1] = np.clip(hsv[..., 1] + sat_boost * blob, 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2] + val_shift * blob, 0.0, 255.0)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def local_discoloration(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "medical":
            return self._medical_lesion_blob(image, mode="mixed")

        out = image.astype(np.float32).copy()
        blob, _ = self._random_blob(out.astype(np.uint8), size_scale=(0.04, 0.10))
        region = blob > 0.2
        if not region.any():
            return out.astype(np.uint8)

        # Base color extraction
        base_color = out[region].mean(axis=0) # [R, G, B]
        base_hsv = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
        
        # Physics: Oxidation or chemical burn (slight hue shift, boost sat, lower value)
        hue_shift = self.rng.uniform(-15.0, 15.0) 
        sat_scale = self.rng.uniform(1.2, 2.0)
        val_scale = self.rng.uniform(0.4, 0.75) # Aggressively darken
        
        base_hsv[0] = np.mod(base_hsv[0] + hue_shift, 180.0)
        base_hsv[1] = np.clip(base_hsv[1] * sat_scale, 0.0, 255.0)
        base_hsv[2] = np.clip(base_hsv[2] * val_scale, 0.0, 255.0)
        
        target_color = cv2.cvtColor(np.uint8([[base_hsv]]), cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)

        alpha = self.rng.uniform(0.7, 0.95) * blob[..., None]
        mixed = out * (1.0 - alpha) + target_color[None, None, :] * alpha

        return np.clip(mixed, 0, 255).astype(np.uint8)

    def burn_patch(self, image: np.ndarray) -> np.ndarray:
        if self.domain == "medical":
            return self._medical_lesion_blob(image, mode="dark")

        out = image.astype(np.float32).copy()
        blob, _ = self._random_blob(out.astype(np.uint8), size_scale=(0.05, 0.12))
        darken = self.rng.uniform(45.0, 90.0)
        contrast = self.rng.uniform(0.75, 0.92)
        mean_rgb = out.mean(axis=(0, 1), keepdims=True)
        adjusted = (out - mean_rgb) * contrast + mean_rgb - darken * blob[..., None]
        mixed = out * (1.0 - blob[..., None]) + adjusted * blob[..., None]
        return np.clip(mixed, 0, 255).astype(np.uint8)

    def contamination_speckle(self, image: np.ndarray, count: int = 18) -> np.ndarray:
        if self.domain == "medical":
            return self._medical_speckle(image, count=max(10, count // 2))

        out = image.astype(np.float32).copy()
        mask = self._foreground_mask(image)
        overlay = out.copy()
        for _ in range(count):
            cx, cy = self._sample_point_in_mask(mask)
            radius = self.rng.randint(1, 3)
            
            # Ground the speckle purely on the local pixel color
            base_color = out[cy, cx].astype(np.float32)
            
            if self.rng.random() < 0.6:
                # Dirt/Oil (Dark variant of base color)
                target_color = np.clip(base_color * self.rng.uniform(0.2, 0.5), 0, 255)
            else:
                # Dust/Oxidation (Bright variant of base color)
                target_color = np.clip(base_color * self.rng.uniform(1.3, 1.8) + 30.0, 0, 255)
                
            cv2.circle(overlay, (cx, cy), radius, target_color.tolist(), -1, lineType=cv2.LINE_AA)
            
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        alpha = self.rng.uniform(0.4, 0.7)
        mixed = overlay * alpha + out * (1.0 - alpha)
        return np.clip(mixed, 0, 255).astype(np.uint8)

    # Medical helpers remain exactly the same as you already verified them!
    def _medical_lesion_blob(self, image: np.ndarray, mode: str) -> np.ndarray:
        out = image.astype(np.float32).copy()
        blob, _ = self._random_blob(out.astype(np.uint8), size_scale=(0.03, 0.08))
        local_mean = float(out.mean())
        if mode == "bright":
            target = np.clip(local_mean + self.rng.uniform(95.0, 155.0), 0.0, 255.0)
            alpha = self.rng.uniform(0.72, 0.95)
        elif mode == "dark":
            target = np.clip(local_mean - self.rng.uniform(40.0, 80.0), 0.0, 255.0)
            alpha = self.rng.uniform(0.45, 0.68)
        else:
            if self.rng.random() < 0.5:
                target = np.clip(local_mean + self.rng.uniform(85.0, 145.0), 0.0, 255.0)
                alpha = self.rng.uniform(0.65, 0.9)
            else:
                target = np.clip(local_mean - self.rng.uniform(35.0, 75.0), 0.0, 255.0)
                alpha = self.rng.uniform(0.42, 0.65)
        out = out * (1.0 - alpha * blob[..., None]) + target * (alpha * blob[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)

    def _medical_speckle(self, image: np.ndarray, count: int = 8) -> np.ndarray:
        out = image.astype(np.float32).copy()
        h, w = image.shape[:2]
        for _ in range(count):
            blob, _ = self._random_blob(out.astype(np.uint8), size_scale=(0.01, 0.03))
            polarity = self.rng.choice([-1.0, 1.0])
            magnitude = self.rng.uniform(22.0, 55.0)
            out += blob[..., None] * magnitude * polarity
        return np.clip(out, 0, 255).astype(np.uint8)

def parse_args():
    parser = argparse.ArgumentParser(description="Preview color perturbations on one RGB image.")
    parser.add_argument("--image-path", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default=DEFAULT_PLOT)
    parser.add_argument("--preview-suite", action="store_true")
    parser.add_argument("--domain", choices=["industrial", "medical"], default="industrial")
    parser.add_argument("--class-hint", type=str, default=None)
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
    for cls in MVTEC_CLASSES:
        image_path = _first_mvtec_train_image(cls)
        image = load_rgb_image(image_path, image_size=image_size)
        mvtec_perturber = ColorPerturber(seed=seed, domain="industrial", class_hint=cls)
        outputs = mvtec_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/mvtec") / cls / "color_damage_preview.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved mvtec color preview for {cls}: {output_path}")

    rsna_perturber = ColorPerturber(seed=seed, domain="medical")
    for idx, image_path in enumerate(_sample_rsna_paths(limit=6), start=1):
        image = load_rsna_preview_image(image_path, image_size=image_size)
        outputs = rsna_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/rsna") / f"color_sample_{idx}_{image_path.stem}.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved rsna color preview {idx}: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.preview_suite:
        build_preview_suite(image_size=args.image_size, seed=args.seed)
    else:
        image = load_rgb_image(args.image_path, image_size=args.image_size)
        perturber = ColorPerturber(seed=args.seed, domain=args.domain, class_hint=args.class_hint)
        outputs = perturber.apply_all(image)
        titles = list(outputs.keys())
        images = [outputs[name] for name in titles]
        save_preview_grid(images, titles, args.output_path)
        print(f"saved preview to {args.output_path}")
