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


DEFAULT_IMAGE = "/data/imageaddatasets/mvtec_anomaly_detection/capsule/train/good/000.png"
DEFAULT_PLOT = "/data/stepdown vision/plots/perturbation/surface_damage_preview.png"
RSNA_ROOT = "/data/imageaddatasets/rsna-pneumonia-detection-challenge"


def load_rgb_image(path: str | Path, image_size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    return np.array(image, dtype=np.uint8)


def save_preview_grid(images: list[np.ndarray], titles: list[str], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, image, title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    for ax in axes.flatten()[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


class SurfaceDamagePerturber:
    def __init__(self, seed: int = 42, domain: str = "industrial"):
        self.rng = random.Random(seed)
        self.domain = domain

    def apply_all(self, image: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "original": image.copy(),
            "scratches": self.scratches(image),
            "cuts": self.cuts(image),
            "cracks": self.cracks(image),
            "holes": self.holes(image),
            "dents": self.dents(image),
            "thread_damage": self.thread_damage(image),
        }

    def _foreground_mask(self, image: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Blur slightly to remove noise before thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Otsu's thresholding dynamically finds the best split between object and background
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # If the corners (usually background) are white, invert the mask
            if thresh[0, 0] == 255 and thresh[-1, -1] == 255:
                thresh = cv2.bitwise_not(thresh)
                
            mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            return mask.astype(bool)

    def _sample_point_in_mask(self, mask: np.ndarray) -> tuple[int, int]:
        ys, xs = np.where(mask)
        idx = self.rng.randrange(len(xs))
        return int(xs[idx]), int(ys[idx])

    def _clip_line_to_mask(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        mask: np.ndarray,
        steps: int = 64,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        points = []
        for t in np.linspace(0.0, 1.0, steps):
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x]:
                points.append((x, y))
        if len(points) < 2:
            return None
        return points[0], points[-1]

    def _soft_blend(self, base: np.ndarray, overlay: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        return cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)

    def scratches(self, image: np.ndarray, count: int = 10) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out)
        overlay = out.copy()
        for _ in range(count):
            x1, y1 = self._sample_point_in_mask(mask)
            x2 = int(np.clip(x1 + self.rng.randint(-w // 4, w // 4), 0, w - 1))
            y2 = int(np.clip(y1 + self.rng.randint(-h // 4, h // 4), 0, h - 1))
            clipped = self._clip_line_to_mask(x1, y1, x2, y2, mask)
            if clipped is None:
                continue
            (x1, y1), (x2, y2) = clipped
            color = self.rng.choice([(205, 205, 205), (40, 40, 40)])
            thickness = self.rng.randint(1, 2)
            cv2.line(overlay, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        return self._soft_blend(out, overlay, alpha=0.45)

    def cuts(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out)
        overlay = out.copy()
        for _ in range(count):
            x1, y1 = self._sample_point_in_mask(mask)
            x2 = int(np.clip(x1 + self.rng.randint(-w // 3, w // 3), 0, w - 1))
            y2 = int(np.clip(y1 + self.rng.randint(-h // 3, h // 3), 0, h - 1))
            clipped = self._clip_line_to_mask(x1, y1, x2, y2, mask)
            if clipped is None:
                continue
            (x1, y1), (x2, y2) = clipped
            thickness = self.rng.randint(1, 2)
            cv2.line(overlay, (x1, y1), (x2, y2), (35, 35, 35), thickness, lineType=cv2.LINE_AA)
            cv2.line(overlay, (x1, y1), (x2, y2), (120, 120, 120), 1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        return self._soft_blend(out, overlay, alpha=0.35)

    def cracks(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out)
        overlay = out.copy()
        for _ in range(count):
            points = []
            x, y = self._sample_point_in_mask(mask)
            points.append((x, y))
            for _ in range(self.rng.randint(5, 8)):
                x = int(np.clip(x + self.rng.randint(-w // 10, w // 10), 0, w - 1))
                y = int(np.clip(y + self.rng.randint(-h // 10, h // 10), 0, h - 1))
                if mask[y, x]:
                    points.append((x, y))
            if len(points) < 2:
                continue
            for idx in range(len(points) - 1):
                cv2.line(overlay, points[idx], points[idx + 1], (45, 45, 45), 1, lineType=cv2.LINE_AA)
            if len(points) > 2:
                branch_root = points[self.rng.randint(1, len(points) - 2)]
                bx = int(np.clip(branch_root[0] + self.rng.randint(-w // 12, w // 12), 0, w - 1))
                by = int(np.clip(branch_root[1] + self.rng.randint(-h // 12, h // 12), 0, h - 1))
                clipped = self._clip_line_to_mask(branch_root[0], branch_root[1], bx, by, mask, steps=24)
                if clipped is not None:
                    _, branch_end = clipped
                    cv2.line(overlay, branch_root, branch_end, (55, 55, 55), 1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        return self._soft_blend(out, overlay, alpha=0.45)

    def holes(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        if self.domain == "medical":
            return self.medical_holes(image, count=count)
        return self.industrial_holes(image, count=count)

    def industrial_holes(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out)
        for _ in range(count):
            center = self._sample_point_in_mask(mask)
            radius = self.rng.randint(max(4, min(h, w) // 36), max(7, min(h, w) // 24))
            num_vertices = self.rng.randint(10, 16)
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            points = []
            for angle in angles:
                jitter = radius * self.rng.uniform(0.75, 1.2)
                px = int(round(center[0] + np.cos(angle) * jitter))
                py = int(round(center[1] + np.sin(angle) * jitter))
                px = int(np.clip(px, 0, w - 1))
                py = int(np.clip(py, 0, h - 1))
                points.append([px, py])
            polygon = np.array(points, dtype=np.int32)
            fill_value = self.rng.randint(8, 35)
            edge_value = int(np.clip(fill_value + self.rng.randint(-35, 35), 0, 255))
            cv2.fillPoly(out, [polygon], (fill_value, fill_value, fill_value))
            cv2.polylines(
                out,
                [polygon],
                isClosed=True,
                color=(edge_value, edge_value, edge_value),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [polygon], 1)
            outside = poly_mask.astype(bool) & ~mask
            out[outside] = image[outside]
        return out

    def medical_holes(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        out = image.astype(np.float32).copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(image)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(count):
            cx, cy = self._sample_point_in_mask(mask)
            radius = self.rng.randint(max(6, min(h, w) // 30), max(10, min(h, w) // 18))
            num_vertices = self.rng.randint(10, 16)
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
            polygon_points = []
            for angle in angles:
                jitter = radius * self.rng.uniform(0.75, 1.2)
                px = int(round(cx + np.cos(angle) * jitter))
                py = int(round(cy + np.sin(angle) * jitter))
                px = int(np.clip(px, 0, w - 1))
                py = int(np.clip(py, 0, h - 1))
                polygon_points.append([px, py])

            poly_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [np.array(polygon_points, dtype=np.int32)], 1)
            poly_mask = poly_mask.astype(bool) & mask
            if not poly_mask.any():
                continue

            sigma = max(2.0, radius / 2.5)
            blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / (2 * sigma * sigma))
            blob *= poly_mask.astype(np.float32)
            blob = cv2.GaussianBlur(blob, (0, 0), sigmaX=max(1.0, radius / 4.0), sigmaY=max(1.0, radius / 4.0))
            blob *= poly_mask.astype(np.float32)

            local_values = out[poly_mask]
            local_mean = float(local_values.mean())
            polarity = self.rng.choice(["dark", "bright"])
            if polarity == "dark":
                delta = self.rng.uniform(25.0, 60.0)
            else:
                delta = self.rng.uniform(70.0, 125.0)
            target_value = local_mean - delta if polarity == "dark" else local_mean + delta
            target_value = float(np.clip(target_value, 0.0, 255.0))

            alpha = self.rng.uniform(0.35, 0.55) if polarity == "dark" else self.rng.uniform(0.55, 0.85)
            for channel in range(3):
                out[..., channel] = out[..., channel] * (1.0 - alpha * blob) + target_value * (alpha * blob)

        return np.clip(out, 0, 255).astype(np.uint8)

    def dents(self, image: np.ndarray, count: int = 3) -> np.ndarray:
        out = image.astype(np.float32).copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(image)
        yy, xx = np.mgrid[0:h, 0:w]
        for _ in range(count):
            cx, cy = self._sample_point_in_mask(mask)
            rx = self.rng.randint(max(8, w // 20), max(14, w // 10))
            ry = self.rng.randint(max(8, h // 20), max(14, h // 10))
            dent = np.exp(-(((xx - cx) ** 2) / (2 * rx * rx) + ((yy - cy) ** 2) / (2 * ry * ry)))
            highlight = np.roll(dent, shift=(-2, -2), axis=(0, 1))
            dent *= mask
            highlight *= mask
            out -= dent[..., None] * 55.0
            out += highlight[..., None] * 18.0
        return np.clip(out, 0, 255).astype(np.uint8)

    def broken_edges(self, image: np.ndarray, count: int = 2) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return out
        contour = max(contours, key=cv2.contourArea)
        for _ in range(count):
            depth = self.rng.randint(max(4, min(h, w) // 36), max(8, min(h, w) // 20))
            span = self.rng.randint(max(4, len(contour) // 80), max(10, len(contour) // 35))
            idx = self.rng.randint(0, len(contour) - 1)
            center = contour[idx][0]
            prev_pt = contour[(idx - span) % len(contour)][0]
            next_pt = contour[(idx + span) % len(contour)][0]
            center_vec = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            p_center = np.array(center, dtype=np.float32)
            direction = center_vec - p_center
            inward = direction / (np.linalg.norm(direction) + 1e-6) * depth
            poly = np.array(
                [
                    prev_pt,
                    next_pt,
                    (next_pt + inward).astype(np.int32),
                    (prev_pt + inward).astype(np.int32),
                ],
                dtype=np.int32,
            )
            fill = tuple(int(x) for x in out.reshape(-1, 3).mean(axis=0))
            cv2.fillPoly(out, [poly], fill)
        return out

    def thread_damage(self, image: np.ndarray, count: int = 5) -> np.ndarray:
        out = image.copy()
        h, w = out.shape[:2]
        mask = self._foreground_mask(out)
        overlay = out.copy()
        for _ in range(count):
            points = []
            x, y = self._sample_point_in_mask(mask)
            points.append((x, y))
            for _ in range(6):
                x = int(np.clip(x + self.rng.randint(-w // 12, w // 12), 0, w - 1))
                y = int(np.clip(y + self.rng.randint(-h // 12, h // 12), 0, h - 1))
                if mask[y, x]:
                    points.append((x, y))
            if len(points) < 2:
                continue
            for idx in range(len(points) - 1):
                cv2.line(overlay, points[idx], points[idx + 1], (220, 220, 220), 1, lineType=cv2.LINE_AA)
            if len(points) > 2:
                fringe_root = points[self.rng.randint(1, len(points) - 2)]
                fx = int(np.clip(fringe_root[0] + self.rng.randint(-w // 20, w // 20), 0, w - 1))
                fy = int(np.clip(fringe_root[1] + self.rng.randint(-h // 20, h // 20), 0, h - 1))
                clipped = self._clip_line_to_mask(fringe_root[0], fringe_root[1], fx, fy, mask, steps=24)
                if clipped is not None:
                    _, fringe_end = clipped
                    cv2.line(overlay, fringe_root, fringe_end, (210, 210, 210), 1, lineType=cv2.LINE_AA)
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        return self._soft_blend(out, overlay, alpha=0.4)


def parse_args():
    parser = argparse.ArgumentParser(description="Preview surface-damage perturbations on one RGB image.")
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
    mvtec_perturber = SurfaceDamagePerturber(seed=seed, domain="industrial")

    for cls in MVTEC_CLASSES:
        image_path = _first_mvtec_train_image(cls)
        image = load_rgb_image(image_path, image_size=image_size)
        outputs = mvtec_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/mvtec") / cls / "surface_damage_preview.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved mvtec preview for {cls}: {output_path}")

    rsna_perturber = SurfaceDamagePerturber(seed=seed, domain="medical")
    rsna_paths = _sample_rsna_paths(limit=6)
    for idx, image_path in enumerate(rsna_paths, start=1):
        image = load_rsna_preview_image(image_path, image_size=image_size)
        outputs = rsna_perturber.apply_all(image)
        output_path = Path("/data/stepdown vision/plots/perturbation/rsna") / f"sample_{idx}_{image_path.stem}.png"
        save_preview_grid([outputs[name] for name in outputs], list(outputs.keys()), output_path)
        print(f"saved rsna preview {idx}: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.preview_suite:
        build_preview_suite(image_size=args.image_size, seed=args.seed)
    else:
        image = load_rgb_image(args.image_path, image_size=args.image_size)
        perturber = SurfaceDamagePerturber(seed=args.seed, domain=args.domain)
        outputs = perturber.apply_all(image)
        titles = list(outputs.keys())
        images = [outputs[name] for name in titles]
        save_preview_grid(images, titles, args.output_path)
        print(f"saved preview to {args.output_path}")
