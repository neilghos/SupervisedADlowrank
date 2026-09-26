from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
D65 = torch.tensor([0.95047, 1.0, 1.08883]).view(1, 3, 1, 1)


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)


def rgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    rgb_linear = srgb_to_linear(rgb)
    r = rgb_linear[:, 0:1]
    g = rgb_linear[:, 1:2]
    b = rgb_linear[:, 2:3]
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return torch.cat([x, y, z], dim=1)


def lab_f(t: torch.Tensor) -> torch.Tensor:
    delta = 6.0 / 29.0
    return torch.where(t > delta**3, t.clamp_min(0.0).pow(1.0 / 3.0), t / (3 * delta**2) + 4.0 / 29.0)


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    white = D65.to(device=rgb.device, dtype=rgb.dtype)
    xyz = rgb_to_xyz(rgb) / white
    fx = lab_f(xyz[:, 0:1])
    fy = lab_f(xyz[:, 1:2])
    fz = lab_f(xyz[:, 2:3])
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([l, a, b], dim=1)


def sobel_edges(channel: torch.Tensor, gamma: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=channel.device,
        dtype=channel.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=channel.device,
        dtype=channel.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(channel, sobel_x, padding=1)
    grad_y = F.conv2d(channel, sobel_y, padding=1)
    edge = torch.sqrt(grad_x.square() + grad_y.square() + eps)
    edge_max = edge.flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(eps)
    return (edge / edge_max).pow(gamma)


def load_image(path: Path, resize_size: int, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, resize_size, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [image_size, image_size])
    return TF.to_tensor(image)


def iter_samples(data_root: Path, split: str) -> list[tuple[Path, str, int]]:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    samples = []
    for label, class_name in enumerate(classes):
        class_root = split_root / class_name
        for path in sorted(class_root.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_name, label))

    if not samples:
        raise FileNotFoundError(f"No images found under {split_root}")
    return samples


def decompose(
    image: torch.Tensor,
    mid_blur_kernel: int,
    mid_blur_sigma: float,
    coarse_blur_kernel: int,
    coarse_blur_sigma: float,
    edge_gamma: float,
) -> dict[str, torch.Tensor]:
    lab = rgb_to_lab(image.unsqueeze(0))
    l = lab[:, 0:1]
    a = lab[:, 1:2]
    b = lab[:, 2:3]

    l_norm = l / 100.0
    l_mid_base = gaussian_blur(
        l_norm,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )
    l_coarse_fill = gaussian_blur(
        l_norm,
        kernel_size=[coarse_blur_kernel, coarse_blur_kernel],
        sigma=[coarse_blur_sigma, coarse_blur_sigma],
    )
    l_mid_fill = l_mid_base - l_coarse_fill
    l_high_detail = l_norm - l_mid_base
    edge_mask = sobel_edges(l_norm, gamma=edge_gamma)
    l_edge_detail = edge_mask * l_high_detail
    l_texture_detail = (1.0 - edge_mask) * l_high_detail

    a_fill = gaussian_blur(
        a,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )
    b_fill = gaussian_blur(
        b,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )

    return {
        "l_coarse_fill": l_coarse_fill.squeeze(0),
        "l_mid_fill": l_mid_fill.squeeze(0),
        "l_edge_detail": l_edge_detail.squeeze(0),
        "l_texture_detail": l_texture_detail.squeeze(0),
        "a_fill": a_fill.squeeze(0),
        "a_detail": (a - a_fill).squeeze(0),
        "b_fill": b_fill.squeeze(0),
        "b_detail": (b - b_fill).squeeze(0),
    }
