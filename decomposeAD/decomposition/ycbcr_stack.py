from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import gaussian_blur


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return torch.cat([y, cb, cr], dim=1)


def sobel_edges(y: torch.Tensor, gamma: float = 0.5, eps: float = 1e-8) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=y.device,
        dtype=y.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(y, sobel_x, padding=1)
    grad_y = F.conv2d(y, sobel_y, padding=1)
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
    ycbcr = rgb_to_ycbcr(image.unsqueeze(0))
    y = ycbcr[:, 0:1]
    cb = ycbcr[:, 1:2]
    cr = ycbcr[:, 2:3]

    y_mid_base = gaussian_blur(
        y,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )
    y_coarse_fill = gaussian_blur(
        y,
        kernel_size=[coarse_blur_kernel, coarse_blur_kernel],
        sigma=[coarse_blur_sigma, coarse_blur_sigma],
    )
    y_mid_fill = y_mid_base - y_coarse_fill
    y_high_detail = y - y_mid_base
    edge_mask = sobel_edges(y, gamma=edge_gamma)
    y_edge_detail = edge_mask * y_high_detail
    y_texture_detail = (1.0 - edge_mask) * y_high_detail

    cb_fill = gaussian_blur(
        cb,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )
    cr_fill = gaussian_blur(
        cr,
        kernel_size=[mid_blur_kernel, mid_blur_kernel],
        sigma=[mid_blur_sigma, mid_blur_sigma],
    )

    return {
        "y_coarse_fill": y_coarse_fill.squeeze(0),
        "y_mid_fill": y_mid_fill.squeeze(0),
        "y_edge_detail": y_edge_detail.squeeze(0),
        "y_texture_detail": y_texture_detail.squeeze(0),
        "cb_fill": cb_fill.squeeze(0),
        "cb_detail": (cb - cb_fill).squeeze(0),
        "cr_fill": cr_fill.squeeze(0),
        "cr_detail": (cr - cr_fill).squeeze(0),
    }
