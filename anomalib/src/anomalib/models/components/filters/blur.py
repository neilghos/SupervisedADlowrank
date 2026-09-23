"""Gaussian blurring via pure PyTorch (Zero external dependencies)."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    """Compute 1D Gaussian kernel."""
    k = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    kernel = torch.exp(-0.5 * (k / sigma) ** 2)
    return kernel / kernel.sum()


def get_gaussian_kernel2d(kernel_size: tuple[int, int], sigma: tuple[float, float]) -> Tensor:
    """Compute 2D Gaussian kernel."""
    kx = get_gaussian_kernel1d(kernel_size[0], sigma[0])
    ky = get_gaussian_kernel1d(kernel_size[1], sigma[1])
    return torch.outer(kx, ky)


def normalize_kernel2d(kernel: Tensor) -> Tensor:
    """Normalize 2D kernel."""
    return kernel / kernel.sum()


def _compute_padding(kernel_size: list[int]) -> list[int]:
    """Compute reflection padding."""
    return [kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2]


def compute_kernel_size(sigma_val: float) -> int:
    """Compute kernel size from sigma value."""
    return 2 * int(4.0 * sigma_val + 0.5) + 1


class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d."""

    def __init__(
        self,
        sigma: float | tuple[float, float],
        channels: int = 1,
        kernel_size: int | tuple[int, int] | None = None,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        super().__init__()
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.channels = channels

        if kernel_size is None:
            kernel_size = (compute_kernel_size(sigma[0]), compute_kernel_size(sigma[1]))
        else:
            kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        kernel = get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma)
        if normalize:
            kernel = normalize_kernel2d(kernel)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(self.channels, -1, -1, -1)
        self.register_buffer("kernel", kernel)

        self.border_type = border_type
        self.padding = padding
        self.height, self.width = kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Blur the input with the computed Gaussian."""
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out
