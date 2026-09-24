"""Geometry-only utilities for spatial pixel modelling."""

from __future__ import annotations

import math

import torch


def spatial_embedding(
    row: torch.Tensor | int | float,
    col: torch.Tensor | int | float,
    embed_dim: int = 128,
    height: int = 255,
    width: int = 255,
    max_period: float = 10_000.0,
) -> torch.Tensor:
    """Return a deterministic 2D positional embedding for pixel coordinates.

    The embedding depends only on ``(row, col)`` and never on image values or
    labels. The final dimension is ordered as row sine/cosine bands followed by
    column sine/cosine bands. Scalar or broadcastable coordinate tensors are
    accepted; coordinates are returned with one final embedding dimension.
    """

    if embed_dim < 4:
        raise ValueError("embed_dim must be at least 4")
    if height < 2 or width < 2:
        raise ValueError("height and width must be at least 2")
    if max_period <= 1:
        raise ValueError("max_period must be greater than 1")

    row_tensor = torch.as_tensor(row, dtype=torch.float32)
    col_tensor = torch.as_tensor(col, dtype=torch.float32, device=row_tensor.device)
    row_tensor, col_tensor = torch.broadcast_tensors(row_tensor, col_tensor)

    row_normalized = row_tensor / (height - 1)
    col_normalized = col_tensor / (width - 1)

    bands = max(1, math.ceil(embed_dim / 4))
    frequency_index = torch.arange(
        bands,
        dtype=row_tensor.dtype,
        device=row_tensor.device,
    )
    frequencies = torch.exp(-math.log(max_period) * frequency_index / max(bands - 1, 1))
    angular_row = 2.0 * math.pi * row_normalized[..., None] * frequencies
    angular_col = 2.0 * math.pi * col_normalized[..., None] * frequencies

    embedding = torch.cat(
        [
            torch.sin(angular_row),
            torch.cos(angular_row),
            torch.sin(angular_col),
            torch.cos(angular_col),
        ],
        dim=-1,
    )
    return embedding[..., :embed_dim]


def spatial_embedding_grid(
    height: int = 255,
    width: int = 255,
    embed_dim: int = 128,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return embeddings for every pixel as ``[height * width, embed_dim]``."""

    rows, cols = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    return spatial_embedding(
        rows.reshape(-1),
        cols.reshape(-1),
        embed_dim=embed_dim,
        height=height,
        width=width,
    )
