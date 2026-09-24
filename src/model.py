"""Patch-based SpatialAD with normal-reference channels and embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .utils import spatial_embedding_grid


@dataclass
class SpatialADOutput:
    """Outputs aligned with the input ``[B, sensors, timestamps]`` layout."""

    logits: Tensor
    embedding: Tensor
    contrastive_embedding: Tensor


class SpatialAD(nn.Module):
    """Spatial patch Transformer with a supervised embedding head.

    Each pixel receives raw intensity and a frozen normal-reference z-score.
    Spatial patch tokens are encoded jointly within each image.  The pooled
    image representation feeds both the classifier and a normalized
    projection head for supervised contrastive training.
    """

    def __init__(
        self,
        image_size: int = 255,
        patch_size: int = 15,
        d_model: int = 128,
        projection_dim: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        reference_mean: Tensor | None = None,
        reference_std: Tensor | None = None,
        z_clip: float = 8.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if z_clip <= 0:
            raise ValueError("z_clip must be positive")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size
        self.sensor_count = image_size * image_size
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.z_clip = z_clip

        mean = torch.zeros(self.sensor_count) if reference_mean is None else reference_mean
        std = torch.ones(self.sensor_count) if reference_std is None else reference_std
        if mean.numel() != self.sensor_count or std.numel() != self.sensor_count:
            raise ValueError("reference statistics must contain image_size**2 values")
        self.register_buffer("reference_mean", mean.detach().float().reshape(-1))
        self.register_buffer("reference_std", std.detach().float().reshape(-1).clamp_min(1e-3))

        self.patch_embedding = nn.Linear(self.patch_dim * 2, d_model)
        positional = spatial_embedding_grid(
            self.grid_size,
            self.grid_size,
            embed_dim=d_model,
        )
        self.register_buffer("positional_embedding", positional, persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )

    def _patchify(self, images: Tensor) -> Tensor:
        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches.transpose(1, 2)

    def forward(self, features: Tensor) -> SpatialADOutput:
        if features.ndim != 3:
            raise ValueError(f"Expected [B, sensors, timestamps], got {tuple(features.shape)}")

        batch_size, sensor_count, timestamps = features.shape
        if sensor_count != self.sensor_count:
            raise ValueError(
                f"Expected {self.sensor_count} sensors for {self.image_size}x{self.image_size}, "
                f"got {sensor_count}"
            )

        mean = self.reference_mean.to(dtype=features.dtype).view(1, sensor_count, 1)
        std = self.reference_std.to(dtype=features.dtype).view(1, sensor_count, 1)
        sensor_z = ((features - mean) / std).clamp(-self.z_clip, self.z_clip)

        raw_images = features.permute(0, 2, 1).reshape(
            batch_size * timestamps, 1, self.image_size, self.image_size
        )
        z_images = sensor_z.permute(0, 2, 1).reshape(
            batch_size * timestamps, 1, self.image_size, self.image_size
        )
        model_images = torch.cat((raw_images, z_images), dim=1)

        tokens = self.patch_embedding(self._patchify(model_images))
        tokens = tokens + self.positional_embedding.unsqueeze(0)
        encoded = self.output_norm(self.spatial_encoder(tokens))
        embedding = encoded.mean(dim=1)
        logits = self.classifier(embedding).squeeze(-1)
        projection = F.normalize(self.projection_head(embedding), dim=-1)

        return SpatialADOutput(
            logits=logits.reshape(batch_size, timestamps),
            embedding=embedding.reshape(batch_size, timestamps, self.d_model),
            contrastive_embedding=projection.reshape(
                batch_size, timestamps, self.projection_dim
            ),
        )
