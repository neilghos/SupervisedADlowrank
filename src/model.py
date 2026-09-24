"""Patch-based spatial anomaly detector for VAD images."""

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
    anomaly_map: Tensor
    reconstruction: Tensor
    reconstruction_score: Tensor


class SpatialAD(nn.Module):
    """Spatial patch Transformer with per-image supervised anomaly scores.

    Input shape:
        ``[B, sensors, timestamps]`` where ``sensors = image_size**2``.

    Output shapes:
        logits: ``[B, timestamps]``
        anomaly_map: ``[B, timestamps, image_size, image_size]``
    """

    def __init__(
        self,
        image_size: int = 255,
        patch_size: int = 15,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size
        self.patch_dim = patch_size * patch_size

        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
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
        self.patch_decoder = nn.Linear(d_model, self.patch_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def _patchify(self, images: Tensor) -> Tensor:
        """Convert ``[BT, 1, H, W]`` into ``[BT, patches, patch_pixels]``."""

        patches = F.unfold(
            images,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        return patches.transpose(1, 2)

    def _unpatchify(self, patches: Tensor) -> Tensor:
        """Convert ``[BT, patches, patch_pixels]`` back to ``[BT, 1, H, W]``."""

        return F.fold(
            patches.transpose(1, 2),
            output_size=(self.image_size, self.image_size),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, features: Tensor) -> SpatialADOutput:
        if features.ndim != 3:
            raise ValueError(f"Expected [B, sensors, timestamps], got {tuple(features.shape)}")

        batch_size, sensor_count, timestamps = features.shape
        expected_sensors = self.image_size * self.image_size
        if sensor_count != expected_sensors:
            raise ValueError(
                f"Expected {expected_sensors} sensors for {self.image_size}x{self.image_size}, "
                f"got {sensor_count}"
            )

        # The timestamp axis contains independent images, not a temporal signal.
        images = features.permute(0, 2, 1).reshape(
            batch_size * timestamps,
            1,
            self.image_size,
            self.image_size,
        )
        patches = self._patchify(images)
        tokens = self.patch_embedding(patches)
        tokens = tokens + self.positional_embedding.unsqueeze(0)
        encoded = self.output_norm(self.spatial_encoder(tokens))

        reconstructed_patches = self.patch_decoder(encoded)
        reconstruction = self._unpatchify(reconstructed_patches)
        anomaly_map = (reconstruction - images).abs()
        reconstruction_score = anomaly_map.flatten(1).mean(dim=1)

        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)

        return SpatialADOutput(
            logits=logits.reshape(batch_size, timestamps),
            anomaly_map=anomaly_map.reshape(
                batch_size,
                timestamps,
                self.image_size,
                self.image_size,
            ),
            reconstruction=reconstruction.reshape(
                batch_size,
                timestamps,
                self.image_size,
                self.image_size,
            ),
            reconstruction_score=reconstruction_score.reshape(batch_size, timestamps),
        )

    def loss(
        self,
        output: SpatialADOutput,
        features: Tensor,
        labels: Tensor,
        reconstruction_weight: float = 1.0,
        classification_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute reconstruction plus per-image supervised classification loss."""

        target_images = features.permute(0, 2, 1).reshape_as(output.reconstruction)
        reconstruction_loss = F.mse_loss(output.reconstruction, target_images)
        classification_loss = F.binary_cross_entropy_with_logits(
            output.logits,
            labels.float(),
        )
        total = reconstruction_weight * reconstruction_loss + classification_weight * classification_loss
        return {
            "loss": total,
            "reconstruction_loss": reconstruction_loss,
            "classification_loss": classification_loss,
        }
