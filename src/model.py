"""ImageNet-ResNet SpatialAD with normal-reference sensor statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)

from .utils import spatial_embedding_grid


@dataclass
class SpatialADOutput:
    """Outputs aligned with the input ``[B, sensors, timestamps]`` layout."""

    logits: Tensor
    embedding: Tensor
    contrastive_embedding: Tensor


class SpatialAD(nn.Module):
    """ResNet spatial-feature encoder with supervised embedding heads.

    Each independent image is converted to a normalized three-channel input
    for an ImageNet ResNet. Its final convolutional feature map provides
    spatial region tokens. A downsampled normal-reference z-score map is added
    as an auxiliary token channel before spatial Transformer attention.
    """

    def __init__(
        self,
        image_size: int = 255,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
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
        if backbone not in {"resnet18", "resnet50"}:
            raise ValueError("backbone must be 'resnet18' or 'resnet50'")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if projection_dim < 1:
            raise ValueError("projection_dim must be positive")
        if z_clip <= 0:
            raise ValueError("z_clip must be positive")

        self.image_size = image_size
        self.sensor_count = image_size * image_size
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.z_clip = z_clip
        self.resnet_input_size = 224

        mean = torch.zeros(self.sensor_count) if reference_mean is None else reference_mean
        std = torch.ones(self.sensor_count) if reference_std is None else reference_std
        if mean.numel() != self.sensor_count or std.numel() != self.sensor_count:
            raise ValueError("reference statistics must contain image_size**2 values")
        self.register_buffer("reference_mean", mean.detach().float().reshape(-1))
        self.register_buffer("reference_std", std.detach().float().reshape(-1).clamp_min(1e-3))
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            network = resnet18(weights=weights)
            backbone_channels = 512
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            network = resnet50(weights=weights)
            backbone_channels = 2048
        self.backbone = nn.Sequential(*list(network.children())[:-2])
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # ResNet layer4 produces a 7x7 map for 224x224 input.
        self.feature_grid = 7
        self.feature_projection = nn.Linear(backbone_channels + 1, d_model)
        positional = spatial_embedding_grid(
            self.feature_grid,
            self.feature_grid,
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

    def _resnet_features(self, images: Tensor) -> Tensor:
        images = F.interpolate(
            images,
            size=(self.resnet_input_size, self.resnet_input_size),
            mode="bilinear",
            align_corners=False,
        )
        images = images.repeat(1, 3, 1, 1)
        images = (images - self.imagenet_mean) / self.imagenet_std
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                return self.backbone(images)
        return self.backbone(images)

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

        feature_map = self._resnet_features(raw_images)
        z_map = F.interpolate(
            z_images,
            size=feature_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        tokens = torch.cat((feature_map, z_map), dim=1).flatten(2).transpose(1, 2)
        tokens = self.feature_projection(tokens)
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
