"""SpatialAD models and controlled backbone ablations."""

from __future__ import annotations

from typing import NamedTuple

import timm
import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .utils import spatial_embedding_grid


class SpatialADOutput(NamedTuple):
    logits: Tensor
    embedding: Tensor


class SpatialAD(nn.Module):
    """Backbone classifier with optional reference z-map and spatial attention.

    ``spatial_mode='pool'`` removes the SpatialAD Transformer.  For DINOv2
    with ``use_reference_z=False`` this is the raw DINOv2 + classifier
    ablation.  Enabling ``use_reference_z`` gives the matched DINOv2 + z-map
    pooled control.
    """

    def __init__(
        self,
        image_size: int = 255,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        spatial_mode: str = "transformer",
        use_reference_z: bool = True,
        z_clip: float = 8.0,
        reference_mean: Tensor | None = None,
        reference_std: Tensor | None = None,
    ) -> None:
        super().__init__()
        if backbone not in {"resnet18", "resnet50", "dino_small"}:
            raise ValueError(f"Unknown backbone: {backbone}")
        if spatial_mode not in {"transformer", "pool"}:
            raise ValueError("spatial_mode must be 'transformer' or 'pool'")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.image_size = image_size
        self.backbone_name = backbone
        self.spatial_mode = spatial_mode
        self.use_reference_z = use_reference_z
        self.freeze_backbone = freeze_backbone
        self.z_clip = float(z_clip)
        self.input_size = 224

        if backbone == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            net = resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self.backbone_channels = 512
            self.feature_grid = (7, 7)
        elif backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self.backbone_channels = 2048
            self.feature_grid = (7, 7)
        else:
            self.backbone = timm.create_model(
                "vit_small_patch14_dinov2",
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
                img_size=self.input_size,
            )
            self.backbone_channels = 384
            self.feature_grid = (16, 16)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()

        self.register_buffer("reference_mean", torch.zeros(image_size * image_size), persistent=False)
        self.register_buffer("reference_std", torch.ones(image_size * image_size), persistent=False)
        if reference_mean is not None or reference_std is not None:
            if reference_mean is None or reference_std is None:
                raise ValueError("reference_mean and reference_std must be provided together")
            self.set_reference_stats(reference_mean, reference_std)

        input_channels = self.backbone_channels + (1 if use_reference_z else 0)
        self.feature_projection = nn.Linear(input_channels, d_model)
        self.global_projection = nn.Linear(self.backbone_channels, d_model)

        if spatial_mode == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.spatial_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.spatial_encoder = None
        self.embedding_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def set_reference_stats(self, mean: Tensor, std: Tensor) -> None:
        mean = torch.as_tensor(mean, dtype=self.reference_mean.dtype, device=self.reference_mean.device)
        std = torch.as_tensor(std, dtype=self.reference_std.dtype, device=self.reference_std.device)
        expected = self.image_size * self.image_size
        if mean.numel() != expected or std.numel() != expected:
            raise ValueError(f"Expected {expected} reference pixels, got {mean.numel()} and {std.numel()}")
        self.reference_mean.copy_(mean.reshape(-1))
        self.reference_std.copy_(std.reshape(-1).clamp_min(1e-6))

    def _reference_z(self, images: Tensor) -> Tensor:
        mean = self.reference_mean.reshape(1, self.image_size, self.image_size)
        std = self.reference_std.reshape(1, self.image_size, self.image_size)
        return ((images - mean) / std.clamp_min(1e-6)).clamp(-self.z_clip, self.z_clip)

    def _backbone_forward(self, rgb: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.backbone_name != "dino_small":
            feature_map = self.backbone(rgb)
            return feature_map, feature_map.mean(dim=(2, 3))

        tokens = self.backbone.forward_features(rgb)
        if isinstance(tokens, dict):
            if "x_norm_clstoken" in tokens and "x_norm_patchtokens" in tokens:
                cls_features = tokens["x_norm_clstoken"]
                patches = tokens["x_norm_patchtokens"]
            else:
                tokens = tokens.get("x_prenorm", tokens.get("x_norm"))
                if tokens is None:
                    raise RuntimeError("Unsupported DINOv2 forward_features output")
                cls_features, patches = tokens[:, 0], tokens[:, 1:]
        else:
            cls_features, patches = tokens[:, 0], tokens[:, 1:]
        height, width = self.feature_grid
        feature_map = patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[2], height, width)
        return feature_map, cls_features

    def _backbone_features(self, images: Tensor) -> tuple[Tensor, Tensor | None]:
        rgb = images.unsqueeze(1).repeat(1, 3, 1, 1)
        rgb = torch.nn.functional.interpolate(rgb, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = rgb.new_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = rgb.new_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        rgb = (rgb - mean) / std
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                return self._backbone_forward(rgb)
        return self._backbone_forward(rgb)

    def _positional_embedding(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        try:
            position = spatial_embedding_grid(height, width, self.feature_projection.out_features)
        except TypeError:
            position = spatial_embedding_grid(height, width, self.feature_projection.out_features, device=device, dtype=dtype)
        position = torch.as_tensor(position, device=device, dtype=dtype)
        if position.ndim == 2:
            position = position.reshape(height, width, -1)
        return position.reshape(1, height * width, -1)

    def forward(self, x: Tensor) -> SpatialADOutput:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, sensors, timestamps], got {tuple(x.shape)}")
        batch, sensors, timestamps = x.shape
        expected = self.image_size * self.image_size
        if sensors != expected:
            raise ValueError(f"Expected {expected} sensors for {self.image_size}x{self.image_size}, got {sensors}")

        images = x.permute(0, 2, 1).reshape(batch * timestamps, self.image_size, self.image_size)
        z_map = self._reference_z(images) if self.use_reference_z else None
        feature_map, cls_features = self._backbone_features(images)

        if self.spatial_mode == "pool" and not self.use_reference_z and self.backbone_name == "dino_small":
            embedding = self.global_projection(cls_features)
        else:
            if z_map is not None:
                z_feature = torch.nn.functional.interpolate(z_map.unsqueeze(1), size=feature_map.shape[-2:], mode="bilinear", align_corners=False)
                features = torch.cat([feature_map, z_feature], dim=1)
            else:
                features = feature_map
            tokens = self.feature_projection(features.flatten(2).transpose(1, 2))
            if self.spatial_mode == "transformer":
                height, width = feature_map.shape[-2:]
                tokens = tokens + self._positional_embedding(height, width, tokens.device, tokens.dtype)
                tokens = self.spatial_encoder(tokens)
            embedding = tokens.mean(dim=1)

        embedding = self.embedding_norm(embedding)
        logits = self.classifier(embedding).squeeze(-1)
        return SpatialADOutput(logits=logits.reshape(batch, timestamps), embedding=embedding.reshape(batch, timestamps, -1))
