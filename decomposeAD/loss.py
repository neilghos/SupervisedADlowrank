from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def stage1loss(
    predicted_components: Dict[str, torch.Tensor],
    target_components: Dict[str, torch.Tensor],
    predicted_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    decomposition_weight: float = 1.0,
    fusion_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    component_losses = []
    for name in sorted(target_components.keys()):
        component_losses.append(F.l1_loss(predicted_components[name], target_components[name]))

    decomposition_loss = torch.stack(component_losses).mean()
    fusion_loss = F.l1_loss(predicted_rgb, target_rgb)
    total_loss = decomposition_weight * decomposition_loss + fusion_weight * fusion_loss

    return {
        "loss": total_loss,
        "decomposition_loss": decomposition_loss,
        "fusion_loss": fusion_loss,
    }


def stage2loss(*args, **kwargs) -> Dict[str, torch.Tensor]:
    predicted_components: Dict[str, torch.Tensor] = kwargs["predicted_components"]
    target_components: Dict[str, torch.Tensor] = kwargs["target_components"]
    predicted_rgb: torch.Tensor = kwargs["predicted_rgb"]
    target_rgb: torch.Tensor = kwargs["target_rgb"]
    decomposition_weight: float = kwargs.get("decomposition_weight", 1.0)
    fusion_weight: float = kwargs.get("fusion_weight", 1.0)

    component_losses = []
    for name in sorted(target_components.keys()):
        component_losses.append(F.mse_loss(predicted_components[name], target_components[name]))

    decomposition_loss = torch.stack(component_losses).mean()
    fusion_loss = F.mse_loss(predicted_rgb, target_rgb)
    total_loss = decomposition_weight * decomposition_loss + fusion_weight * fusion_loss

    return {
        "loss": total_loss,
        "decomposition_loss": decomposition_loss,
        "fusion_loss": fusion_loss,
    }
