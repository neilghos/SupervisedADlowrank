"""Connected component labeling."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np
import torch
try:
    from kornia.contrib import connected_components
except ImportError:
    connected_components = None
from torch import Tensor


def connected_components_gpu(image: Tensor, num_iterations: int = 1000) -> Tensor:
    """Perform connected component labeling with fallback to CPU."""
    if connected_components is None:
        return connected_components_cpu(image)
    components = connected_components(image, num_iterations=num_iterations)

    # remap component values from 0 to N
    labels = components.unique()
    for new_label, old_label in enumerate(labels):
        components[components == old_label] = new_label

    return components.int()


def connected_components_cpu(image: Tensor) -> Tensor:
    """Connected component labeling on CPU."""
    device = image.device
    image_cpu = image.detach().cpu()
    components = torch.zeros_like(image_cpu, dtype=torch.int32)
    label_idx = 1
    for i, mask in enumerate(image_cpu):
        mask_np = mask.squeeze().numpy().astype(np.uint8)
        if not mask_np.any():
            continue
        n_labels, comps = cv2.connectedComponents(mask_np)
        for label in range(1, n_labels):
            components[i, 0][torch.from_numpy(comps == label)] = label_idx
            label_idx += 1
    return components.to(device)
