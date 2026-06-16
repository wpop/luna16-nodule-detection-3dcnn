"""
Data transforms for LUNA16 3D CT patches.
"""

import numpy as np
import torch

from src.data.preprocessing import normalize_hu


class NormalizeHU:
    """
    Normalize CT Hounsfield Units to [0, 1].
    """

    def __call__(self, patch: np.ndarray) -> np.ndarray:
        return normalize_hu(patch)


class ToTensor3D:
    """
    Convert a 3D NumPy patch to a PyTorch tensor.

    Input shape:
        (D, H, W)

    Output shape:
        (1, D, H, W)
    """

    def __call__(self, patch: np.ndarray) -> torch.Tensor:
        patch = np.expand_dims(patch, axis=0)
        return torch.from_numpy(patch).float()

