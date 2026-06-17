"""
Utilities for extracting 3D patches from CT volumes.
"""

import numpy as np


def extract_patch(
    volume: np.ndarray,
    voxel_x: int,
    voxel_y: int,
    voxel_z: int,
    patch_size: int = 64,
) -> np.ndarray:
    """
    Extract a cubic 3D patch centered at the specified voxel.

    If the patch crosses the volume boundary, padding is added.
    """

    half = patch_size // 2

    z_min = voxel_z - half
    z_max = voxel_z + half

    y_min = voxel_y - half
    y_max = voxel_y + half

    x_min = voxel_x - half
    x_max = voxel_x + half

    pad_z_before = max(0, -z_min)
    pad_y_before = max(0, -y_min)
    pad_x_before = max(0, -x_min)

    pad_z_after = max(0, z_max - volume.shape[0])
    pad_y_after = max(0, y_max - volume.shape[1])
    pad_x_after = max(0, x_max - volume.shape[2])

    z_min = max(z_min, 0)
    y_min = max(y_min, 0)
    x_min = max(x_min, 0)

    z_max = min(z_max, volume.shape[0])
    y_max = min(y_max, volume.shape[1])
    x_max = min(x_max, volume.shape[2])

    patch = volume[
        z_min:z_max,
        y_min:y_max,
        x_min:x_max,
    ]

    patch = np.pad(
        patch,
        pad_width=(
            (pad_z_before, pad_z_after),
            (pad_y_before, pad_y_after),
            (pad_x_before, pad_x_after),
        ),
        mode="constant",
        constant_values=-1000,
    )

    return patch
