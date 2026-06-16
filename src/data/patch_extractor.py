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

    Parameters
    ----------
    volume : np.ndarray
        CT volume in NumPy format (z, y, x).

    voxel_x : int
        X coordinate of the patch center.

    voxel_y : int
        Y coordinate of the patch center.

    voxel_z : int
        Z coordinate of the patch center.

    patch_size : int
        Edge length of the cubic patch.

    Returns
    -------
    np.ndarray
        Extracted 3D patch.
    """

    half = patch_size // 2

    z_min = max(voxel_z - half, 0)
    z_max = min(voxel_z + half, volume.shape[0])

    y_min = max(voxel_y - half, 0)
    y_max = min(voxel_y + half, volume.shape[1])

    x_min = max(voxel_x - half, 0)
    x_max = min(voxel_x + half, volume.shape[2])

    return volume[
        z_min:z_max,
        y_min:y_max,
        x_min:x_max,
    ]

