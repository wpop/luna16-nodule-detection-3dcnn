"""
Preprocessing utilities for LUNA16 CT volumes.
"""

import numpy as np


def world_to_voxel(
    world_coord: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Convert world coordinates in millimeters to voxel coordinates.

    Parameters
    ----------
    world_coord : np.ndarray
        World coordinate in (x, y, z) order.

    origin : np.ndarray
        CT image origin in (x, y, z) order.

    spacing : np.ndarray
        CT voxel spacing in (x, y, z) order.

    Returns
    -------
    np.ndarray
        Voxel coordinate in (x, y, z) order.
    """

    return np.round((world_coord - origin) / spacing).astype(int)


def normalize_hu(
    volume: np.ndarray,
    hu_min: int = -1000,
    hu_max: int = 400,
) -> np.ndarray:
    """
    Clip Hounsfield Units and normalize to [0, 1].
    """

    volume = np.clip(volume, hu_min, hu_max)

    volume = (volume - hu_min) / (hu_max - hu_min)

    return volume.astype(np.float32)
