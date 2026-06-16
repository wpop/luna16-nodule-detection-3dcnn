"""
Preprocessing utilities for LUNA16 CT volumes.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def find_ct_path(
    series_uid: str,
    mhd_files: list[Path],
) -> Path:
    """
    Find the CT file corresponding to a series UID.

    If duplicate files exist, return the first match.
    """

    matches = [path for path in mhd_files if path.stem == series_uid]

    if not matches:
        raise FileNotFoundError(f"Could not locate CT for {series_uid}")

    return matches[0]


def load_ct_volume(ct_path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a CT volume from a .mhd file.
    """

    image = sitk.ReadImage(str(ct_path))
    volume = sitk.GetArrayFromImage(image)

    origin = np.array(image.GetOrigin(), dtype=np.float32)
    spacing = np.array(image.GetSpacing(), dtype=np.float32)

    return volume, origin, spacing


def world_to_voxel(
    world_coord: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> np.ndarray:
    """
    Convert world coordinates in millimeters to voxel coordinates.
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
