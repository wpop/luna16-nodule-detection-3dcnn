"""
Create example 3D patches for the Streamlit demo.
"""

from pathlib import Path

import numpy as np

from src.data.candidate_loader import filter_available_scans, load_csv
from src.data.patch_extractor import extract_patch
from src.data.preprocessing import (
    find_ct_path,
    load_ct_volume,
    normalize_hu,
    world_to_voxel,
)

PATCH_SHAPE = (64, 64, 64)


def center_pad_patch(patch: np.ndarray) -> np.ndarray:
    """
    Center a smaller 3D patch inside a zero-padded 64x64x64 patch.
    """

    if len(patch.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch.shape}")

    if any(size > target for size, target in zip(patch.shape, PATCH_SHAPE)):
        raise ValueError(f"Patch shape {patch.shape} is larger than {PATCH_SHAPE}")

    output = np.zeros(PATCH_SHAPE, dtype=np.float32)
    starts = [
        (target - size) // 2
        for size, target in zip(patch.shape, PATCH_SHAPE)
    ]
    slices = tuple(
        slice(start, start + size)
        for start, size in zip(starts, patch.shape)
    )
    output[slices] = patch.astype(np.float32)

    return output


def extract_candidate_patch(
    candidates_path: Path,
    mhd_files: list[Path],
    label: int,
) -> np.ndarray | None:
    """
    Extract a normalized patch centered on an available LUNA16 candidate.
    """

    if not candidates_path.exists() or not mhd_files:
        return None

    candidates = load_csv(candidates_path)
    available_series_uids = {path.stem for path in mhd_files}
    candidates = filter_available_scans(candidates, available_series_uids)
    candidates = candidates[candidates["class"] == label]

    if candidates.empty:
        return None

    row = candidates.iloc[0]
    series_uid = row["seriesuid"]
    world_coord = np.array(
        [row["coordX"], row["coordY"], row["coordZ"]],
        dtype=np.float32,
    )

    ct_path = find_ct_path(series_uid, mhd_files)
    volume, origin, spacing = load_ct_volume(ct_path)
    voxel_x, voxel_y, voxel_z = world_to_voxel(world_coord, origin, spacing)
    patch = extract_patch(volume, voxel_x, voxel_y, voxel_z, patch_size=64)

    return normalize_hu(patch)


def load_fallback_positive_patch(project_root: Path) -> np.ndarray:
    """
    Load and pad the existing positive demo patch when raw LUNA16 data is absent.
    """

    input_path = project_root / "data" / "patches" / "first_positive_nodule_patch.npy"
    patch = np.load(input_path)

    return center_pad_patch(patch)


def create_negative_patch() -> np.ndarray:
    """
    Create an air-like normalized negative example patch.
    """

    return np.zeros(PATCH_SHAPE, dtype=np.float32)


def save_patch(path: Path, patch: np.ndarray) -> None:
    """
    Save one patch and print its summary statistics.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, patch.astype(np.float32))

    print("Saved:", path)
    print("Shape:", patch.shape)
    print("Min:", float(np.min(patch)))
    print("Max:", float(np.max(patch)))
    print("Mean:", float(np.mean(patch)))


def main() -> None:
    """
    Create positive and negative example patches.
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw" / "LUNA16"
    candidates_path = data_dir / "data-unversioned" / "part2" / "luna" / "candidates.csv"
    mhd_files = sorted(data_dir.rglob("*.mhd"))

    positive_patch = extract_candidate_patch(candidates_path, mhd_files, label=1)
    if positive_patch is None:
        positive_patch = load_fallback_positive_patch(project_root)

    negative_patch = extract_candidate_patch(candidates_path, mhd_files, label=0)
    if negative_patch is None:
        negative_patch = create_negative_patch()

    examples_dir = project_root / "examples"
    save_patch(examples_dir / "positive_patch.npy", positive_patch)
    save_patch(examples_dir / "negative_patch.npy", negative_patch)


if __name__ == "__main__":
    main()
