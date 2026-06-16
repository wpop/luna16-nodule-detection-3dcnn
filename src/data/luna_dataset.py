"""
PyTorch dataset for the LUNA16 lung nodule dataset.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.candidate_loader import load_csv, filter_available_scans
from src.data.preprocessing import (
    find_ct_path,
    load_ct_volume,
    world_to_voxel,
    normalize_hu,
)
from src.data.patch_extractor import extract_patch


class LunaDataset(Dataset):
    """
    PyTorch Dataset that returns normalized 3D CT patches and labels.
    """

    def __init__(
        self,
        candidates_path: Path,
        data_dir: Path,
        patch_size: int = 64,
    ):
        """
        Initialize the dataset.
        """

        self.mhd_files = sorted(data_dir.rglob("*.mhd"))
        self.available_series_uids = {p.stem for p in self.mhd_files}

        candidates = load_csv(candidates_path)

        self.candidates = filter_available_scans(
            candidates,
            self.available_series_uids,
        )

        self.patch_size = patch_size

    def __len__(self):
        """
        Return the number of available candidates.
        """

        return len(self.candidates)

    def __getitem__(self, index):
        """
        Return one normalized 3D patch and its label.
        """

        row = self.candidates.iloc[index]

        series_uid = row["seriesuid"]

        world_coord = np.array(
            [row["coordX"], row["coordY"], row["coordZ"]],
            dtype=np.float32,
        )

        label = int(row["class"])

        ct_path = find_ct_path(
            series_uid=series_uid,
            mhd_files=self.mhd_files,
        )

        volume, origin, spacing = load_ct_volume(ct_path)

        voxel_coord = world_to_voxel(
            world_coord=world_coord,
            origin=origin,
            spacing=spacing,
        )

        voxel_x = voxel_coord[0]
        voxel_y = voxel_coord[1]
        voxel_z = voxel_coord[2]

        patch = extract_patch(
            volume=volume,
            voxel_x=voxel_x,
            voxel_y=voxel_y,
            voxel_z=voxel_z,
            patch_size=self.patch_size,
        )

        patch = normalize_hu(patch)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        patch = np.expand_dims(patch, axis=0)

        image = torch.from_numpy(patch).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label
