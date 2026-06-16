"""
PyTorch dataset for the LUNA16 lung nodule dataset.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.candidate_loader import load_csv, filter_available_scans


class LunaDataset(Dataset):
    """
    Dataset for loading LUNA16 candidate rows.

    This is the first minimal version.
    It only loads and stores available candidates.
    """

    def __init__(
        self,
        candidates_path: Path,
        available_series_uids: set[str],
    ):
        """
        Initialize the dataset.
        """

        candidates = load_csv(candidates_path)

        self.candidates = filter_available_scans(
            candidates,
            available_series_uids,
        )

    def __len__(self):
        """
        Return the number of available candidates.
        """

        return len(self.candidates)

    def __getitem__(self, index):
        """
        Return one candidate row.

        Full patch loading will be added later.
        """

        row = self.candidates.iloc[index]

        label = int(row["class"])

        return {
            "seriesuid": row["seriesuid"],
            "coordX": float(row["coordX"]),
            "coordY": float(row["coordY"]),
            "coordZ": float(row["coordZ"]),
            "label": torch.tensor(label, dtype=torch.long),
        }
