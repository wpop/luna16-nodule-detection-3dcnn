"""
Confusion matrix export utilities.
"""

import json
from pathlib import Path

import torch


def save_confusion_matrix_json(
    confusion_matrix: torch.Tensor,
    output_path: Path,
) -> Path:
    """
    Save a confusion matrix tensor as a JSON file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "confusion_matrix": confusion_matrix.detach().cpu().tolist(),
    }

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file)

    return output_path
