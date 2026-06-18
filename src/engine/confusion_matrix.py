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


def save_confusion_matrix_plot(
    confusion_matrix: torch.Tensor,
    output_path: Path,
    class_names: list[str] | None = None,
) -> Path:
    """
    Save a confusion matrix tensor as a PNG figure.
    """

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = ["negative", "positive"]

    matrix = confusion_matrix.detach().cpu()

    figure, axes = plt.subplots()
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)

    axes.set_title("Validation Confusion Matrix")
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_xticks(range(len(class_names)), labels=class_names)
    axes.set_yticks(range(len(class_names)), labels=class_names)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axes.text(
                column_index,
                row_index,
                str(int(matrix[row_index, column_index].item())),
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)

    return output_path
