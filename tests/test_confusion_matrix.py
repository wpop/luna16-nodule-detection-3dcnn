import json

import torch

from src.engine.confusion_matrix import (
    save_confusion_matrix_json,
    save_confusion_matrix_plot,
)


def test_save_confusion_matrix_json(tmp_path):
    confusion_matrix = torch.tensor([[2, 1], [0, 3]])
    output_path = tmp_path / "metrics" / "confusion_matrix.json"

    saved_path = save_confusion_matrix_json(confusion_matrix, output_path)

    with output_path.open("r", encoding="utf-8") as output_file:
        data = json.load(output_file)

    assert saved_path == output_path
    assert data == {
        "confusion_matrix": [
            [2, 1],
            [0, 3],
        ],
    }


def test_save_confusion_matrix_plot(tmp_path):
    confusion_matrix = torch.tensor([[2, 1], [0, 3]])
    output_path = tmp_path / "figures" / "confusion_matrix.png"

    saved_path = save_confusion_matrix_plot(confusion_matrix, output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
