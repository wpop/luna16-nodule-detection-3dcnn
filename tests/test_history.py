import json
from pathlib import Path

from src.engine.history import TrainingHistory


def test_training_history_add_epoch_and_to_dict():
    history = TrainingHistory()

    history.add_epoch(
        train_loss=0.5,
        train_accuracy=0.8,
        val_loss=0.4,
        val_accuracy=0.9,
        learning_rate=0.001,
    )

    assert history.to_dict() == {
        "train_loss": [0.5],
        "train_accuracy": [0.8],
        "val_loss": [0.4],
        "val_accuracy": [0.9],
        "learning_rate": [0.001],
    }


def test_training_history_save_json(tmp_path: Path):
    history = TrainingHistory()

    history.add_epoch(
        train_loss=0.5,
        train_accuracy=0.8,
        val_loss=0.4,
        val_accuracy=0.9,
        learning_rate=0.001,
    )

    output_path = tmp_path / "metrics" / "training_history.json"

    saved_path = history.save_json(output_path)

    with saved_path.open("r", encoding="utf-8") as input_file:
        saved_history = json.load(input_file)

    assert saved_path == output_path
    assert saved_history == history.to_dict()
