"""
Training history utility for epoch metrics.
"""

import json
from pathlib import Path


class TrainingHistory:
    """
    Store training and validation metrics across epochs.
    """

    def __init__(self):
        """
        Initialize empty metric history.
        """

        self.train_loss: list[float] = []
        self.train_accuracy: list[float] = []
        self.val_loss: list[float] = []
        self.val_accuracy: list[float] = []
        self.learning_rate: list[float] = []

    def add_epoch(
        self,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        learning_rate: float,
    ) -> None:
        """
        Add metrics for one epoch.
        """

        self.train_loss.append(train_loss)
        self.train_accuracy.append(train_accuracy)
        self.val_loss.append(val_loss)
        self.val_accuracy.append(val_accuracy)
        self.learning_rate.append(learning_rate)

    def to_dict(self) -> dict[str, list[float]]:
        """
        Return history as a dictionary.
        """

        return {
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
        }

    def save_json(self, output_path: Path) -> Path:
        """
        Save history as a JSON file.
        """

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(self.to_dict(), output_file)

        return output_path
