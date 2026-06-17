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

    def save_plot(self, output_path: Path) -> Path:
        """
        Save history metrics as a plot image.
        """

        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.train_loss) + 1)

        plt.figure()
        plt.plot(epochs, self.train_loss, marker="o", label="train_loss")
        plt.plot(epochs, self.val_loss, marker="o", label="val_loss")
        plt.plot(epochs, self.train_accuracy, marker="o", label="train_accuracy")
        plt.plot(epochs, self.val_accuracy, marker="o", label="val_accuracy")
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        return output_path
