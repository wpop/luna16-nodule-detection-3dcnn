"""
Checkpoint utilities for saving model weights.
"""

from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Save model checkpoints.
    """

    def __init__(self, checkpoint_dir: Path):
        """
        Initialize checkpoint manager.
        """

        self.checkpoint_dir = checkpoint_dir
        self.best_validation_loss: float | None = None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: nn.Module,
        filename: str,
    ) -> Path:
        """
        Save model state dictionary.
        """

        checkpoint_path = self.checkpoint_dir / filename

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        return checkpoint_path

    def save_best_model(
        self,
        model: nn.Module,
        validation_loss: float,
    ) -> Path | None:
        """
        Save model only when validation loss improves.
        """

        if (
            self.best_validation_loss is not None
            and validation_loss >= self.best_validation_loss
        ):
            return None

        self.best_validation_loss = validation_loss

        return self.save_model(
            model=model,
            filename="best_model.pth",
        )
