"""
Validation utilities for PyTorch models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.engine.metrics import accuracy_from_logits, confusion_matrix_from_logits


class Validator:
    """
    Validator for PyTorch classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: TrainConfig,
    ):
        """
        Initialize validator.
        """

        self.model = model
        self.loss_fn = loss_fn
        self.device = torch.device(config.device)

        self.model.to(self.device)

    def validate_epoch(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[float, float, torch.Tensor]:
        """
        Validate model for one epoch without gradient updates.
        """

        self.model.eval()

        total_loss = 0.0
        total_accuracy = 0.0
        confusion_matrix = torch.zeros((2, 2), dtype=torch.long, device=self.device)
        num_batches = 0

        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.loss_fn(logits, labels)
                accuracy = accuracy_from_logits(logits, labels)
                batch_confusion_matrix = confusion_matrix_from_logits(logits, labels)

                total_loss += float(loss.item())
                total_accuracy += accuracy
                confusion_matrix += batch_confusion_matrix
                num_batches += 1

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches

        return average_loss, average_accuracy, confusion_matrix
