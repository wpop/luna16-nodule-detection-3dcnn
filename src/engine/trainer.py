"""
Training utilities for PyTorch models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.engine.metrics import accuracy_from_logits


class Trainer:
    """
    Minimal trainer for PyTorch classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        config: TrainConfig,
    ):
        """
        Initialize trainer.
        """

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(config.device)

        self.model.to(self.device)

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, float]:
        """
        Run one training step and return loss and accuracy.
        """

        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)
        accuracy = accuracy_from_logits(logits, labels)

        loss.backward()
        self.optimizer.step()

        return float(loss.item()), accuracy

    def train_epoch(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[float, float]:
        """
        Train model for one epoch and return average loss and accuracy.
        """

        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            loss, accuracy = self.train_step(
                images=images,
                labels=labels,
            )

            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        average_loss = total_loss / num_batches
        average_accuracy = total_accuracy / num_batches

        return average_loss, average_accuracy
