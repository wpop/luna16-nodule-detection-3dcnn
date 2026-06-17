"""
Training utilities for PyTorch models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig


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
    ) -> float:
        """
        Run one training step.
        """

        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        logits = self.model(images)
        loss = self.loss_fn(logits, labels)

        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def train_epoch(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> float:
        """
        Train model for one epoch and return average loss.
        """

        total_loss = 0.0

        num_batches = 0

        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            loss = self.train_step(
                images=images,
                labels=labels,
            )
            total_loss += loss
            num_batches += 1

        average_loss = total_loss / num_batches

        return average_loss
