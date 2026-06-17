"""
Training utilities for PyTorch models.
"""

import torch
import torch.nn as nn

from src.config.train_config import TrainConfig


class Trainer:
    """
    Minimal trainer for one training step.
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

