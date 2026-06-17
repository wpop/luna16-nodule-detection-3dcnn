"""
Factory for creating learning rate schedulers.
"""

import torch

from src.config.train_config import TrainConfig


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler.
    """

    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5,
    )

