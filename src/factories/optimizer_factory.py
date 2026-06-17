"""
Factory for creating PyTorch optimizers.
"""

import torch
import torch.nn as nn

from src.config.train_config import TrainConfig


def create_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    """

    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
