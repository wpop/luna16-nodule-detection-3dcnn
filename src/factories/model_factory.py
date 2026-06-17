"""
Factory for creating model instances.
"""

import torch.nn as nn

from src.config.train_config import TrainConfig
from src.models.baseline_3dcnn import Baseline3DCNN
from src.models.multiscale_3dcnn import MultiScale3DCNN
from src.models.multiscale_se_3dcnn import MultiScaleSE3DCNN
from src.models.residual_3dcnn import Residual3DCNN
from src.models.residual_se_3dcnn import ResidualSE3DCNN


def create_model(config: TrainConfig) -> nn.Module:
    """
    Create model from training configuration.
    """

    if config.model_name == "baseline":
        return Baseline3DCNN()

    if config.model_name == "residual":
        return Residual3DCNN()

    if config.model_name == "residual_se":
        return ResidualSE3DCNN()

    if config.model_name == "multiscale":
        return MultiScale3DCNN()

    if config.model_name == "multiscale_se":
        return MultiScaleSE3DCNN()

    raise ValueError(f"Unknown model name: {config.model_name}")
