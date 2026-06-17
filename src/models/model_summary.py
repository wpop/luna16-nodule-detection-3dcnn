"""
Model summary utilities.
"""

import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Count all model parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
