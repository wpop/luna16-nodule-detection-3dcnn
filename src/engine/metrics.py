"""
Metrics for training and validation.
"""

import torch


def accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute classification accuracy from model logits.
    """

    predictions = torch.argmax(logits, dim=1)

    correct = (predictions == labels).sum().item()
    total = labels.numel()

    return correct / total

