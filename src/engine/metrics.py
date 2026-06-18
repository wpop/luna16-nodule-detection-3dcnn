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


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
) -> torch.Tensor:
    """
    Compute a confusion matrix from model logits.
    """

    predictions = torch.argmax(logits, dim=1)
    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
        device=logits.device,
    )

    for label, prediction in zip(labels, predictions):
        matrix[label, prediction] += 1

    return matrix
