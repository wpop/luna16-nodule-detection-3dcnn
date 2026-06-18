"""
Inference utilities for model predictions.
"""

import torch


def predict_from_logits(logits: torch.Tensor) -> dict[str, float | int]:
    """
    Compute predicted class and probabilities from model logits.
    """

    probabilities = torch.softmax(logits, dim=1)
    predicted_class = int(torch.argmax(probabilities, dim=1).item())
    confidence = float(probabilities[0, predicted_class].item())
    positive_probability = float(probabilities[0, 1].item())

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "positive_probability": positive_probability,
    }
