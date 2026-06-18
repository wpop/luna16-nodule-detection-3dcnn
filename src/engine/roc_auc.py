"""
ROC curve and AUC utilities.
"""

from pathlib import Path

import torch


def compute_roc_curve(
    labels: torch.Tensor,
    probabilities: torch.Tensor,
) -> dict[str, list[float]]:
    """
    Compute ROC curve points from binary labels and positive-class probabilities.
    """

    labels = labels.detach().cpu().long()
    probabilities = probabilities.detach().cpu().float()
    thresholds = torch.unique(probabilities).sort(descending=True).values

    false_positive_rates = [0.0]
    true_positive_rates = [0.0]

    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())

    for threshold in thresholds:
        predictions = probabilities >= threshold

        true_positives = int(((predictions == 1) & (labels == 1)).sum().item())
        false_positives = int(((predictions == 1) & (labels == 0)).sum().item())

        true_positive_rate = true_positives / positives if positives > 0 else 0.0
        false_positive_rate = false_positives / negatives if negatives > 0 else 0.0

        true_positive_rates.append(true_positive_rate)
        false_positive_rates.append(false_positive_rate)

    true_positive_rates.append(1.0)
    false_positive_rates.append(1.0)

    return {
        "false_positive_rates": false_positive_rates,
        "true_positive_rates": true_positive_rates,
    }


def compute_auc(
    false_positive_rates: list[float],
    true_positive_rates: list[float],
) -> float:
    """
    Compute AUC using the trapezoidal rule.
    """

    auc = 0.0

    for index in range(1, len(false_positive_rates)):
        width = false_positive_rates[index] - false_positive_rates[index - 1]
        height = (
            true_positive_rates[index] + true_positive_rates[index - 1]
        ) / 2.0
        auc += width * height

    return auc


def save_roc_curve_plot(
    false_positive_rates: list[float],
    true_positive_rates: list[float],
    auc: float,
    output_path: Path,
) -> Path:
    """
    Save a ROC curve as a PNG figure.
    """

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots()
    axes.plot(
        false_positive_rates,
        true_positive_rates,
        label=f"AUC = {auc:.3f}",
    )
    axes.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Random classifier")
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("ROC Curve")
    axes.legend()
    axes.grid(True)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)

    return output_path
