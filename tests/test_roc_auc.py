import torch

from src.engine.roc_auc import compute_auc, compute_roc_curve


def test_compute_roc_curve():
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([0.1, 0.4, 0.35, 0.8])

    roc_curve = compute_roc_curve(labels, probabilities)

    assert roc_curve == {
        "false_positive_rates": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
        "true_positive_rates": [0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
    }


def test_compute_auc():
    false_positive_rates = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
    true_positive_rates = [0.0, 0.5, 0.5, 1.0, 1.0, 1.0]

    auc = compute_auc(false_positive_rates, true_positive_rates)

    assert auc == 0.75
