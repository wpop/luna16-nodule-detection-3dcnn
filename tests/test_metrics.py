import torch

from src.engine.metrics import accuracy_from_logits


def test_accuracy_from_logits():
    logits = torch.tensor(
        [
            [2.0, 0.1],
            [0.2, 1.5],
            [3.0, 0.1],
            [0.1, 2.0],
        ]
    )

    labels = torch.tensor([0, 1, 1, 1])

    accuracy = accuracy_from_logits(logits, labels)

    assert accuracy == 0.75

