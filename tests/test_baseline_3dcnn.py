import torch

from src.models.baseline_3dcnn import Baseline3DCNN


def test_baseline_3dcnn_forward():
    model = Baseline3DCNN()

    x = torch.randn(4, 1, 64, 64, 64)
    logits = model(x)

    assert logits.shape == (4, 2)

