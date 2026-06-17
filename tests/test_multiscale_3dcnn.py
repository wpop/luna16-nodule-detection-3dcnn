import torch

from src.models.multiscale_3dcnn import MultiScale3DCNN


def test_multiscale_3dcnn_forward_shape():
    model = MultiScale3DCNN()
    x = torch.randn(4, 1, 64, 64, 64)

    output = model(x)

    assert output.shape == (4, 2)
