import torch

from src.models.multiscale_se_3dcnn import MultiScaleSE3DCNN


def test_multiscale_se_3dcnn_forward_shape():
    model = MultiScaleSE3DCNN()
    x = torch.randn(4, 1, 64, 64, 64)

    output = model(x)

    assert output.shape == (4, 2)
