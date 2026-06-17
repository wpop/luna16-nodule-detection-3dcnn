import torch

from src.models.residual_3dcnn import Residual3DCNN


def test_residual_3dcnn_forward_shape():
    model = Residual3DCNN()
    x = torch.randn(4, 1, 64, 64, 64)

    output = model(x)

    assert output.shape == (4, 2)
