import torch

from src.models.residual_se_3dcnn import ResidualSE3DCNN


def test_residual_se_3dcnn_forward_shape():
    model = ResidualSE3DCNN()
    x = torch.randn(4, 1, 64, 64, 64)

    output = model(x)

    assert output.shape == (4, 2)
