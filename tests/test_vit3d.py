import torch

from src.models.vit3d import VisionTransformer3D


def test_vit3d_forward_shape():
    model = VisionTransformer3D()
    x = torch.randn(2, 1, 64, 64, 64)

    output = model(x)

    assert output.shape == (2, 2)
