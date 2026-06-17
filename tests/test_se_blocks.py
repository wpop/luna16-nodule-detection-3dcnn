import torch

from src.models.se_blocks import SEBlock3D


def test_se_block_3d_output_shape_and_dtype():
    block = SEBlock3D(channels=16)
    x = torch.randn(2, 16, 32, 32, 32)

    output = block(x)

    assert output.shape == (2, 16, 32, 32, 32)
    assert output.dtype == x.dtype
