import torch

from src.models.multiscale_se_blocks import MultiScaleSEBlock3D


def test_multiscale_se_block_3d_same_channels_output_shape():
    block = MultiScaleSEBlock3D(
        in_channels=16,
        out_channels=16,
    )
    x = torch.randn(2, 16, 32, 32, 32)

    output = block(x)

    assert output.shape == (2, 16, 32, 32, 32)


def test_multiscale_se_block_3d_different_channels_output_shape():
    block = MultiScaleSEBlock3D(
        in_channels=16,
        out_channels=32,
    )
    x = torch.randn(2, 16, 32, 32, 32)

    output = block(x)

    assert output.shape == (2, 32, 32, 32, 32)
