import torch

from src.models.vit3d_blocks import (
    ClassToken,
    PatchEmbedding3D,
    PositionEmbedding3D,
)


def test_patch_embedding_3d_output_shape():
    patch_embedding = PatchEmbedding3D()
    x = torch.randn(2, 1, 64, 64, 64)

    output = patch_embedding(x)

    assert output.shape == (2, 64, 128)


def test_class_token_shape():
    class_token = ClassToken(embed_dim=128)
    x = torch.randn(2, 64, 128)

    output = class_token(x)

    assert output.shape == (2, 65, 128)


def test_position_embedding_shape():
    position_embedding = PositionEmbedding3D(
        num_patches=64,
        embed_dim=128,
    )
    x = torch.randn(2, 65, 128)

    output = position_embedding(x)

    assert output.shape == (2, 65, 128)
