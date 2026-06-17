import torch

from src.models.vit3d_blocks import PatchEmbedding3D


def test_patch_embedding_3d_output_shape():
    patch_embedding = PatchEmbedding3D()
    x = torch.randn(2, 1, 64, 64, 64)

    output = patch_embedding(x)

    assert output.shape == (2, 64, 128)
