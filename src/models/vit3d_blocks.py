"""
Reusable blocks for 3D Vision Transformer models.
"""

import torch
import torch.nn as nn


class PatchEmbedding3D(nn.Module):
    """
    Convert 3D patches into embedding tokens.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        embed_dim: int = 128,
    ):
        """
        Initialize patch embedding projection.
        """

        super().__init__()

        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input volume into patch embeddings.
        """

        x = self.projection(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)

        return x


class ClassToken(nn.Module):
    """
    Prepend a learnable class token to a patch sequence.
    """

    def __init__(self, embed_dim: int):
        """
        Initialize class token parameter.
        """

        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add class token to patch embeddings.
        """

        batch_size = x.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)

        return torch.cat((cls_token, x), dim=1)


class PositionEmbedding3D(nn.Module):
    """
    Add learnable position embeddings to 3D patch tokens.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
    ):
        """
        Initialize position embedding parameter.
        """

        super().__init__()

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add position embeddings to token sequence.
        """

        return x + self.position_embedding
