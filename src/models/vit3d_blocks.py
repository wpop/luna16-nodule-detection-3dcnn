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
