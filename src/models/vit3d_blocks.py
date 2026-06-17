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


class TransformerEncoderBlock3D(nn.Module):
    """
    Transformer encoder block for 3D patch tokens.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer encoder layers.
        """

        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run transformer encoder block forward pass.
        """

        attention_input = self.norm1(x)
        attention_output, _ = self.attention(
            attention_input,
            attention_input,
            attention_input,
        )
        x = x + attention_output

        return x + self.mlp(self.norm2(x))
