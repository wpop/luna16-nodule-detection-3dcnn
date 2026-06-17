"""
3D Vision Transformer model for LUNA16 nodule classification.
"""

import torch
import torch.nn as nn

from src.models.vit3d_blocks import (
    ClassToken,
    PatchEmbedding3D,
    PositionEmbedding3D,
    TransformerEncoderBlock3D,
)


class VisionTransformer3D(nn.Module):
    """
    Vision Transformer for 3D volume classification.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize 3D Vision Transformer layers.
        """

        super().__init__()

        num_patches = (64 // patch_size) ** 3

        self.patch_embedding = PatchEmbedding3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.class_token = ClassToken(embed_dim=embed_dim)
        self.position_embedding = PositionEmbedding3D(
            num_patches=num_patches,
            embed_dim=embed_dim,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock3D(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.
        """

        x = self.patch_embedding(x)
        x = self.class_token(x)
        x = self.position_embedding(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        x = self.norm(x)
        cls_token = x[:, 0]

        return self.head(cls_token)
