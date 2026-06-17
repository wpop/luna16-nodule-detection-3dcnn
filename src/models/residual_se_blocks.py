"""
Reusable residual squeeze-and-excitation blocks for 3D CNNs.
"""

import torch
import torch.nn as nn

from src.models.se_blocks import SEBlock3D


class ResidualSEBlock3D(nn.Module):
    """
    Residual 3D block with squeeze-and-excitation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Initialize residual squeeze-and-excitation block layers.
        """

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run residual squeeze-and-excitation block forward pass.
        """

        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x = x + residual

        return self.relu(x)
