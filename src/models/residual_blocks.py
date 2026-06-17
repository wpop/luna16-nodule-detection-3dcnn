"""
Reusable 3D residual blocks.
"""

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Basic 3D residual block with an optional channel projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Initialize residual block layers.
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

        if in_channels != out_channels:
            self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run residual block forward pass.
        """

        residual = self.skip(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual

        return self.relu(x)
