"""
Reusable multi-scale 3D blocks.
"""

import torch
import torch.nn as nn


class MultiScaleBlock3D(nn.Module):
    """
    Multi-scale 3D block with three convolution branches and a residual shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """
        Initialize multi-scale block layers.
        """

        super().__init__()

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )
        self.branch7 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

        self.fuse = nn.Conv3d(
            out_channels * 3,
            out_channels,
            kernel_size=1,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run multi-scale block forward pass.
        """

        branch3 = self.branch3(x)
        branch5 = self.branch5(x)
        branch7 = self.branch7(x)

        fused = torch.cat((branch3, branch5, branch7), dim=1)
        fused = self.fuse(fused)

        return self.relu(fused + self.shortcut(x))
