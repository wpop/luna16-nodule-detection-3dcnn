"""
Reusable squeeze-and-excitation blocks for 3D CNNs.
"""

import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    """
    Squeeze-and-excitation block for 3D feature maps.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ):
        """
        Initialize squeeze-and-excitation layers.
        """

        super().__init__()

        reduced_channels = max(1, channels // reduction)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise squeeze-and-excitation.
        """

        weights = self.pool(x)
        weights = self.excitation(weights)
        weights = weights.view(x.size(0), x.size(1), 1, 1, 1)

        return x * weights
