"""
Residual squeeze-and-excitation 3D CNN model for LUNA16 nodule classification.
"""

import torch
import torch.nn as nn

from src.models.residual_se_blocks import ResidualSEBlock3D


class ResidualSE3DCNN(nn.Module):
    """
    3D CNN classifier built with residual squeeze-and-excitation blocks.
    """

    def __init__(self, num_classes: int = 2):
        """
        Initialize residual squeeze-and-excitation 3D CNN layers.
        """

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualSEBlock3D(16, 16),
            nn.MaxPool3d(2),
            ResidualSEBlock3D(16, 32),
            nn.MaxPool3d(2),
            ResidualSEBlock3D(32, 64),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass.
        """

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x
