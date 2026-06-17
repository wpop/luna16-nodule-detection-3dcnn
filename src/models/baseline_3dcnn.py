"""
Baseline 3D CNN model for LUNA16 nodule classification.
"""

import torch
import torch.nn as nn


class Baseline3DCNN(nn.Module):
    """
    Simple baseline 3D CNN for binary nodule classification.

    The model uses adaptive pooling, so it is not tightly coupled
    to a fixed 3D patch size.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
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
