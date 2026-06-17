"""
Training configuration for the LUNA16 project.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """
    Stores training hyperparameters.
    """

    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 20
    num_workers: int = 4
    device: str = "cuda"

