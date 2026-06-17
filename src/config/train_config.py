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
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    num_epochs: int = 20
    num_workers: int = 4
    device: str = "cuda"
