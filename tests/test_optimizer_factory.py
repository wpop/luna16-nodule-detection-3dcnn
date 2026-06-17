import torch

from src.config.train_config import TrainConfig
from src.factories.optimizer_factory import create_optimizer
from src.models.baseline_3dcnn import Baseline3DCNN


def test_create_optimizer():
    model = Baseline3DCNN()
    config = TrainConfig(learning_rate=1e-3)

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    assert isinstance(optimizer, torch.optim.Adam)

