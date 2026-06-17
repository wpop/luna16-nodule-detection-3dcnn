import torch

from src.config.train_config import TrainConfig
from src.factories.optimizer_factory import create_optimizer
from src.factories.scheduler_factory import create_scheduler
from src.models.baseline_3dcnn import Baseline3DCNN


def test_create_scheduler():
    model = Baseline3DCNN()
    config = TrainConfig()

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        config=config,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

