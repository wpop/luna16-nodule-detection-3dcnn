import torch

from src.config.train_config import TrainConfig
from src.factories.optimizer_factory import create_optimizer
from src.factories.scheduler_factory import create_scheduler
from src.models.baseline_3dcnn import Baseline3DCNN


def test_create_scheduler():
    model = Baseline3DCNN()
    config = TrainConfig(
        scheduler_step_size=3,
        scheduler_gamma=0.25,
    )

    optimizer = create_optimizer(
        model=model,
        config=config,
    )

    scheduler = create_scheduler(
        optimizer=optimizer,
        config=config,
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == config.scheduler_step_size
    assert scheduler.gamma == config.scheduler_gamma
