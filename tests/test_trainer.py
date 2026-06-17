import torch
import torch.nn as nn

from src.config.train_config import TrainConfig
from src.engine.trainer import Trainer
from src.models.baseline_3dcnn import Baseline3DCNN


def test_trainer_creation():
    model = Baseline3DCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    config = TrainConfig(device="cpu")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )

    assert trainer is not None

