import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


def test_train_epoch():
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

    images = torch.randn(4, 1, 64, 64, 64)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    dataset = TensorDataset(images, labels)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    average_loss = trainer.train_epoch(loader)

    assert isinstance(average_loss, float)
    assert average_loss > 0.0
