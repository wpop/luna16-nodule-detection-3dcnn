from pathlib import Path

from src.config.train_config import TrainConfig
from src.engine.checkpoint import CheckpointManager
from src.factories.optimizer_factory import create_optimizer
from src.models.baseline_3dcnn import Baseline3DCNN


def test_save_model(tmp_path: Path):
    model = Baseline3DCNN()

    config = TrainConfig(device="cpu")
    _ = create_optimizer(model, config)

    manager = CheckpointManager(tmp_path)

    checkpoint = manager.save_model(
        model=model,
        filename="model.pth",
    )

    assert checkpoint.exists()
    assert checkpoint.name == "model.pth"


def test_save_best_model(tmp_path: Path):
    model = Baseline3DCNN()

    manager = CheckpointManager(tmp_path)

    first_checkpoint = manager.save_best_model(
        model=model,
        validation_loss=0.8,
    )

    worse_checkpoint = manager.save_best_model(
        model=model,
        validation_loss=0.9,
    )

    better_checkpoint = manager.save_best_model(
        model=model,
        validation_loss=0.7,
    )

    assert first_checkpoint is not None
    assert first_checkpoint.exists()
    assert first_checkpoint.name == "best_model.pth"
    assert worse_checkpoint is None
    assert better_checkpoint is not None
    assert better_checkpoint.exists()
    assert better_checkpoint.name == "best_model.pth"
    assert manager.best_validation_loss == 0.7
