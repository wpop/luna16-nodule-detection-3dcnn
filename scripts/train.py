"""
Minimal training entry point for the LUNA16 baseline 3D CNN.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config.train_config import TrainConfig
from src.data.luna_dataset import LunaDataset
from src.engine.checkpoint import CheckpointManager
from src.engine.early_stopping import EarlyStopping
from src.engine.trainer import Trainer
from src.engine.validator import Validator
from src.factories.optimizer_factory import create_optimizer
from src.factories.scheduler_factory import create_scheduler
from src.models.baseline_3dcnn import Baseline3DCNN


def main() -> None:
    """
    Run one debug training and validation epoch.
    """

    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data" / "raw" / "LUNA16"
    candidates_path = data_dir / "data-unversioned" / "part2" / "luna" / "candidates.csv"

    config = TrainConfig(
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=1,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    dataset = LunaDataset(
        candidates_path=candidates_path,
        data_dir=data_dir,
        patch_size=64,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = Baseline3DCNN()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_manager = CheckpointManager(project_root / "outputs" / "checkpoints")
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )

    validator = Validator(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    print("Device:", config.device)
    print("Dataset size:", len(dataset))
    print("Train size:", len(train_dataset))
    print("Validation size:", len(val_dataset))

    for epoch in range(config.num_epochs):
        train_loss, train_accuracy = trainer.train_epoch(
            train_loader,
            max_batches=5,
        )

        val_loss, val_accuracy = validator.validate_epoch(
            val_loader,
            max_batches=5,
        )

        checkpoint_path = checkpoint_manager.save_best_model(
            model=model,
            validation_loss=val_loss,
        )
        scheduler.step()

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print("Epoch:", epoch + 1)
        print("Train loss:", train_loss)
        print("Train accuracy:", train_accuracy)
        print("Validation loss:", val_loss)
        print("Validation accuracy:", val_accuracy)
        print("Learning rate:", current_learning_rate)
        if checkpoint_path is not None:
            print("Saved checkpoint:", checkpoint_path)

        if early_stopping.should_stop(val_loss):
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()
