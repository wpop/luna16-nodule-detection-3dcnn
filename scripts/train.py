"""
Minimal training entry point for the LUNA16 baseline 3D CNN.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.config.train_config import TrainConfig
from src.data.luna_dataset import LunaDataset
from src.engine.benchmark import BenchmarkResult
from src.engine.checkpoint import CheckpointManager
from src.engine.early_stopping import EarlyStopping
from src.engine.experiment import ExperimentManager
from src.engine.history import TrainingHistory
from src.engine.trainer import Trainer
from src.engine.validator import Validator
from src.factories.model_factory import create_model
from src.factories.optimizer_factory import create_optimizer
from src.factories.scheduler_factory import create_scheduler
from src.models.model_summary import (
    count_parameters,
    count_trainable_parameters,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "baseline",
            "residual",
            "residual_se",
            "multiscale",
            "multiscale_se",
            "vit3d",
        ),
        default="baseline",
    )

    return parser.parse_args()


def main() -> None:
    """
    Run one debug training and validation epoch.
    """

    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / "data" / "raw" / "LUNA16"
    candidates_path = data_dir / "data-unversioned" / "part2" / "luna" / "candidates.csv"

    config = TrainConfig(
        model_name=args.model,
        batch_size=4,
        learning_rate=1e-3,
        num_epochs=1,
        num_workers=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    experiment = ExperimentManager(
        project_root=project_root,
        model_name=config.model_name,
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

    model = create_model(config)
    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    loss_fn = nn.CrossEntropyLoss()
    checkpoint_manager = CheckpointManager(experiment.checkpoint_dir)
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )
    history = TrainingHistory()

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
    print("Model:", config.model_name)
    print("Total parameters:", total_parameters)
    print("Trainable parameters:", trainable_parameters)
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
        history.add_epoch(
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=current_learning_rate,
        )

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

    history_path = history.save_json(experiment.metrics_dir / "training_history.json")
    plot_path = history.save_plot(experiment.figures_dir / "training_history.png")
    benchmark_result = BenchmarkResult(
        model_name=config.model_name,
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        train_loss=history.train_loss[-1],
        val_loss=history.val_loss[-1],
        train_accuracy=history.train_accuracy[-1],
        val_accuracy=history.val_accuracy[-1],
        learning_rate=history.learning_rate[-1],
    )
    benchmark_path = benchmark_result.save_json(
        experiment.results_dir / "benchmark.json"
    )

    print("Experiment directory:", experiment.experiment_dir)
    print("Training history:", history.to_dict())
    print("Saved history:", history_path)
    print("Saved plot:", plot_path)
    print("Saved benchmark:", benchmark_path)


if __name__ == "__main__":
    main()
