"""
Minimal training entry point for the LUNA16 baseline 3D CNN.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.data.luna_dataset import LunaDataset
from src.engine.trainer import Trainer
from src.factories.optimizer_factory import create_optimizer
from src.models.baseline_3dcnn import Baseline3DCNN


def main() -> None:
    """
    Run one training step to validate the full pipeline.
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

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    model = Baseline3DCNN()
    optimizer = create_optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )

    images, labels = next(iter(loader))

    loss = trainer.train_step(
        images=images,
        labels=labels,
    )

    print("Batch images:", images.shape)
    print("Batch labels:", labels.shape)
    print("Loss:", loss)


if __name__ == "__main__":
    main()

