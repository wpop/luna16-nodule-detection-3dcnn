import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.train_config import TrainConfig
from src.engine.validator import Validator
from src.models.baseline_3dcnn import Baseline3DCNN


def test_validate_epoch():
    model = Baseline3DCNN()
    loss_fn = nn.CrossEntropyLoss()
    config = TrainConfig(device="cpu")

    validator = Validator(
        model=model,
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

    (
        average_loss,
        average_accuracy,
        confusion_matrix,
        all_labels,
        all_probabilities,
    ) = validator.validate_epoch(loader)

    assert isinstance(average_loss, float)
    assert isinstance(average_accuracy, float)
    assert average_loss > 0.0
    assert 0.0 <= average_accuracy <= 1.0
    assert confusion_matrix.shape == (2, 2)
    assert confusion_matrix.dtype == torch.long
    assert confusion_matrix.sum().item() == len(dataset)
    assert torch.equal(all_labels, labels)
    assert all_probabilities.shape == (len(dataset),)
    assert all_probabilities.dtype == torch.float32
    assert torch.all((0.0 <= all_probabilities) & (all_probabilities <= 1.0))
