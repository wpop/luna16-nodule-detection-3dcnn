import pytest

from src.config.train_config import TrainConfig
from src.factories.model_factory import create_model
from src.models.baseline_3dcnn import Baseline3DCNN
from src.models.multiscale_3dcnn import MultiScale3DCNN
from src.models.residual_3dcnn import Residual3DCNN


def test_create_baseline_model():
    config = TrainConfig(model_name="baseline")

    model = create_model(config)

    assert isinstance(model, Baseline3DCNN)


def test_create_residual_model():
    config = TrainConfig(model_name="residual")

    model = create_model(config)

    assert isinstance(model, Residual3DCNN)


def test_create_multiscale_model():
    config = TrainConfig(model_name="multiscale")

    model = create_model(config)

    assert isinstance(model, MultiScale3DCNN)


def test_create_unknown_model_raises_value_error():
    config = TrainConfig(model_name="unknown")

    with pytest.raises(ValueError):
        create_model(config)
