from src.models.baseline_3dcnn import Baseline3DCNN
from src.models.model_summary import (
    count_parameters,
    count_trainable_parameters,
)
from src.models.residual_3dcnn import Residual3DCNN


def test_count_parameters_baseline_model():
    model = Baseline3DCNN()

    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    assert total_parameters > 0
    assert trainable_parameters == total_parameters


def test_count_parameters_residual_model():
    model = Residual3DCNN()

    total_parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    assert total_parameters > 0
    assert trainable_parameters == total_parameters
