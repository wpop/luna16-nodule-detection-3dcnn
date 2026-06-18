import torch

from src.engine.inference import predict_from_logits


def test_predict_from_logits():
    logits = torch.tensor([[1.0, 3.0]])
    probabilities = torch.softmax(logits, dim=1)

    prediction = predict_from_logits(logits)

    assert prediction == {
        "predicted_class": 1,
        "confidence": float(probabilities[0, 1].item()),
        "positive_probability": float(probabilities[0, 1].item()),
    }
