"""
Predict a class for a single 3D NumPy patch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config.train_config import TrainConfig
from src.engine.inference import predict_from_logits
from src.factories.model_factory import create_model

TARGET_PATCH_SHAPE = (64, 64, 64)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        choices=(
            "baseline",
            "residual",
            "residual_se",
            "multiscale",
            "multiscale_se",
            "vit3d",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/predictions/prediction.json"),
    )

    return parser.parse_args()


def pad_patch(patch: np.ndarray) -> np.ndarray:
    """
    Pad a 3D patch with zeros to the target model input shape.
    """

    if len(patch.shape) != 3:
        raise ValueError(f"Expected 3D patch, got shape {patch.shape}")

    if any(size > target for size, target in zip(patch.shape, TARGET_PATCH_SHAPE)):
        raise ValueError(
            f"Expected patch shape no larger than {TARGET_PATCH_SHAPE}, got {patch.shape}"
        )

    pad_width = [
        (0, target - size)
        for size, target in zip(patch.shape, TARGET_PATCH_SHAPE)
    ]

    return np.pad(patch, pad_width, mode="constant")


def load_patch(input_path: Path, device: torch.device) -> torch.Tensor:
    """
    Load a 3D NumPy patch and convert it to a model input tensor.
    """

    patch = np.load(input_path)
    patch = pad_patch(patch)

    return torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)


def main() -> None:
    """
    Run prediction for one NumPy patch.
    """

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(model_name=args.model_name, device=str(device))
    model = create_model(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    patch = load_patch(args.input, device)

    with torch.no_grad():
        logits = model(patch)

    prediction = predict_from_logits(logits)
    output = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "input": str(args.input),
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "positive_probability": prediction["positive_probability"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file)

    print(prediction)
    print("Saved prediction:", args.output)


if __name__ == "__main__":
    main()
