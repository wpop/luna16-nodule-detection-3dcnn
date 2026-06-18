"""
Export a trained model checkpoint to ONNX format.
"""

import argparse
from pathlib import Path

import onnx
import torch

from src.config.train_config import TrainConfig
from src.factories.model_factory import create_model


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
    parser.add_argument("--output", type=Path)

    return parser.parse_args()


def main() -> None:
    """
    Export a trained model checkpoint to ONNX.
    """

    args = parse_args()
    output_path = args.output or Path("outputs") / "onnx" / f"{args.model_name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(model_name=args.model_name, device="cpu")
    model = create_model(config)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 1, 64, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        # Pin the opset so exported graphs are reproducible across PyTorch versions.
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )

    # Validate immediately so broken exports fail before downstream tools use them.
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
    except Exception as error:
        raise RuntimeError(str(error)) from error

    print("Exported ONNX model:", output_path)
    print("ONNX model validation: PASSED")


if __name__ == "__main__":
    main()
