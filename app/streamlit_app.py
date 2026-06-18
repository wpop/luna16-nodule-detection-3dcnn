"""
Streamlit demo app for single 3D patch inference.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.config.train_config import TrainConfig
from src.engine.inference import predict_from_logits
from src.factories.model_factory import create_model

TARGET_PATCH_SHAPE = (64, 64, 64)
MODEL_NAMES = (
    "baseline",
    "residual",
    "residual_se",
    "multiscale",
    "multiscale_se",
    "vit3d",
)
PATCH_SOURCES = (
    "Example positive patch",
    "Example negative patch",
    "Upload custom .npy patch",
)
EXAMPLE_PATCH_PATHS = {
    "Example positive patch": Path("examples/positive_patch.npy"),
    "Example negative patch": Path("examples/negative_patch.npy"),
}


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


def plot_middle_axial_slice(patch: np.ndarray) -> plt.Figure:
    """
    Plot the middle axial slice of a 3D patch.
    """

    middle_slice_index = patch.shape[0] // 2
    figure, axes = plt.subplots(figsize=(4, 4))
    axes.imshow(patch[middle_slice_index], cmap="gray")
    axes.axis("off")
    figure.tight_layout()

    return figure


def run_inference(
    model_name: str,
    checkpoint_path: Path,
    patch: np.ndarray,
) -> dict[str, float | int]:
    """
    Run CPU inference for a single padded patch.
    """

    config = TrainConfig(model_name=model_name, device="cpu")
    model = create_model(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)

    return predict_from_logits(logits)


def load_selected_patch(
    patch_source: str,
    uploaded_file,
) -> np.ndarray | None:
    """
    Load a patch from the selected source.
    """

    if patch_source == "Upload custom .npy patch":
        if uploaded_file is None:
            return None

        return np.load(uploaded_file)

    return np.load(EXAMPLE_PATCH_PATHS[patch_source])


def main() -> None:
    """
    Run the Streamlit app.
    """

    st.set_page_config(page_title="LUNA16 Lung Nodule Classification", layout="centered")
    st.title("LUNA16 Lung Nodule Classification")
    st.caption(
        "Research demo for 3D CNN and Vision Transformer inference on LUNA16 patches."
    )

    model_name = st.sidebar.selectbox("Model", MODEL_NAMES)
    checkpoint_path_text = st.sidebar.text_input(
        "Checkpoint path",
        "outputs/experiments/20260618_103653_baseline/checkpoints/best_model.pth",
    )
    checkpoint_path = Path(checkpoint_path_text)
    st.sidebar.markdown("### Run configuration")
    st.sidebar.write("Model:", model_name)
    st.sidebar.write("Checkpoint:", checkpoint_path.name)
    st.sidebar.write("Device:", "CPU")

    patch_source = st.selectbox("Patch source", PATCH_SOURCES)
    uploaded_file = None

    if patch_source == "Upload custom .npy patch":
        uploaded_file = st.file_uploader("Upload .npy patch", type=["npy"])

    if patch_source == "Upload custom .npy patch" and uploaded_file is None:
        st.info("Upload a 3D NumPy patch to begin.")
        st.caption("This demo is intended for research purposes only.")
        return

    try:
        patch = load_selected_patch(patch_source, uploaded_file)
        if patch is None:
            st.info("Upload a 3D NumPy patch to begin.")
            st.caption("This demo is intended for research purposes only.")
            return

        padded_patch = pad_patch(patch)
    except Exception as error:
        st.error(str(error))
        st.caption("This demo is intended for research purposes only.")
        return

    with st.expander("Patch Information", expanded=False):
        st.write("Selected patch source:", patch_source)
        st.write("Patch shape:", tuple(patch.shape))
        st.write("Data type:", str(patch.dtype))
        st.write("Minimum intensity:", float(np.min(patch)))
        st.write("Maximum intensity:", float(np.max(patch)))
        st.write("Mean intensity:", float(np.mean(patch)))

    preview_column, prediction_column = st.columns(2)

    with preview_column:
        st.subheader("Image Preview")
        st.pyplot(plot_middle_axial_slice(padded_patch))
        st.caption("Middle axial slice")

    with prediction_column:
        st.subheader("Prediction")

        if st.button("Run Prediction"):
            try:
                start_time = time.perf_counter()
                prediction = run_inference(model_name, checkpoint_path, padded_patch)
                inference_time_ms = (time.perf_counter() - start_time) * 1000.0
            except Exception as error:
                st.error(str(error))
                st.caption("This demo is intended for research purposes only.")
                return

            if prediction["predicted_class"] == 0:
                st.success("🟢 Benign")
            else:
                st.error("🔴 Suspicious nodule")

            st.metric("Confidence", f"{prediction['confidence'] * 100.0:.2f} %")
            st.metric(
                "Positive probability",
                f"{prediction['positive_probability'] * 100.0:.2f} %",
            )
            st.metric("Inference time (milliseconds)", f"{inference_time_ms:.2f}")

            with st.expander("Prediction JSON"):
                st.json(prediction)
        else:
            st.info("Run prediction to view results.")

    st.caption("This demo is intended for research purposes only.")


if __name__ == "__main__":
    main()
