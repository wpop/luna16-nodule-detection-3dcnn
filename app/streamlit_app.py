"""
Streamlit demo app for single 3D patch inference.
"""

import json
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


def find_latest_experiment(project_root: Path) -> Path | None:
    """
    Return the latest experiment directory if one exists.
    """

    experiments_dir = project_root / "outputs" / "experiments"

    if not experiments_dir.exists():
        return None

    experiment_dirs = [
        path
        for path in experiments_dir.iterdir()
        if path.is_dir()
    ]

    if not experiment_dirs:
        return None

    return max(experiment_dirs, key=lambda path: path.name)


def training_history_epoch_count(history_path: Path) -> int:
    """
    Return the number of epochs stored in a training history JSON file.
    """

    with history_path.open("r", encoding="utf-8") as input_file:
        history = json.load(input_file)

    return len(history.get("train_loss", []))


def render_prediction_tab(
    model_name: str,
    checkpoint_path: Path,
) -> None:
    """
    Render the single-patch prediction interface.
    """

    patch_source = st.selectbox("Patch source", PATCH_SOURCES)
    uploaded_file = None

    if patch_source == "Upload custom .npy patch":
        uploaded_file = st.file_uploader("Upload .npy patch", type=["npy"])

    if patch_source == "Upload custom .npy patch" and uploaded_file is None:
        st.info("Upload a 3D NumPy patch to begin.")
        return

    try:
        patch = load_selected_patch(patch_source, uploaded_file)
        if patch is None:
            st.info("Upload a 3D NumPy patch to begin.")
            return

        padded_patch = pad_patch(patch)
    except Exception as error:
        st.error(str(error))
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
                return

            device_name = "CUDA" if torch.cuda.is_available() else "CPU"
            system_information = [
                "**System Information**",
                "",
                f"**Model:** {model_name}",
                "",
                f"**Checkpoint:** {checkpoint_path.name}",
                "",
                f"**Device:** {device_name}",
            ]

            if torch.cuda.is_available():
                system_information.extend(
                    ["", f"**GPU:** {torch.cuda.get_device_name(0)}"]
                )

            st.info("\n".join(system_information))

            st.subheader("Prediction Result")

            if prediction["predicted_class"] == 0:
                st.success("🟢 Benign Nodule")
                st.caption(
                    "The model predicts that this CT patch is more likely to "
                    "represent a benign finding."
                )
            else:
                st.error("🔴 Suspicious Nodule")
                st.caption(
                    "The model predicts that this CT patch may contain a "
                    "pulmonary nodule requiring further clinical review."
                )

            st.warning(
                "This prediction is intended for research purposes only and "
                "must not be used as a clinical diagnosis."
            )

            st.subheader("Confidence")
            st.progress(float(prediction["confidence"]))
            st.caption(f"{prediction['confidence'] * 100.0:.2f} %")

            st.subheader("Positive probability")
            st.progress(float(prediction["positive_probability"]))
            st.caption(f"{prediction['positive_probability'] * 100.0:.2f} %")

            st.metric("Inference time", f"{inference_time_ms:.2f} ms")

            with st.expander("Prediction JSON"):
                st.json(prediction)
        else:
            st.info("Run prediction to view results.")


def render_evaluation_tab(project_root: Path) -> None:
    """
    Render evaluation artifacts from the latest experiment.
    """

    latest_experiment = find_latest_experiment(project_root)

    if latest_experiment is None:
        st.info("No evaluation results available.")
        return

    st.write("Latest experiment:", latest_experiment.name)

    displayed_any = False

    for figure_path in [
        latest_experiment / "figures" / "confusion_matrix.png",
        latest_experiment / "figures" / "roc_curve.png",
    ]:
        if figure_path.exists():
            st.image(str(figure_path), caption=figure_path.name)
            displayed_any = True

    history_path = latest_experiment / "metrics" / "training_history.json"
    history_figure_path = latest_experiment / "figures" / "training_history.png"

    if history_path.exists():
        epoch_count = training_history_epoch_count(history_path)

        if epoch_count > 1 and history_figure_path.exists():
            st.image(str(history_figure_path), caption=history_figure_path.name)
            displayed_any = True
        elif epoch_count == 1:
            st.info(
                "Training history plot is skipped because this run contains only one epoch."
            )
            displayed_any = True

    if not displayed_any:
        st.info("No evaluation results available.")


def render_about_tab() -> None:
    """
    Render project summary information.
    """

    st.write("This is a research demo for LUNA16 lung nodule classification.")
    st.write("It supports 3D CNNs and a 3D Vision Transformer.")
    st.write(
        "It includes training, validation, benchmark export, TensorBoard, ROC/AUC, "
        "confusion matrix, ONNX export, and inference."
    )


def main() -> None:
    """
    Run the Streamlit app.
    """

    project_root = Path(__file__).resolve().parents[1]

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

    prediction_tab, evaluation_tab, about_tab = st.tabs(
        ["Prediction", "Evaluation", "About"]
    )

    with prediction_tab:
        render_prediction_tab(model_name, checkpoint_path)

    with evaluation_tab:
        render_evaluation_tab(project_root)

    with about_tab:
        render_about_tab()

    st.caption("This demo is intended for research purposes only.")


if __name__ == "__main__":
    main()
