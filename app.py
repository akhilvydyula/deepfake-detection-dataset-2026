from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from deepfake_detection.config import load_config, resolve_path
from deepfake_detection.predict import load_trained_model, predict_image


DEFAULT_CONFIG = "configs/default.yaml"


@st.cache_resource
def get_model(config_path: str, checkpoint_path: str | None):
    return load_trained_model(config_path=config_path, checkpoint_path=checkpoint_path or None)


def checkpoint_exists(config: dict, checkpoint_path: str | None) -> tuple[bool, Path]:
    training_cfg = config["training"]
    resolved_checkpoint = (
        resolve_path(checkpoint_path)
        if checkpoint_path
        else resolve_path(training_cfg["output_dir"]) / training_cfg["best_model_name"]
    )
    return resolved_checkpoint.exists(), resolved_checkpoint


def main() -> None:
    st.set_page_config(page_title="Deepfake Detection", layout="centered")

    st.title("Deepfake Detection Runtime Test")
    st.write(
        "Upload a face image and score it with the trained model checkpoint. "
        "The output is a prototype risk signal, not a final identity decision."
    )

    with st.sidebar:
        st.header("Model Settings")
        config_path = st.text_input("Config path", DEFAULT_CONFIG)
        checkpoint_path = st.text_input(
            "Checkpoint path",
            "",
            help="Leave blank to use the checkpoint from configs/default.yaml.",
        )

    try:
        config = load_config(config_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    exists, resolved_checkpoint = checkpoint_exists(config, checkpoint_path or None)
    if not exists:
        st.warning(
            f"Checkpoint not found at `{resolved_checkpoint}`. Train the model first, then rerun this app."
        )
        st.code("python -m deepfake_detection.train --config configs/default.yaml", language="powershell")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is None:
        st.info("Upload an image to run a prediction.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model, device, loaded_config = get_model(config_path, checkpoint_path or None)
        result = predict_image(
            image,
            model,
            device,
            image_size=int(loaded_config["download"]["image_size"]),
        )
    except Exception as exc:  # pragma: no cover - surfaced in the UI for interactive use.
        st.error(f"Prediction failed: {exc}")
        st.stop()

    label = str(result["label"])
    confidence = float(result["confidence"])
    real_probability = float(result["real_probability"])
    fake_probability = float(result["fake_probability"])

    st.subheader("Prediction")
    st.metric("Model label", label, f"{confidence:.1%} confidence")

    st.progress(real_probability, text=f"REAL probability: {real_probability:.1%}")
    st.progress(fake_probability, text=f"FAKE probability: {fake_probability:.1%}")

    if label == "FAKE":
        st.warning("This image was scored as higher risk for synthetic content.")
    else:
        st.success("This image was scored as more likely real.")

    st.caption(
        "Default convention from this project: label_numeric=1 means REAL and "
        "label_numeric=0 means FAKE."
    )


if __name__ == "__main__":
    main()
