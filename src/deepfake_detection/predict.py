from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from deepfake_detection.config import load_config, resolve_path
from deepfake_detection.data import make_transforms
from deepfake_detection.model import build_resnet18, choose_device


def load_trained_model(
    config_path: str | Path = "configs/default.yaml",
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, torch.device, dict[str, Any]]:
    config = load_config(config_path)
    training_cfg = config["training"]
    device = choose_device(str(training_cfg["device"]))

    model = build_resnet18(pretrained=False).to(device)
    resolved_checkpoint = (
        resolve_path(checkpoint_path)
        if checkpoint_path is not None
        else resolve_path(training_cfg["output_dir"]) / training_cfg["best_model_name"]
    )
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            f"Could not find checkpoint at {resolved_checkpoint}. Train the model first with "
            "`python -m deepfake_detection.train --config configs/default.yaml`."
        )

    checkpoint = torch.load(resolved_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, config


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
) -> dict[str, float | str]:
    transform = make_transforms(image_size=image_size, train=False)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze(1)
        real_probability = float(torch.sigmoid(logit).item())

    fake_probability = 1.0 - real_probability
    label = "REAL" if real_probability >= 0.5 else "FAKE"
    confidence = max(real_probability, fake_probability)
    return {
        "label": label,
        "confidence": confidence,
        "real_probability": real_probability,
        "fake_probability": fake_probability,
    }


def predict_image_path(
    image_path: str | Path,
    config_path: str | Path = "configs/default.yaml",
    checkpoint_path: str | Path | None = None,
) -> dict[str, float | str]:
    model, device, config = load_trained_model(config_path, checkpoint_path)
    image = Image.open(resolve_path(image_path))
    return predict_image(
        image,
        model,
        device,
        image_size=int(config["download"]["image_size"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether one face image is REAL or FAKE.")
    parser.add_argument("image", help="Path to the image to score.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Optional path to a trained checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = predict_image_path(args.image, args.config, args.checkpoint)
    print(json.dumps(result, indent=2))
