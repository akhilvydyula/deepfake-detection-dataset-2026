from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.config import ensure_dir, load_config, resolve_path
from deepfake_detection.data import DeepfakeImageDataset, make_transforms
from deepfake_detection.model import build_resnet18, choose_device
from deepfake_detection.train import _metrics


def evaluate(config_path: str = "configs/default.yaml", split: str = "test") -> dict[str, object]:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    device = choose_device(str(training_cfg["device"]))

    metadata = pd.read_csv(resolve_path(dataset_cfg["metadata_path"]))
    metadata = metadata[metadata["download_ok"] == True].copy()  # noqa: E712
    split_df = metadata[metadata[dataset_cfg["split_column"]].str.lower() == split.lower()]
    if split_df.empty:
        raise ValueError(f"No rows found for split '{split}'.")

    dataset = DeepfakeImageDataset(
        split_df,
        label_column=dataset_cfg["label_column"],
        transform=make_transforms(image_size=int(config["download"]["image_size"]), train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
    )

    model = build_resnet18(pretrained=False).to(device)
    checkpoint_path = resolve_path(training_cfg["output_dir"]) / training_cfg["best_model_name"]
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    targets: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating {split}"):
            logits = model(images.to(device)).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            targets.extend(labels.tolist())

    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    report = {
        "split": split,
        "metrics": _metrics(targets, probabilities),
        "classification_report": classification_report(
            targets,
            predictions,
            target_names=["FAKE", "REAL"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions).tolist(),
    }

    reports_dir = ensure_dir(config["reports"]["output_dir"])
    output_path = reports_dir / f"{split}_metrics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved metrics to {output_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained deepfake classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.split)
