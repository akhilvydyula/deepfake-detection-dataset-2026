from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.config import ensure_dir, load_config, resolve_path
from deepfake_detection.data import DeepfakeImageDataset, make_transforms
from deepfake_detection.model import build_resnet18, choose_device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _metrics(targets: list[float], probabilities: list[float]) -> dict[str, float]:
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    metrics = {
        "accuracy": accuracy_score(targets, predictions),
        "f1": f1_score(targets, predictions),
    }
    if len(set(targets)) > 1:
        metrics["roc_auc"] = roc_auc_score(targets, probabilities)
    return {key: float(value) for key, value in metrics.items()}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    targets: list[float] = []
    probabilities: list[float] = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())

    return total_loss / len(loader.dataset), _metrics(targets, probabilities)


def train(config_path: str = "configs/default.yaml") -> dict[str, object]:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]

    seed_everything(int(training_cfg["seed"]))
    device = choose_device(str(training_cfg["device"]))

    metadata = pd.read_csv(resolve_path(dataset_cfg["metadata_path"]))
    metadata = metadata[metadata["download_ok"] == True].copy()  # noqa: E712
    split_column = dataset_cfg["split_column"]

    train_df = metadata[metadata[split_column] == "train"]
    val_df = metadata[metadata[split_column].isin(["val", "validation", "valid"])]
    if train_df.empty or val_df.empty:
        raise ValueError("Training requires non-empty train and val splits.")

    image_size = int(config["download"]["image_size"])
    train_dataset = DeepfakeImageDataset(
        train_df,
        label_column=dataset_cfg["label_column"],
        transform=make_transforms(image_size=image_size, train=True),
    )
    val_dataset = DeepfakeImageDataset(
        val_df,
        label_column=dataset_cfg["label_column"],
        transform=make_transforms(image_size=image_size, train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg["num_workers"]),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg["num_workers"]),
    )

    model = build_resnet18(pretrained=bool(training_cfg["pretrained"])).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_cfg["learning_rate"]))

    output_dir = ensure_dir(training_cfg["output_dir"])
    best_model_path = output_dir / training_cfg["best_model_name"]
    history: list[dict[str, object]] = []
    best_val_f1 = -1.0

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, indent=2))

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved best model to {best_model_path}")
    return {"history": history, "best_model_path": str(best_model_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet-18 deepfake classifier.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
