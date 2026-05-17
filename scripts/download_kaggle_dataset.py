from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from deepfake_detection.config import ensure_dir, load_config


def download_kaggle_dataset(config_path: str = "configs/default.yaml") -> Path:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    raw_dir = ensure_dir(dataset_cfg["raw_dir"])

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_cfg["kaggle_slug"], path=raw_dir, quiet=False)

    zip_path = raw_dir / f"{dataset_cfg['kaggle_slug'].split('/')[-1]}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
        print(f"Extracted {zip_path} to {raw_dir}")

    csv_path = raw_dir / "FINAL_DATASET.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Download completed, but {csv_path} was not found.")
    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the Kaggle dataset CSV.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = download_kaggle_dataset(args.config)
    print(f"Dataset ready at {path}")
