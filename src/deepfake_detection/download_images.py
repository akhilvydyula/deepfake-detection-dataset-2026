from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from deepfake_detection.config import ensure_dir, load_config, resolve_path
from deepfake_detection.data import normalize_splits, read_metadata


def _label_name(row: pd.Series) -> str:
    if "label" in row and isinstance(row["label"], str):
        return row["label"].lower()
    return "real" if int(row["label_numeric"]) == 1 else "fake"


def _extension_from_content_type(content_type: str | None) -> str:
    if content_type and "png" in content_type.lower():
        return ".png"
    return ".jpg"


def _download_one(
    row: pd.Series,
    images_dir: Path,
    timeout_seconds: int,
    image_size: int,
    split_column: str,
    url_column: str,
    id_column: str,
) -> dict[str, Any]:
    split_name = str(row[split_column]).lower().replace("valid", "val")
    label_name = _label_name(row)
    image_id = str(row[id_column])

    try:
        response = requests.get(row[url_column], timeout=timeout_seconds)
        response.raise_for_status()
        extension = _extension_from_content_type(response.headers.get("content-type"))
        output_dir = images_dir / split_name / label_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_id}{extension}"

        if not output_path.exists():
            output_path.write_bytes(response.content)
            with Image.open(output_path) as image:
                image.convert("RGB").resize((image_size, image_size)).save(output_path)

        return {
            id_column: row[id_column],
            "local_path": str(output_path),
            "download_ok": True,
            "error": "",
        }
    except (requests.RequestException, OSError, UnidentifiedImageError) as exc:
        return {id_column: row[id_column], "local_path": "", "download_ok": False, "error": str(exc)}


def download_images(config_path: str = "configs/default.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    download_cfg = config["download"]

    df = read_metadata(dataset_cfg["csv_path"])
    df = normalize_splits(
        df,
        label_column=dataset_cfg["label_column"],
        split_column=dataset_cfg["split_column"],
        seed=config["training"]["seed"],
    )
    limit = download_cfg.get("limit")
    if limit:
        df = df.head(int(limit)).copy()

    images_dir = ensure_dir(dataset_cfg["images_dir"])
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=int(download_cfg["max_workers"])) as executor:
        futures = [
            executor.submit(
                _download_one,
                row,
                images_dir,
                int(download_cfg["timeout_seconds"]),
                int(download_cfg["image_size"]),
                dataset_cfg["split_column"],
                dataset_cfg["url_column"],
                dataset_cfg["id_column"],
            )
            for _, row in df.iterrows()
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            results.append(future.result())

    result_df = pd.DataFrame(results)
    merged = df.merge(result_df, on=dataset_cfg["id_column"], how="left")
    metadata_path = resolve_path(dataset_cfg["metadata_path"])
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(metadata_path, index=False)

    failures = int((~merged["download_ok"]).sum())
    print(f"Saved metadata to {metadata_path}")
    print(f"Downloaded {len(merged) - failures}/{len(merged)} images successfully")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images listed in FINAL_DATASET.csv.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_images(args.config)
