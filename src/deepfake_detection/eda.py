from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from deepfake_detection.config import ensure_dir, load_config
from deepfake_detection.data import normalize_splits, read_metadata


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).items()}


def run_eda(config_path: str = "configs/default.yaml") -> dict[str, object]:
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    reports_dir = ensure_dir(config["reports"]["output_dir"])

    df = read_metadata(dataset_cfg["csv_path"])
    df = normalize_splits(
        df,
        label_column=dataset_cfg["label_column"],
        split_column=dataset_cfg["split_column"],
        seed=config["training"]["seed"],
    )

    summary = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "labels": _value_counts(df, "label"),
        "label_numeric": _value_counts(df, dataset_cfg["label_column"]),
        "splits": _value_counts(df, dataset_cfg["split_column"]),
        "gender": _value_counts(df, "gender"),
        "age_group": _value_counts(df, "age_group"),
        "image_quality": _value_counts(df, "image_quality"),
        "source": _value_counts(df, "source"),
        "fake_method": _value_counts(df, "fake_method"),
    }

    summary_path = reports_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_columns = ["label", dataset_cfg["split_column"], "gender", "age_group", "image_quality"]
    for column in plot_columns:
        if column not in df.columns:
            continue
        ax = df[column].value_counts(dropna=False).plot(kind="bar", title=column)
        ax.set_xlabel(column)
        ax.set_ylabel("count")
        plt.tight_layout()
        plt.savefig(reports_dir / f"{column}_distribution.png")
        plt.close()

    print(f"Saved EDA outputs to {reports_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset summary and distribution plots.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda(args.config)
