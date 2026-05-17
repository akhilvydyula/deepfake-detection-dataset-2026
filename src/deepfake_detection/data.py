from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from deepfake_detection.config import resolve_path


VALID_SPLITS = {"train", "val", "valid", "validation", "test"}


def read_metadata(csv_path: str | Path) -> pd.DataFrame:
    path = resolve_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the Kaggle dataset first or update configs/default.yaml."
        )
    return pd.read_csv(path)


def normalize_splits(
    df: pd.DataFrame,
    label_column: str = "label_numeric",
    split_column: str = "split",
    seed: int = 42,
) -> pd.DataFrame:
    """Use dataset-provided splits when present, otherwise create 70/10/20 splits."""
    df = df.copy()
    if split_column in df.columns:
        df[split_column] = df[split_column].astype(str).str.lower().replace({"valid": "val"})
        known_mask = df[split_column].isin(VALID_SPLITS)
        if known_mask.all():
            return df

    train_df, holdout_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df[label_column],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=2 / 3,
        stratify=holdout_df[label_column],
        random_state=seed,
    )

    df[split_column] = "train"
    df.loc[val_df.index, split_column] = "val"
    df.loc[test_df.index, split_column] = "test"
    return df


def make_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    steps: list[torch.nn.Module] = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(steps)


class DeepfakeImageDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        label_column: str = "label_numeric",
        path_column: str = "local_path",
        transform: transforms.Compose | None = None,
    ) -> None:
        missing = {label_column, path_column} - set(metadata.columns)
        if missing:
            raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

        self.metadata = metadata.reset_index(drop=True)
        self.label_column = label_column
        self.path_column = path_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        image_path = resolve_path(row[self.path_column])
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(float(row[self.label_column]), dtype=torch.float32)
        return image, label
