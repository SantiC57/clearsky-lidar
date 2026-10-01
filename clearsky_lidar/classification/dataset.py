"""Roboflow Waste Classification dataset loader for YOLOv8-cls."""

import csv
from pathlib import Path
from typing import Tuple, List, Optional, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


# Waste class names in fixed order (matching Roboflow export)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


class WasteClassificationDataset(Dataset):
    """Dataset for Roboflow classification format with _classes.csv labels.

    Roboflow classification export structure:
    - split/ (train/valid/test) folders contain images directly
    - split/_classes.csv contains one-hot encoded labels
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        transform: Optional[Callable] = None,
    ):
        """Initialize dataset.

        Args:
            root: Root directory containing train/valid/test folders
            split: One of "train", "valid", "test"
            transform: Albumentations transform pipeline
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.split_dir = self.root / split
        self.csv_path = self.split_dir / "_classes.csv"

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Classes CSV not found: {self.csv_path}")

        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load image filenames and labels from _classes.csv."""
        self.samples: List[Tuple[Path, int]] = []

        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # filename, cardboard, glass, metal, paper, plastic, trash

            # Verify class names in header match expected (strip whitespace for robustness)
            expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
            actual_classes = [c.strip() for c in header[1:]]
            if actual_classes != expected_classes:
                raise ValueError(
                    f"CSV class columns don't match expected. Got: {actual_classes}, "
                    f"Expected: {expected_classes}"
                )

            for row in reader:
                if not row or not row[0]:
                    continue
                filename = row[0]
                one_hot = [int(x) for x in row[1:7]]
                class_idx = one_hot.index(1) if 1 in one_hot else -1

                if class_idx == -1:
                    raise ValueError(f"Invalid one-hot encoding for {filename}: {one_hot}")

                img_path = self.split_dir / filename
                if img_path.exists():
                    self.samples.append((img_path, class_idx))
                else:
                    # Skip missing images with warning
                    pass  # In production, might want to log this

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image as RGB numpy array
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(image=image_np)
            image_np = transformed["image"]

        return image_np, label


def get_train_transform(img_size: int = 640) -> A.Compose:
    """Get training transform with augmentation."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transform(img_size: int = 640) -> A.Compose:
    """Get validation/test transform (no augmentation)."""
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_dataloaders(
    root: Union[str, Path],
    batch_size: int = 16,
    workers: int = 4,
    cache: bool = True,
    img_size: int = 640,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for Roboflow classification dataset.

    Args:
        root: Root directory containing train/valid/test folders with _classes.csv
        batch_size: Batch size for all loaders
        workers: Number of DataLoader workers
        cache: Whether to use pinned memory and persistent workers
        img_size: Input image size (default: 640 for YOLOv8)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    root = Path(root)

    train_dataset = WasteClassificationDataset(
        root=root,
        split="train",
        transform=get_train_transform(img_size),
    )

    val_dataset = WasteClassificationDataset(
        root=root,
        split="valid",
        transform=get_valid_transform(img_size),
    )

    test_dataset = WasteClassificationDataset(
        root=root,
        split="test",
        transform=get_valid_transform(img_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=cache,
        persistent_workers=cache and workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=cache,
        persistent_workers=cache and workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=cache,
        persistent_workers=cache and workers > 0,
    )

    return train_loader, val_loader, test_loader


def verify_dataset_lengths(root: Union[str, Path]) -> dict:
    """Verify dataset splits have expected number of samples.

    Args:
        root: Root directory of dataset

    Returns:
        Dictionary with split names and sample counts
    """
    root = Path(root)
    counts = {}

    for split in ["train", "valid", "test"]:
        csv_path = root / split / "_classes.csv"
        if csv_path.exists():
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                counts[split] = sum(1 for row in reader if row and row[0])
        else:
            counts[split] = 0

    return counts