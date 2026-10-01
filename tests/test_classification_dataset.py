"""Tests for classification dataset module."""

import pytest
import csv
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import torch
import albumentations as A

from clearsky_lidar.classification.dataset import (
    WasteClassificationDataset,
    get_train_transform,
    get_valid_transform,
    get_dataloaders,
    verify_dataset_lengths,
    CLASS_NAMES,
)


class TestWasteClassificationDataset:
    """Tests for WasteClassificationDataset class."""

    def setup_method(self):
        """Create a temporary dataset structure for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.split_dir = self.temp_dir / "train"
        self.split_dir.mkdir()

        # Create _classes.csv with one-hot labels
        self.csv_path = self.split_dir / "_classes.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "cardboard", "glass", "metal", "paper", "plastic", "trash"])
            writer.writerow(["img1.jpg", "1", "0", "0", "0", "0", "0"])  # cardboard
            writer.writerow(["img2.jpg", "0", "1", "0", "0", "0", "0"])  # glass
            writer.writerow(["img3.jpg", "0", "0", "1", "0", "0", "0"])  # metal
            writer.writerow(["img4.jpg", "0", "0", "0", "1", "0", "0"])  # paper
            writer.writerow(["img5.jpg", "0", "0", "0", "0", "1", "0"])  # plastic
            writer.writerow(["img6.jpg", "0", "0", "0", "0", "0", "1"])  # trash

        # Create dummy images
        for i in range(1, 7):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(self.split_dir / f"img{i}.jpg")

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_init_loads_samples(self):
        """Test dataset initialization loads all samples."""
        dataset = WasteClassificationDataset(self.temp_dir, "train", transform=None)

        assert len(dataset) == 6
        assert len(dataset.samples) == 6

    def test_init_verifies_class_columns(self):
        """Test that CSV class columns are verified."""
        # Create CSV with wrong class order
        bad_split = self.temp_dir / "bad"
        bad_split.mkdir()
        bad_csv = bad_split / "_classes.csv"
        with open(bad_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "glass", "cardboard", "metal", "paper", "plastic", "trash"])
            writer.writerow(["img1.jpg", "1", "0", "0", "0", "0", "0"])

        (bad_split / "img1.jpg").touch()

        with pytest.raises(ValueError, match="CSV class columns don't match expected"):
            WasteClassificationDataset(self.temp_dir, "bad", transform=None)

    def test_init_raises_for_missing_split_dir(self):
        """Test FileNotFoundError for missing split directory."""
        with pytest.raises(FileNotFoundError, match="Split directory not found"):
            WasteClassificationDataset(self.temp_dir, "nonexistent", transform=None)

    def test_init_raises_for_missing_csv(self):
        """Test FileNotFoundError for missing _classes.csv."""
        missing_split = self.temp_dir / "missing"
        missing_split.mkdir()
        # No _classes.csv

        with pytest.raises(FileNotFoundError, match="Classes CSV not found"):
            WasteClassificationDataset(self.temp_dir, "missing", transform=None)

    def test_getitem_returns_tensor_and_label(self):
        """Test __getitem__ returns (tensor, label) tuple."""
        dataset = WasteClassificationDataset(self.temp_dir, "train", transform=get_valid_transform(64))

        img, label = dataset[0]

        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 64, 64)  # C, H, W
        assert isinstance(label, int)
        assert 0 <= label <= 5

    def test_getitem_with_transform(self):
        """Test that transform is applied."""
        transform = get_valid_transform(128)
        dataset = WasteClassificationDataset(self.temp_dir, "train", transform=transform)

        img, _ = dataset[0]

        assert img.shape == (3, 128, 128)

    def test_labels_match_class_names(self):
        """Test that labels correspond to correct class names."""
        dataset = WasteClassificationDataset(self.temp_dir, "train", transform=None)

        # Check each sample has correct label
        expected_labels = [0, 1, 2, 3, 4, 5]  # cardboard, glass, metal, paper, plastic, trash
        for i, (_, label) in enumerate(dataset.samples):
            assert label == expected_labels[i]


class TestTransforms:
    """Tests for transform functions."""

    def test_get_train_transform(self):
        """Test training transform creation."""
        transform = get_train_transform(640)
        assert isinstance(transform, A.Compose)

    def test_get_valid_transform(self):
        """Test validation transform creation."""
        transform = get_valid_transform(640)
        assert isinstance(transform, A.Compose)

    def test_train_transform_augments(self):
        """Test that train transform includes augmentation."""
        transform = get_train_transform(100)
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Apply multiple times - should get different results due to augmentation
        results = [transform(image=img)["image"].numpy().tobytes() for _ in range(5)]

        # At least some should be different (with high probability)
        # Note: this is probabilistic, but with 5 runs it's very likely
        unique_results = len(set(results))
        assert unique_results > 1  # Should have variation

    def test_valid_transform_deterministic(self):
        """Test that valid transform is deterministic (no augmentation)."""
        transform = get_valid_transform(100)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        results = [transform(image=img)["image"].numpy().tobytes() for _ in range(5)]

        # All should be identical
        unique_results = len(set(results))
        assert unique_results == 1


class TestGetDataloaders:
    """Tests for get_dataloaders function."""

    def setup_method(self):
        """Create temporary dataset with all splits."""
        self.temp_dir = Path(tempfile.mkdtemp())

        for split in ["train", "valid", "test"]:
            split_dir = self.temp_dir / split
            split_dir.mkdir()

            csv_path = split_dir / "_classes.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "cardboard", "glass", "metal", "paper", "plastic", "trash"])
                for i in range(4):  # 4 samples per split
                    row = ["0", "0", "0", "0", "0", "0"]
                    row[i % 6] = "1"
                    writer.writerow([f"img{i}.jpg"] + row)

            # Create dummy images
            for i in range(4):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                from PIL import Image
                Image.fromarray(img).save(split_dir / f"img{i}.jpg")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_get_dataloaders_returns_three_loaders(self):
        """Test that get_dataloaders returns 3 DataLoaders."""
        train_loader, val_loader, test_loader = get_dataloaders(
            self.temp_dir, batch_size=2, workers=0, cache=False, img_size=64
        )

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert isinstance(test_loader, torch.utils.data.DataLoader)

    def test_dataloader_lengths_match_samples(self):
        """Test that DataLoader lengths match CSV rows."""
        train_loader, val_loader, test_loader = get_dataloaders(
            self.temp_dir, batch_size=2, workers=0, cache=False, img_size=64
        )

        # 4 samples per split, batch_size=2 -> 2 batches each
        assert len(train_loader) == 2
        assert len(val_loader) == 2
        assert len(test_loader) == 2

    def test_dataloader_batch_shape(self):
        """Test that batches have correct shape."""
        train_loader, _, _ = get_dataloaders(
            self.temp_dir, batch_size=2, workers=0, cache=False, img_size=64
        )

        batch = next(iter(train_loader))
        images, labels = batch

        assert images.shape == (2, 3, 64, 64)  # batch, channels, height, width
        assert labels.shape == (2,)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.long


class TestVerifyDatasetLengths:
    """Tests for verify_dataset_lengths function."""

    def setup_method(self):
        """Create temporary dataset with known counts."""
        self.temp_dir = Path(tempfile.mkdtemp())

        for split, count in [("train", 10), ("valid", 5), ("test", 3)]:
            split_dir = self.temp_dir / split
            split_dir.mkdir()

            csv_path = split_dir / "_classes.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["filename", "cardboard", "glass", "metal", "paper", "plastic", "trash"])
                for i in range(count):
                    row = ["0", "0", "0", "0", "0", "0"]
                    row[i % 6] = "1"
                    writer.writerow([f"img{i}.jpg"] + row)

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def test_verify_dataset_lengths(self):
        """Test that verify_dataset_lengths returns correct counts."""
        counts = verify_dataset_lengths(self.temp_dir)

        assert counts["train"] == 10
        assert counts["valid"] == 5
        assert counts["test"] == 3


class TestClassNames:
    """Test CLASS_NAMES constant."""

    def test_class_names_order(self):
        """Test CLASS_NAMES has correct order."""
        expected = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        assert CLASS_NAMES == expected

    def test_class_names_length(self):
        """Test CLASS_NAMES has 6 classes."""
        assert len(CLASS_NAMES) == 6