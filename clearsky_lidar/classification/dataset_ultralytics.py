"""Convert Roboflow classification dataset to Ultralytics format."""

import csv
import os
from pathlib import Path
from typing import Optional

from .dataset import CLASS_NAMES, WasteClassificationDataset


def convert_to_ultralytics_format(
    source_root: Path,
    target_root: Path,
    splits: Optional[list] = None,
    use_symlinks: bool = True,
) -> Path:
    """Convert Roboflow format dataset to Ultralytics classification format.

    Roboflow format:
        source_root/
            train/
                img1.jpg
                img2.jpg
                ...
                _classes.csv
            valid/
                ...
            test/
                ...

    Ultralytics format:
        target_root/
            train/
                class1/
                    img1.jpg -> ../../../source_root/train/img1.jpg
                class2/
                    ...
            val/
                class1/
                    ...
            test/
                class1/
                    ...

    Args:
        source_root: Source dataset root (Roboflow format)
        target_root: Target directory for Ultralytics format
        splits: List of splits to convert (default: ["train", "valid", "test"])
        use_symlinks: Whether to use symlinks (True) or copy files (False)

    Returns:
        Path to target_root
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    # Ultralytics uses "val" not "valid"
    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test",
    }

    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for source_split in splits:
        if source_split not in split_mapping:
            continue

        target_split = split_mapping[source_split]
        source_split_dir = source_root / source_split
        target_split_dir = target_root / target_split

        if not source_split_dir.exists():
            print(f"Warning: Split {source_split} not found at {source_split_dir}")
            continue

        # Load dataset to get samples
        try:
            dataset = WasteClassificationDataset(
                root=source_root,
                split=source_split,
                transform=None,
            )
        except FileNotFoundError as e:
            print(f"Warning: Could not load {source_split}: {e}")
            continue

        # Create class directories
        for class_name in CLASS_NAMES:
            (target_split_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Create symlinks or copy files
        for img_path, class_idx in dataset.samples:
            class_name = CLASS_NAMES[class_idx]
            target_file = target_split_dir / class_name / img_path.name

            if target_file.exists():
                continue

            try:
                if use_symlinks:
                    # Use relative symlink
                    rel_path = os.path.relpath(img_path, target_file.parent)
                    target_file.symlink_to(rel_path)
                else:
                    import shutil
                    shutil.copy2(img_path, target_file)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not link {img_path} to {target_file}: {e}")

    return target_root


def create_ultralytics_data_yaml(dataset_root: Path, output_path: Path) -> Path:
    """Create data.yaml for Ultralytics classification dataset.

    For classification, Ultralytics expects the data argument to be a directory,
    but we can also create a yaml file for reference.
    """
    data_dict = {
        "train": str(dataset_root / "train"),
        "val": str(dataset_root / "val"),
        "test": str(dataset_root / "test"),
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    with open(output_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    return output_path