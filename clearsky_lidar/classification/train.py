"""Training script for YOLOv8 waste classification with MLflow tracking."""

# Apply PyTorch 2.6+ compatibility patches BEFORE importing ultralytics
from .patches import *  # noqa: F403,F401

import argparse
import csv
import os
from pathlib import Path
from typing import List, Optional

import mlflow
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from ultralytics import YOLO

from .dataset import CLASS_NAMES, get_dataloaders
from .dataset_ultralytics import convert_to_ultralytics_format
from .mlflow_utils import init_mlflow
from .model import create_model, freeze_backbone_fn, save_model, unfreeze_head


def create_data_yaml(
    data_root: Path,
    output_path: Path,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Create data.yaml file for Ultralytics YOLO training.

    Args:
        data_root: Root directory containing train/valid/test splits
        output_path: Path to write data.yaml
        class_names: Optional list of class names (defaults to CLASS_NAMES)

    Returns:
        Path to created data.yaml
    """
    if class_names is None:
        class_names = CLASS_NAMES

    data_dict = {
        "train": str(data_root / "train"),
        "val": str(data_root / "valid"),
        "test": str(data_root / "test"),
        "names": {i: name for i, name in enumerate(class_names)},
        "nc": len(class_names),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data_dict, f, default_flow_style=False)

    return output_path


def create_freeze_unfreeze_callback(model: YOLO, freeze_epochs: int, total_epochs: int):
    """Create a callback function for freeze/unfreeze schedule.

    Args:
        model: YOLO model instance
        freeze_epochs: Number of epochs to freeze backbone
        total_epochs: Total training epochs

    Returns:
        Callback function for on_train_epoch_start event
    """
    state = {"unfrozen": False}

    def on_train_epoch_start(trainer):
        """Called at the start of each training epoch."""
        epoch = trainer.epoch  # 0-indexed

        if epoch == 0 and freeze_epochs > 0:
            # Freeze backbone at the beginning
            freeze_backbone_fn(model, freeze=True)
            unfreeze_head(model)
            print(f"Epoch {epoch + 1}: Backbone frozen, head unfrozen")

        elif epoch == freeze_epochs and not state["unfrozen"]:
            # Unfreeze backbone after freeze_epochs
            freeze_backbone_fn(model, freeze=False)
            state["unfrozen"] = True
            print(f"Epoch {epoch + 1}: Backbone unfrozen, full fine-tuning")

    return on_train_epoch_start


def evaluate_on_validation_set(model: YOLO, data_root: Path, imgsz: int = 640) -> dict:
    """Run validation and compute detailed metrics including confusion matrix.

    Args:
        model: Trained YOLO model
        data_root: Dataset root directory
        imgsz: Image size for validation

    Returns:
        Dictionary with metrics, confusion matrix, and classification report
    """
    # Get validation dataloader
    _, val_loader, _ = get_dataloaders(
        root=data_root,
        batch_size=16,
        workers=4,
        cache=True,
        img_size=imgsz,
    )

    # Collect predictions and ground truth
    all_preds = []
    all_labels = []

    model.model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(model.device)
            outputs = model.model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        output_dict=True,
        zero_division=0,
    )

    # Per-class F1 scores
    f1_per_class = {CLASS_NAMES[i]: report[CLASS_NAMES[i]]["f1-score"] for i in range(len(CLASS_NAMES))}

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "f1_per_class": f1_per_class,
        "f1_macro": report["macro avg"]["f1-score"],
    }


def save_confusion_matrix_plot(cm: np.ndarray, output_path: Path, class_names: List[str]) -> None:
    """Save confusion matrix as heatmap image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_classification_report_txt(report: dict, output_path: Path, class_names: List[str]) -> None:
    """Save classification report as text file."""
    lines = []
    lines.append("Classification Report")
    lines.append("=" * 50)
    lines.append(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    lines.append("-" * 55)

    for name in class_names:
        r = report[name]
        lines.append(
            f"{name:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1-score']:>10.4f} {r['support']:>10.0f}"
        )

    lines.append("-" * 55)
    for avg in ["macro avg", "weighted avg"]:
        r = report[avg]
        lines.append(
            f"{avg:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1-score']:>10.4f} {r['support']:>10.0f}"
        )

    output_path.write_text("\n".join(lines))


def train_model(
    data_root: Path,
    epochs: int = 100,
    batch: int = 16,
    lr0: float = 0.002,
    patience: int = 20,
    imgsz: int = 640,
    freeze_epochs: int = 10,
    run_name: Optional[str] = None,
) -> dict:
    """Train YOLOv8 classification model with MLflow tracking.

    Args:
        data_root: Root directory of dataset (train/valid/test with _classes.csv)
        epochs: Total training epochs
        batch: Batch size
        lr0: Initial learning rate
        patience: Early stopping patience
        imgsz: Input image size
        freeze_epochs: Number of epochs to freeze backbone
        run_name: Optional MLflow run name

    Returns:
        Dictionary with training results and metrics
    """
    # Initialize MLflow
    init_mlflow()

    # Convert dataset to Ultralytics format (class subdirectories)
    # Use a temporary directory in the project folder
    ultralytics_data_root = Path.cwd() / "ultralytics_dataset"
    print(f"Converting dataset to Ultralytics format at {ultralytics_data_root}...")
    convert_to_ultralytics_format(data_root, ultralytics_data_root, use_symlinks=True)

    # Create model with frozen backbone
    model = create_model(num_classes=len(CLASS_NAMES), pretrained=True, freeze_backbone=True)

    # Setup freeze/unfreeze callback
    freeze_callback_fn = create_freeze_unfreeze_callback(model, freeze_epochs, epochs)
    model.add_callback("on_train_epoch_start", freeze_callback_fn)

    # Train with MLflow tracking
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow run: {run_id}")

        # Log hyperparameters
        params = {
            "epochs": epochs,
            "batch": batch,
            "lr0": lr0,
            "patience": patience,
            "imgsz": imgsz,
            "freeze_epochs": freeze_epochs,
            "model": "yolov8n-cls",
            "num_classes": len(CLASS_NAMES),
            "class_names": ",".join(CLASS_NAMES),
        }
        mlflow.log_params(params)

        # Train using Ultralytics API - pass the dataset directory directly
        # Note: Ultralytics handles its own epoch loop, we use callbacks for freeze/unfreeze
        results = model.train(
            data=str(ultralytics_data_root),
            epochs=epochs,
            batch=batch,
            lr0=lr0,
            patience=patience,
            imgsz=imgsz,
            device=0 if torch.cuda.is_available() else "cpu",
            workers=4,
            project="runs/classify",
            name=f"train_{run_id}",
            exist_ok=True,
            verbose=True,
        )

        # Log per-epoch metrics from results
        if hasattr(results, "results_dict"):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=epochs - 1)

        # Evaluate on validation set for detailed metrics
        print("Evaluating on validation set...")
        eval_results = evaluate_on_validation_set(model, data_root, imgsz)

        # Log validation metrics
        mlflow.log_metric("val_f1_macro", eval_results["f1_macro"])
        for class_name, f1 in eval_results["f1_per_class"].items():
            mlflow.log_metric(f"val_f1_{class_name}", f1)

        # Save and log confusion matrix
        cm_path = Path("confusion_matrix.png")
        save_confusion_matrix_plot(eval_results["confusion_matrix"], cm_path, CLASS_NAMES)
        mlflow.log_artifact(str(cm_path))

        # Save and log classification report
        report_path = Path("classification_report.txt")
        save_classification_report_txt(eval_results["classification_report"], report_path, CLASS_NAMES)
        mlflow.log_artifact(str(report_path))

        # Save best model
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        best_pt_path = models_dir / "best.pt"
        save_model(model, best_pt_path)
        mlflow.log_artifact(str(best_pt_path))

        # Also log the ultralytics dataset structure info
        import yaml
        dataset_info_path = Path("dataset_info.yaml")
        dataset_info = {
            "train": str(ultralytics_data_root / "train"),
            "val": str(ultralytics_data_root / "val"),
            "test": str(ultralytics_data_root / "test"),
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            "nc": len(CLASS_NAMES),
        }
        with open(dataset_info_path, "w") as f:
            yaml.dump(dataset_info, f, default_flow_style=False)
        mlflow.log_artifact(str(dataset_info_path))

        print(f"Training complete. Best model saved to {best_pt_path}")
        print(f"Validation F1-macro: {eval_results['f1_macro']:.4f}")

    return {
        "run_id": run_id,
        "best_model_path": str(best_pt_path),
        "val_f1_macro": eval_results["f1_macro"],
        "f1_per_class": eval_results["f1_per_class"],
        "confusion_matrix": eval_results["confusion_matrix"].tolist(),
    }


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 waste classification model")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--freeze-epochs", type=int, default=10, help="Epochs to freeze backbone")
    parser.add_argument("--run-name", type=str, default=None, help="MLflow run name")

    args = parser.parse_args()

    train_model(
        data_root=args.data_root,
        epochs=args.epochs,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        imgsz=args.imgsz,
        freeze_epochs=args.freeze_epochs,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()