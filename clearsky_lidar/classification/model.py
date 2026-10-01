"""YOLOv8 classification model utilities for waste classification."""

# Apply PyTorch 2.6+ compatibility patches BEFORE importing ultralytics
from .patches import *  # noqa: F403,F401

from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO


def create_model(num_classes: int = 6, pretrained: bool = True, freeze_backbone: bool = True) -> YOLO:
    """Create YOLOv8n-cls model with custom classification head.

    Args:
        num_classes: Number of output classes (default: 6 for waste classes)
        pretrained: Whether to use pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone layers initially (default: True)

    Returns:
        YOLO model configured for classification with num_classes output
    """
    model = YOLO("yolov8n-cls.pt" if pretrained else "yolov8n-cls.yaml")

    # Replace classification head for the specified number of classes
    # Ultralytics handles this automatically when training with data.yaml
    # that specifies the correct number of classes

    if freeze_backbone:
        freeze_backbone_fn(model, freeze=True)

    return model


def load_model(path: Union[str, Path]) -> YOLO:
    """Load a YOLO model from .pt (PyTorch) or .engine (TensorRT) file.

    Args:
        path: Path to model file (.pt or .engine)

    Returns:
        Loaded YOLO model

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If file extension is not supported
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".pt", ".engine"):
        raise ValueError(f"Unsupported model format: {suffix}. Expected .pt or .engine")

    # Note: File existence is checked by YOLO internally when loading
    # We only validate the extension here
    # Safe globals are registered at module level via patches
    return YOLO(str(path))


def freeze_backbone_fn(model: YOLO, freeze: bool = True) -> None:
    """Freeze or unfreeze backbone layers (layers 0-7 in YOLOv8n-cls).

    Args:
        model: YOLO model instance
        freeze: True to freeze, False to unfreeze
    """
    # YOLOv8n-cls backbone is typically the first 8 layers (indices 0-7)
    # Access the underlying torch model
    for layer in model.model.model[:8]:
        for param in layer.parameters():
            param.requires_grad = not freeze


def unfreeze_head(model: YOLO) -> None:
    """Unfreeze classification head (layers 8+ in YOLOv8n-cls).

    Args:
        model: YOLO model instance
    """
    for layer in model.model.model[8:]:
        for param in layer.parameters():
            param.requires_grad = True


def save_model(model: YOLO, path: Union[str, Path]) -> Path:
    """Save model weights to file.

    Args:
        model: YOLO model instance
        path: Output path for saved model

    Returns:
        Path to saved model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return path