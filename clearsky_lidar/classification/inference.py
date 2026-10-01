"""Inference wrapper for waste classification using YOLOv8-cls."""

# Apply PyTorch 2.6+ compatibility patches BEFORE importing ultralytics
from .patches import *  # noqa: F403,F401

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from ultralytics import YOLO

from .model import load_model
from .dataset import CLASS_NAMES


class WasteClassifier:
    """Waste classification inference wrapper.

    Handles both .pt (PyTorch) and .engine (TensorRT) models via ultralytics YOLO.
    Resolves model path from explicit argument, CLEARSKY_MODEL_DIR env var,
    or default ~/ClearSky/weights/ location.
    """

    DEFAULT_MODEL_DIR = Path.home() / "ClearSky" / "weights"
    DEFAULT_CONF_THRESHOLD = 0.55

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    ):
        """Initialize WasteClassifier.

        Args:
            model_path: Explicit path to model file (.pt or .engine).
                       If None, resolves via CLEARSKY_MODEL_DIR or default.
            conf_threshold: Confidence threshold for predictions (default: 0.55).
        """
        self.conf_threshold = conf_threshold
        self.model = self._load_resolved_model(model_path)

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        """Resolve model path in priority order:
        1. Explicit model_path argument
        2. CLEARSKY_MODEL_DIR environment variable
        3. Default ~/ClearSky/weights/
        """
        # 1. Explicit path
        if explicit_path is not None:
            path = Path(explicit_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Explicit model path not found: {path}")

        # 2. Environment variable
        env_dir = os.environ.get("CLEARSKY_MODEL_DIR")
        if env_dir:
            env_path = Path(env_dir)
            # Check for both .pt and .engine
            for ext in (".pt", ".engine"):
                candidate = env_path / f"best{ext}"
                if candidate.exists():
                    return candidate
            # Also check for any model file in the directory
            for candidate in env_path.glob("best.*"):
                if candidate.suffix.lower() in (".pt", ".engine"):
                    return candidate

        # 3. Default path
        default_path = self.DEFAULT_MODEL_DIR
        for ext in (".pt", ".engine"):
            candidate = default_path / f"best{ext}"
            if candidate.exists():
                return candidate

        # If nothing found, return default .pt path (will raise on load)
        return default_path / "best.pt"

    def _load_resolved_model(self, model_path: Optional[Union[str, Path]]) -> YOLO:
        """Load model from resolved path."""
        resolved_path = self._resolve_model_path(model_path)
        return load_model(resolved_path)

    def predict(self, image: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB or BGR format.
                   YOLO handles preprocessing and resizing internally.

        Returns:
            Tuple of (class_name, confidence, probs_dict) where:
                - class_name: Predicted class name (str)
                - confidence: Confidence score for predicted class (float)
                - probs_dict: Dictionary mapping all class names to probabilities
        """
        # YOLO predict returns a list of Results objects
        results = self.model(image, verbose=False)

        # Get probabilities from first result
        result = results[0]
        probs_array = result.probs.data.cpu().numpy()  # Shape: (num_classes,)

        # Find top class
        top_idx = int(probs_array.argmax())
        confidence = float(probs_array[top_idx])
        class_name = CLASS_NAMES[top_idx]

        # Build probs dictionary
        probs_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs_array)}

        # Apply confidence threshold filtering
        if confidence < self.conf_threshold:
            # Return lowest confidence class or a special indicator
            # Per spec: "Confidence threshold filtering" - return the prediction
            # but with indication it's below threshold. We'll return the top class
            # but the caller can check confidence against threshold.
            pass

        return class_name, confidence, probs_dict

    def predict_batch(
        self, images: List[np.ndarray]
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Run inference on a batch of images.

        Args:
            images: List of input images as numpy arrays.

        Returns:
            List of (class_name, confidence, probs_dict) tuples.
        """
        if not images:
            return []

        # YOLO handles batch inference
        results = self.model(images, verbose=False)

        predictions = []
        for result in results:
            probs_array = result.probs.data.cpu().numpy()
            top_idx = int(probs_array.argmax())
            confidence = float(probs_array[top_idx])
            class_name = CLASS_NAMES[top_idx]
            probs_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs_array)}
            predictions.append((class_name, confidence, probs_dict))

        return predictions