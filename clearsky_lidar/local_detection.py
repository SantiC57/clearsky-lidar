"""Local waste detection using YOLOv8 model.

Provides WasteDetectorLocal for local inference using a trained YOLOv8 model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


class WasteDetectorLocal:
    """Local waste detector using YOLOv8 model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        data_yaml_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
        device: str | None = None,
    ) -> None:
        """Initialize local waste detector.

        Args:
            model_path: Path to YOLOv8 model file (.pt or .onnx).
                       If None, uses models/best.pt from package directory.
            data_yaml_path: Path to data.yaml with class names.
                          If None, uses models/data.yaml from package directory.
            confidence_threshold: Minimum confidence for detections (default: 0.5).
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto).
        """
        package_dir = Path(__file__).parent.parent
        
        if model_path is None:
            # Default to models/best.pt relative to this file
            model_path = package_dir / "models" / "best.pt"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Load data.yaml for class names
        if data_yaml_path is None:
            data_yaml_path = package_dir / "models" / "data.yaml"
        
        self.data_yaml_path = Path(data_yaml_path)
        self.classes = self._load_class_names()

        self.confidence_threshold = confidence_threshold
        self.device = device

        # Load model
        # Handle PyTorch 2.6+ compatibility for older model files
        import torch
        try:
            self.model = YOLO(str(self.model_path), task='detect')
        except Exception as e:
            if "Weights only load failed" in str(e) and self.model_path.suffix == '.pt':
                # PyTorch 2.6+ requires weights_only=False for older models
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Load with weights_only=False for compatibility
                    original_load = torch.load
                    torch.load = lambda *args, **kwargs: original_load(*args, weights_only=False, **{k:v for k,v in kwargs.items() if k != 'weights_only'})
                    try:
                        self.model = YOLO(str(self.model_path), task='detect')
                    finally:
                        torch.load = original_load
            else:
                raise

    def _load_class_names(self) -> list[str]:
        """Load class names from data.yaml file.

        Returns:
            List of class names.
        """
        if not self.data_yaml_path.exists():
            # Default classes if data.yaml not found
            return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

        with open(self.data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Extract class names from data.yaml
        if 'names' in data:
            # Format: names: {0: cardboard, 1: glass, ...}
            names_dict = data['names']
            if isinstance(names_dict, dict):
                # Sort by key to ensure correct order
                return [names_dict[i] for i in sorted(names_dict.keys())]
            elif isinstance(names_dict, list):
                return names_dict

        # Fallback to default classes
        return ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run inference on an image.

        Args:
            image: BGR image as numpy array (OpenCV format).

        Returns:
            Dictionary with detection results:
            {
                'boxes': list of [x1, y1, x2, y2] coordinates,
                'confidences': list of confidence scores,
                'class_ids': list of class indices,
                'class_names': list of class names,
                'count': total number of detections
            }
        """
        # Run inference
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )

        # Extract results
        detections = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'class_names': [],
            'count': 0
        }

        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                detections['boxes'] = boxes.tolist()
                detections['confidences'] = confidences.tolist()
                detections['class_ids'] = class_ids.tolist()
                detections['class_names'] = [self.classes[cid] for cid in class_ids]
                detections['count'] = len(boxes)

        return detections

    def predict_from_file(self, image_path: str | Path) -> dict[str, Any]:
        """Run inference on an image file.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary with detection results.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict(image)

    def draw_detections(
        self,
        image: np.ndarray,
        detections: dict[str, Any],
        show_labels: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """Draw detection boxes on image.

        Args:
            image: BGR image as numpy array.
            detections: Detection results from predict().
            show_labels: Whether to show class labels.
            show_confidence: Whether to show confidence scores.

        Returns:
            Image with detections drawn.
        """
        annotated = image.copy()

        for i in range(detections['count']):
            box = detections['boxes'][i]
            confidence = detections['confidences'][i]
            class_name = detections['class_names'][i]

            x1, y1, x2, y2 = map(int, box)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label
            if show_labels and show_confidence:
                label = f"{class_name}: {confidence:.2f}"
            elif show_labels:
                label = class_name
            elif show_confidence:
                label = f"{confidence:.2f}"
            else:
                label = ""

            if label:
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )
                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        return annotated
