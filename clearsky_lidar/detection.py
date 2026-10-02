"""Detection and classification module for ClearSky waste classification.

Provides:
- WasteClassifier: Local inference using YOLOv8-cls model.
- WasteDetectorRemote: Remote inference using Roboflow serverless API.
- WasteDetectorLocal: Local inference using YOLOv8 detection model.
"""

from .classification.inference import WasteClassifier
from .local_detection import WasteDetectorLocal
from .remote_api import WasteDetectorRemote

__all__ = [
    "WasteClassifier",
    "WasteDetectorLocal",
    "WasteDetectorRemote",
]