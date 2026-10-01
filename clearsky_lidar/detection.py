"""Detection and classification module for ClearSky waste classification.

Provides:
- WasteClassifier: Local inference using YOLOv8-cls model.
- WasteDetectorRemote: Remote inference using Roboflow serverless API.
"""

from .classification.inference import WasteClassifier
from .remote_api import WasteDetectorRemote

__all__ = [
    "WasteClassifier",
    "WasteDetectorRemote",
]