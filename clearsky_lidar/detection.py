"""Detection and classification module for ClearSky waste classification.

Provides:
- WasteDetectorRemote: Remote inference using Roboflow serverless API.
"""

from .remote_detection import WasteDetectorRemote

__all__ = [
    "WasteDetectorRemote",
]