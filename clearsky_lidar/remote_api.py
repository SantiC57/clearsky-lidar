"""Remote inference API client for ClearSky using Roboflow serverless.

Provides WasteDetectorRemote for waste classification via Roboflow API.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


class WasteDetectorRemote:
    """Remote waste detector using Roboflow serverless API."""

    def __init__(
        self,
        api_url: str = "https://serverless.roboflow.com",
        api_key: str | None = None,
        workspace_name: str = "yolov8-ofcbj",
        workflow_id: str = "general-segmentation-api-9",
        classes: list[str] | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize remote waste detector.

        Args:
            api_url: Roboflow serverless API URL.
            api_key: Roboflow API key. If None, reads from ROBOFLOW_API_KEY env var.
            workspace_name: Roboflow workspace name.
            workflow_id: Roboflow workflow ID.
            classes: List of classes to detect. If None, uses default waste classes.
            use_cache: Whether to cache workflow definition.
        """
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Pass api_key or set ROBOFLOW_API_KEY env var."
            )

        self.api_url = api_url
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.classes = classes or ["paper", "plastic", "glass", "metal", "cardboard"]
        self.use_cache = use_cache

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run inference on an image.

        Args:
            image: BGR image as numpy array (OpenCV format).

        Returns:
            Dictionary with inference results from Roboflow API.
        """
        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Build API URL
        url = f"{self.api_url}/{self.workspace_name}/{self.workflow_id}"

        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        payload = {
            "images": {"image": image_base64},
            "parameters": {"classes": ", ".join(self.classes)},
            "use_cache": self.use_cache,
        }

        # Make request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        return response.json()

    def predict_from_file(self, image_path: str | Path) -> dict[str, Any]:
        """Run inference on an image file.

        Args:
            image_path: Path to image file.

        Returns:
            Dictionary with inference results from Roboflow API.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.predict(image)
