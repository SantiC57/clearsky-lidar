"""Remote inference API client for ClearSky using Roboflow serverless.

Provides WasteDetectorRemote for waste classification via Roboflow API.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
except ImportError:
    InferenceHTTPClient = None
    InferenceConfiguration = None


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
        if InferenceHTTPClient is None:
            raise ImportError(
                "inference-sdk is required for remote inference. "
                "Install with: pip install inference-sdk"
            )

        import os

        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Pass api_key or set ROBOFLOW_API_KEY env var."
            )

        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.classes = classes or ["paper", "plastic", "glass", "metal", "cardboard"]
        self.use_cache = use_cache

        self.client = InferenceHTTPClient(
            api_url=api_url,
            api_key=self.api_key,
        ).configure(InferenceConfiguration(api_key_transport="header"))

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

        # Run workflow
        result = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": image_base64},
            parameters={"classes": ", ".join(self.classes)},
            use_cache=self.use_cache,
        )

        return result

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

        result = self.client.run_workflow(
            workspace_name=self.workspace_name,
            workflow_id=self.workflow_id,
            images={"image": str(image_path)},
            parameters={"classes": ", ".join(self.classes)},
            use_cache=self.use_cache,
        )

        return result
