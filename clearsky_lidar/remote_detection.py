"""Remote waste detection using Roboflow Inference API.

Provides WasteDetectorRemote for real-time waste detection via Roboflow's
hosted inference service with optimized threading and rate limiting.

Note: This module requires Python 3.8-3.13 and the following optional dependencies:
- supervision
- inference-sdk

Install with: pip install clearsky-lidar[remote]
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

# Optional dependencies
try:
    import supervision as sv
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    REMOTE_DETECTION_AVAILABLE = True
except ImportError:
    REMOTE_DETECTION_AVAILABLE = False
    sv = None
    InferenceHTTPClient = None
    InferenceConfiguration = None


# Configuration
API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
MODEL_ID = "yolov8-trash-detections/6"

MAX_UPLOAD_WIDTH = 640  # Downscale before upload for faster inference
INFERENCE_FPS_LIMIT = 4.0  # Cap API calls per second
CONFIDENCE_THRESHOLD = 0.4  # Hide weak detections
MAX_CONSECUTIVE_ERRORS = 12  # Stop after this many failures
ERROR_BACKOFF_SECONDS = 2.0  # Pause between retries


def downscale(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink a frame to at most max_width px wide, preserving aspect ratio."""
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)))


def check_remote_detection_available() -> None:
    """Check if remote detection dependencies are available."""
    if not REMOTE_DETECTION_AVAILABLE:
        raise ImportError(
            "Remote detection requires Python 3.8-3.13 and the following packages:\n"
            "  - supervision\n"
            "  - inference-sdk\n\n"
            "Install with: pip install clearsky-lidar[remote]"
        )


class InferenceWorker(threading.Thread):
    """Background thread that infers on the newest frame at a bounded rate."""

    def __init__(self) -> None:
        check_remote_detection_available()
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._stop_event = threading.Event()

        self.detections = sv.Detections.empty()
        self.inference_count = 0
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error: str = ""
        self.latency_ms = 0.0
        self.fatal = False

        # Initialize client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=API_KEY,
        ).configure(InferenceConfiguration(api_key_transport="header"))

    def submit(self, frame: np.ndarray) -> None:
        """Offer a frame. Only the newest one is kept; stale ones are dropped."""
        with self._lock:
            self._latest_frame = frame

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Main inference loop with rate limiting."""
        min_interval = 1.0 / INFERENCE_FPS_LIMIT

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            with self._lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                self._stop_event.wait(0.02)
                continue

            try:
                started = time.monotonic()
                result = self.client.infer(
                    downscale(frame, MAX_UPLOAD_WIDTH), model_id=MODEL_ID
                )
                self.latency_ms = (time.monotonic() - started) * 1000

                detections = sv.Detections.from_inference(result).with_nms(threshold=0.3)
                if len(detections) > 0:
                    detections = detections[detections.confidence >= CONFIDENCE_THRESHOLD]
                self.detections = detections

                self.inference_count += 1
                self.consecutive_errors = 0

            except KeyboardInterrupt:
                break
            except Exception as exc:
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"

                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.fatal = True
                    break

                self._stop_event.wait(ERROR_BACKOFF_SECONDS)
                continue

            # Respect the rate limit
            elapsed = time.monotonic() - loop_start
            if elapsed < min_interval:
                self._stop_event.wait(min_interval - elapsed)


class WasteDetectorRemote:
    """Remote waste detector using Roboflow Inference API."""

    def __init__(
        self,
        api_key: str = API_KEY,
        model_id: str = MODEL_ID,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_upload_width: int = MAX_UPLOAD_WIDTH,
        inference_fps_limit: float = INFERENCE_FPS_LIMIT,
    ) -> None:
        """Initialize remote waste detector.

        Args:
            api_key: Roboflow API key
            model_id: Model ID to use for inference
            confidence_threshold: Minimum confidence for detections
            max_upload_width: Maximum width for uploaded frames (downscaled)
            inference_fps_limit: Maximum API calls per second
        """
        check_remote_detection_available()
        
        self.api_key = api_key
        self.model_id = model_id
        self.confidence_threshold = confidence_threshold
        self.max_upload_width = max_upload_width
        self.inference_fps_limit = inference_fps_limit

        # Initialize client
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        ).configure(InferenceConfiguration(api_key_transport="header"))

        # Annotators for visualization
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run inference on a single image.

        Args:
            image: BGR image as numpy array

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
        # Downscale for faster inference
        downscaled = downscale(image, self.max_upload_width)

        # Run inference
        result = self.client.infer(downscaled, model_id=self.model_id)

        # Parse detections
        detections = sv.Detections.from_inference(result).with_nms(threshold=0.3)
        if len(detections) > 0:
            detections = detections[detections.confidence >= self.confidence_threshold]

        # Extract results
        detections_dict = {
            'boxes': [],
            'confidences': [],
            'class_ids': [],
            'class_names': [],
            'count': 0
        }

        if len(detections) > 0:
            detections_dict['boxes'] = detections.xyxy.tolist()
            detections_dict['confidences'] = detections.confidence.tolist()
            detections_dict['class_ids'] = detections.class_id.tolist() if detections.class_id is not None else []
            detections_dict['class_names'] = [self.model_id] * len(detections)  # API doesn't return class names
            detections_dict['count'] = len(detections)

        return detections_dict

    def draw_detections(
        self,
        image: np.ndarray,
        detections: sv.Detections,
    ) -> np.ndarray:
        """Draw detection boxes on image using supervision annotators.

        Args:
            image: BGR image as numpy array
            detections: Supervision Detections object

        Returns:
            Image with detections drawn
        """
        annotated = image.copy()

        if len(detections) > 0:
            labels = [
                f"#{cid} {conf:.2f}" if cid is not None else f"{conf:.2f}"
                for cid, conf in zip(detections.class_id, detections.confidence)
            ]
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels=labels)

        return annotated

    def create_worker(self) -> InferenceWorker:
        """Create an inference worker for real-time detection.

        Returns:
            InferenceWorker instance (not started)
        """
        return InferenceWorker()
