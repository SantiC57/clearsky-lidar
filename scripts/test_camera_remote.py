#!/usr/bin/env python3
"""Live webcam waste detection using Roboflow Inference API.

Usage:
    python scripts/test_camera_remote.py
    python scripts/test_camera_remote.py --camera 0
    python scripts/test_camera_remote.py --list-cameras

Note: Requires Python 3.8-3.13 and supervision + inference-sdk packages.
"""

import argparse
import glob
import os
import signal
import sys
import time

# Force X11/XWayland backend so OpenCV's bundled Qt works under Wayland
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2

# Check for optional dependencies
try:
    import supervision as sv
    from clearsky_lidar.remote_detection import (
        WasteDetectorRemote,
        InferenceWorker,
        downscale,
        MAX_UPLOAD_WIDTH,
        INFERENCE_FPS_LIMIT,
        CONFIDENCE_THRESHOLD,
        REMOTE_DETECTION_AVAILABLE,
    )
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    print("\nRemote detection requires Python 3.8-3.13 and the following packages:", file=sys.stderr)
    print("  - supervision", file=sys.stderr)
    print("  - inference-sdk", file=sys.stderr)
    print("\nInstall with: pip install clearsky-lidar[remote]", file=sys.stderr)
    sys.exit(1)

if not REMOTE_DETECTION_AVAILABLE:
    print("Error: Remote detection dependencies not available.", file=sys.stderr)
    print("Install with: pip install clearsky-lidar[remote]", file=sys.stderr)
    sys.exit(1)


WINDOW_NAME = "Live Waste Detection"


def _read_device_name(index: int) -> str:
    """Read the human-readable V4L2 card name for a /dev/videoN index."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown device"


def find_device_holder(index: int) -> str | None:
    """Return the name of the process holding /dev/videoN open, if any."""
    target = os.path.realpath(f"/dev/video{index}")
    for proc_fd_dir in glob.glob("/proc/[0-9]*/fd"):
        pid = proc_fd_dir.split("/")[2]
        try:
            for fd in os.listdir(proc_fd_dir):
                try:
                    if os.path.realpath(os.path.join(proc_fd_dir, fd)) == target:
                        with open(f"/proc/{pid}/comm", encoding="utf-8") as fh:
                            return fh.read().strip()
                except OSError:
                    continue
        except OSError:
            continue
    return None


def list_cameras(probe: bool = True) -> list[tuple[int, str, bool, str | None]]:
    """Return (index, name, usable, held_by) for every video node found."""
    results = []
    for path in sorted(
        glob.glob("/dev/video*"),
        key=lambda p: int("".join(c for c in p if c.isdigit()) or 0),
    ):
        digits = "".join(c for c in os.path.basename(path) if c.isdigit())
        if not digits:
            continue
        index = int(digits)
        name = _read_device_name(index)

        usable = False
        if probe:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                for _ in range(5):
                    if cap.read()[0]:
                        usable = True
                        break
            cap.release()

        held_by = None if usable else find_device_holder(index)
        results.append((index, name, usable, held_by))
    return results


def print_cameras() -> None:
    """Print detected cameras, flagging which ones can actually stream."""
    print("Probing video devices... (this opens each node briefly)\n")
    cameras = list_cameras()

    if not cameras:
        print("No /dev/video* devices found.")
        return

    usable_indexes = []
    for idx, name, ok, held_by in cameras:
        if ok:
            mark = "USABLE"
            usable_indexes.append(idx)
        elif held_by:
            mark = f"BUSY - in use by '{held_by}'"
        else:
            mark = "no frames (metadata node)"
        print(f"  index {idx}: {name}  [{mark}]")

    if usable_indexes:
        print(f"\nRun with --camera N, e.g.:  --camera {usable_indexes[0]}")
    else:
        print("\nNo usable camera found. Close apps that may hold one (browser, video calls).")


def draw_hud(frame: np.ndarray, worker: InferenceWorker) -> np.ndarray:
    """Overlay live status so API health and throughput are visible."""
    annotated = frame.copy()

    if len(worker.detections) > 0:
        labels = [
            f"#{cid} {conf:.2f}" if cid is not None else f"{conf:.2f}"
            for cid, conf in zip(worker.detections.class_id, worker.detections.confidence)
        ]
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        annotated = box_annotator.annotate(annotated, worker.detections)
        annotated = label_annotator.annotate(annotated, worker.detections, labels=labels)

    det_count = len(worker.detections)
    status = (
        f"detections: {det_count}  |  api calls: {worker.inference_count}  |  "
        f"latency: {worker.latency_ms:.0f} ms  |  errors: {worker.error_count}"
    )
    cv2.putText(
        annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
    )

    if worker.last_error:
        cv2.putText(
            annotated,
            f"API error: {worker.last_error[:70]}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )
    if worker.fatal:
        cv2.putText(
            annotated,
            "Too many errors - press q to exit",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return annotated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam waste detection via Roboflow Inference API."
    )
    parser.add_argument(
        "-c", "--camera",
        type=int,
        default=0,
        help="camera index to use (default: 0)",
    )
    parser.add_argument(
        "-l", "--list-cameras",
        action="store_true",
        help="detect and list available cameras, then exit",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        default=INFERENCE_FPS_LIMIT,
        help=f"max API calls per second (default: {INFERENCE_FPS_LIMIT})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_cameras:
        print_cameras()
        return 0

    # Override global FPS limit
    import clearsky_lidar.remote_detection as rd
    rd.INFERENCE_FPS_LIMIT = max(0.5, args.fps_limit)

    camera_index = args.camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        held_by = find_device_holder(camera_index)
        if held_by:
            print(
                f"Error: camera index {camera_index} ({_read_device_name(camera_index)}) "
                f"is being used by '{held_by}'. Close it and retry, "
                f"or pick another with --camera N.",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: could not open camera index {camera_index}. "
                f"Run with --list-cameras to see what is available.",
                file=sys.stderr,
            )
        return 1

    # Verify camera delivers frames
    warmup_ok = False
    for _ in range(10):
        if cap.read()[0]:
            warmup_ok = True
            break
    if not warmup_ok:
        print(
            f"Error: camera index {camera_index} opened but delivered no frames. "
            f"It may be a metadata-only node. Run --list-cameras to check.",
            file=sys.stderr,
        )
        cap.release()
        return 1

    # Reduce internal buffering
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Using camera index {camera_index}: {_read_device_name(camera_index)}")
    print("Live detection started. Press 'q' or ESC to exit.")
    print(f"Inference capped at {rd.INFERENCE_FPS_LIMIT:.0f} API calls/second.")

    worker = InferenceWorker()
    worker.start()

    def _handle_signal(signum, frame):
        worker.stop()

    signal.signal(signal.SIGINT, _handle_signal)

    exit_code = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from camera.", file=sys.stderr)
                exit_code = 1
                break

            worker.submit(frame)
            cv2.imshow(WINDOW_NAME, draw_hud(frame, worker))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if worker.fatal:
                print(
                    f"Stopping: {worker.consecutive_errors} consecutive API errors. "
                    f"Last error: {worker.last_error}",
                    file=sys.stderr,
                )
                exit_code = 1
                break
    finally:
        worker.stop()
        worker.join(timeout=3.0)
        cap.release()
        cv2.destroyAllWindows()
        print(
            f"\nStopped. Successful inferences: {worker.inference_count}, "
            f"errors: {worker.error_count}."
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
