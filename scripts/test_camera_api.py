#!/usr/bin/env python3
"""Test camera inference with Roboflow remote API.

Usage:
    python scripts/test_camera_api.py
    python scripts/test_camera_api.py --camera 0 --width 1280 --height 720
    python scripts/test_camera_api.py --api-key YOUR_API_KEY
"""

import argparse
import os
import time

import cv2
import numpy as np

from clearsky_lidar.remote_api import WasteDetectorRemote


def main():
    parser = argparse.ArgumentParser(
        description="Test camera inference with Roboflow remote API"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Frame width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720, help="Frame height (default: 720)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Target FPS (default: 30)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="yolov8-ofcbj",
        help="Roboflow workspace name (default: yolov8-ofcbj)",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="general-segmentation-api-9",
        help="Roboflow workflow ID (default: general-segmentation-api-9)",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["paper", "plastic", "glass", "metal", "cardboard"],
        help="Classes to detect (default: paper plastic glass metal cardboard)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save frames with detections to output/",
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=5,
        help="Run inference every N frames (default: 5)",
    )
    args = parser.parse_args()

    # Initialize detector
    print("[ClearSky] Initializing remote waste detector...")
    detector = WasteDetectorRemote(
        api_key=args.api_key,
        workspace_name=args.workspace,
        workflow_id=args.workflow,
        classes=args.classes,
    )
    print(f"[ClearSky] Classes: {', '.join(args.classes)}")

    # Open camera
    print(f"[ClearSky] Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[ClearSky] Camera opened: {actual_width}x{actual_height}")
    print("[ClearSky] Press 'q' to quit")

    if args.save:
        os.makedirs("output", exist_ok=True)
        frame_count = 0

    frame_times = []
    last_result = None
    inference_every_n_frames = args.inference_every

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ClearSky] Failed to read frame")
                break

            current_time = time.time()

            # Run inference every N frames
            if len(frame_times) % inference_every_n_frames == 0:
                try:
                    result = detector.predict(frame)
                    last_result = result

                    # Print results
                    print(f"\n[ClearSky] Detection result:")
                    print(result)

                    if args.save:
                        output_path = f"output/frame_{frame_count:04d}.jpg"
                        cv2.imwrite(output_path, frame)
                        frame_count += 1

                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")

            # Calculate FPS
            frame_times.append(current_time)
            if len(frame_times) > 15:
                frame_times.pop(0)
            if len(frame_times) > 1:
                avg_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
            else:
                avg_fps = 0

            # Display FPS on frame
            cv2.putText(
                frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Display last detection result on frame
            if last_result:
                y_offset = 60
                for key, value in last_result.items():
                    text = f"{key}: {value}"
                    cv2.putText(
                        frame,
                        text[:50],  # Truncate long text
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    y_offset += 20

            # Show frame
            cv2.imshow("ClearSky - Remote API Inference", frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[ClearSky] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[ClearSky] Camera closed")


if __name__ == "__main__":
    main()
