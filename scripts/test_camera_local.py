#!/usr/bin/env python3
"""Test camera inference with local YOLOv8 detection model.

Usage:
    python scripts/test_camera_local.py
    python scripts/test_camera_local.py --camera 0 --width 1280 --height 720
    python scripts/test_camera_local.py --model models/best.pt --confidence 0.6
"""

import argparse
import os
import time

import cv2
import numpy as np

from clearsky_lidar.local_detection import WasteDetectorLocal


def main():
    parser = argparse.ArgumentParser(
        description="Test camera inference with local YOLOv8 detection model"
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
        "--model",
        type=str,
        default=None,
        help="Path to model file (default: models/best.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cpu, cuda, mps, or None for auto)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save frames with detections to output/",
    )
    parser.add_argument(
        "--inference-every",
        type=int,
        default=1,
        help="Run inference every N frames (default: 1)",
    )
    args = parser.parse_args()

    # Initialize detector
    print("[ClearSky] Initializing local waste detector...")
    detector = WasteDetectorLocal(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
    )
    print(f"[ClearSky] Model: {detector.model_path}")
    print(f"[ClearSky] Classes: {', '.join(detector.classes)}")
    print(f"[ClearSky] Confidence threshold: {args.confidence}")
    print(f"[ClearSky] Device: {args.device or 'auto'}")

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
                    if result['count'] > 0:
                        print(f"\n[ClearSky] Detections ({result['count']}):")
                        for i in range(result['count']):
                            class_name = result['class_names'][i]
                            confidence = result['confidences'][i]
                            print(f"  - {class_name}: {confidence:.2f}")

                    if args.save and result['count'] > 0:
                        output_path = f"output/frame_{frame_count:04d}.jpg"
                        annotated_frame = detector.draw_detections(frame, result)
                        cv2.imwrite(output_path, annotated_frame)
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

            # Draw detections on frame
            if last_result:
                frame = detector.draw_detections(frame, last_result)

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

            # Display detection count
            if last_result:
                count_text = f"Detections: {last_result['count']}"
                cv2.putText(
                    frame,
                    count_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Show frame
            cv2.imshow("ClearSky - Local Detection", frame)

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
