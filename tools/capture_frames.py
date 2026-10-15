#!/usr/bin/env python3
"""Capture photos from a USB camera in MJPG and save a JSON next to each photo.

The JSON records what the camera really applied (width, height, FPS, FOURCC),
read back from the device, plus what was requested. Cameras often ignore a
request silently, so the requested values alone prove nothing.

Usage:
    python3 tools/capture_frames.py --camera 0 --width 1280 --height 720 \
        --count 30 --interval 1.0 --out data/experiments/<ID>/raw --tag pos5_5cm
"""
import argparse
import datetime
import json
import os
import sys
import time


class CaptureError(RuntimeError):
    pass


def decode_fourcc(value):
    """Turn the number returned by cap.get(CAP_PROP_FOURCC) into 'MJPG'."""
    code = int(value)
    chars = [chr((code >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars).strip("\x00")


def frame_filename(tag, index, when):
    return "{}_{}_{:03d}.jpg".format(tag, when.strftime("%Y%m%dT%H%M%S"), index)


def read_applied(cap):
    """Read back from the camera the values it is actually using."""
    import cv2

    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "fourcc": decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC)),
    }


def build_metadata(image_name, when, index, tag, camera, requested, applied):
    """Metadata for one photo. Top-level width/height/fps/fourcc are the applied values.

    matched_request compares resolution and FOURCC; FPS is reported but not
    compared because many drivers report a rounded or zero value.
    """
    matched = all(applied[k] == requested[k] for k in ("width", "height", "fourcc"))
    return {
        "image": image_name,
        "timestamp": when.isoformat(),
        "index": index,
        "tag": tag,
        "camera": camera,
        "fourcc": applied["fourcc"],
        "width": applied["width"],
        "height": applied["height"],
        "fps": applied["fps"],
        "requested": requested,
        "matched_request": matched,
    }


def capture_frames(cap, out_dir, tag, count, interval, camera, requested,
                   warmup=5, sleep=time.sleep, now=datetime.datetime.now, imwrite=None):
    """Save ``count`` photos plus one JSON each. Returns the photo file names."""
    if imwrite is None:
        import cv2
        imwrite = cv2.imwrite
    os.makedirs(out_dir, exist_ok=True)

    for _ in range(warmup):  # first frames are often dark or unstable
        ok, _frame = cap.read()
        if not ok:
            raise CaptureError("camera stopped delivering frames during warm-up")

    names = []
    for index in range(count):
        ok, frame = cap.read()
        if not ok:
            raise CaptureError("camera stopped delivering frames at photo {}".format(index))
        when = now()
        name = frame_filename(tag, index, when)
        if os.path.exists(os.path.join(out_dir, name)):
            raise CaptureError("{} already exists; refusing to overwrite, "
                               "use another --tag or --out".format(name))
        if not imwrite(os.path.join(out_dir, name), frame):
            raise CaptureError("could not write {}".format(name))
        meta = build_metadata(name, when, index, tag, camera, requested, read_applied(cap))
        with open(os.path.join(out_dir, name[:-len(".jpg")] + ".json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
            f.write("\n")
        names.append(name)
        if index < count - 1:
            sleep(interval)
    return names


def open_camera(camera, width, height, fps):
    """Open a V4L2/USB camera asking for MJPG. Returns None if it cannot be opened."""
    import cv2

    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera, backend)
    if not cap.isOpened():
        cap.release()
        return None
    # FOURCC first: some drivers reset the resolution when the format changes.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _number(kind, minimum, allow_equal):
    """argparse type: a number of ``kind`` above (or equal to) ``minimum``."""
    def parse(text):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError("not a number: {!r}".format(text))
        if value < minimum or (value == minimum and not allow_equal):
            raise argparse.ArgumentTypeError(
                "must be {} {}, got {}".format(">=" if allow_equal else ">", minimum, text))
        return value
    return parse


def main(argv=None, open_camera=open_camera):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=_number(int, 0, True), default=0,
                        help="V4L2 index (N in /dev/videoN)")
    parser.add_argument("--width", type=_number(int, 0, False), default=1280)
    parser.add_argument("--height", type=_number(int, 0, False), default=720)
    parser.add_argument("--fps", type=_number(float, 0, False), default=30.0)
    parser.add_argument("--count", type=_number(int, 0, False), default=30)
    parser.add_argument("--interval", type=_number(float, 0, True), default=1.0,
                        help="seconds between photos")
    parser.add_argument("--warmup", type=_number(int, 0, True), default=5,
                        help="frames discarded before saving")
    parser.add_argument("--out", required=True, help="folder for photos and JSON files")
    parser.add_argument("--tag", default="frame", help="prefix for file names")
    args = parser.parse_args(argv)

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if cap is None:
        sys.stderr.write("error: cannot open camera {}\n".format(args.camera))
        return 1
    requested = {"width": args.width, "height": args.height, "fps": args.fps, "fourcc": "MJPG"}
    try:
        applied = read_applied(cap)
        print("requested: {}".format(requested))
        print("applied:   {}".format(applied))
        if any(applied[k] != requested[k] for k in ("width", "height", "fourcc")):
            sys.stderr.write("warning: the camera did not accept the request; the JSON "
                             "records the values it used\n")
        names = capture_frames(cap, args.out, args.tag, args.count, args.interval,
                               args.camera, requested, warmup=args.warmup)
    except CaptureError as exc:
        sys.stderr.write("error: {}\n".format(exc))
        return 1
    finally:
        cap.release()
    print("saved {} photos in {}".format(len(names), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
