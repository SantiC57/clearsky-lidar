import datetime
import json
import os

import cv2
import numpy as np
import pytest

from tools.capture_frames import (
    CaptureError,
    build_metadata,
    capture_frames,
    decode_fourcc,
    frame_filename,
    main,
    read_applied,
)

MJPG_CODE = ord("M") | (ord("J") << 8) | (ord("P") << 16) | (ord("G") << 24)
WHEN = datetime.datetime(2026, 10, 16, 9, 5, 3)


class FakeCap(object):
    """Stands in for cv2.VideoCapture; reports what the 'camera applied'."""

    def __init__(self, width=1280, height=720, fps=30.0, fourcc=MJPG_CODE, frames=100):
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: float(width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(height),
            cv2.CAP_PROP_FPS: float(fps),
            cv2.CAP_PROP_FOURCC: float(fourcc),
        }
        self._frames = frames
        self.reads = 0

    def get(self, prop):
        return self._props[prop]

    def read(self):
        self.reads += 1
        if self.reads > self._frames:
            return False, None
        return True, np.zeros((48, 64, 3), dtype=np.uint8)


REQUESTED = {"width": 1280, "height": 720, "fps": 30.0, "fourcc": "MJPG"}


def test_decode_fourcc_reads_mjpg_and_empty():
    assert decode_fourcc(float(MJPG_CODE)) == "MJPG"
    assert decode_fourcc(0.0) == ""


def test_frame_filename_is_tag_timestamp_index():
    assert frame_filename("pos5_5cm", 2, WHEN) == "pos5_5cm_20261016T090503_002.jpg"


def test_read_applied_reads_values_back_from_the_camera():
    applied = read_applied(FakeCap(width=640, height=480, fps=15.0))

    assert applied == {"width": 640, "height": 480, "fps": 15.0, "fourcc": "MJPG"}


def test_metadata_reports_applied_values_and_flags_a_mismatch():
    applied = {"width": 640, "height": 480, "fps": 30.0, "fourcc": "MJPG"}

    meta = build_metadata("a.jpg", WHEN, 2, "t", 0, REQUESTED, applied)

    assert meta["fourcc"] == "MJPG"
    assert (meta["width"], meta["height"]) == (640, 480)
    assert meta["requested"] == REQUESTED
    assert meta["matched_request"] is False
    assert meta["image"] == "a.jpg"
    assert meta["timestamp"] == "2026-10-16T09:05:03"


def test_metadata_marks_matching_request():
    meta = build_metadata("a.jpg", WHEN, 0, "t", 2, REQUESTED, dict(REQUESTED))

    assert meta["matched_request"] is True


def test_capture_writes_one_jpg_and_one_json_per_frame(tmp_path):
    sleeps = []

    names = capture_frames(
        FakeCap(), str(tmp_path), tag="verif", count=3, interval=1.5, camera=2,
        requested=REQUESTED, warmup=2, sleep=sleeps.append, now=lambda: WHEN,
    )

    files = sorted(os.listdir(str(tmp_path)))
    assert len([f for f in files if f.endswith(".jpg")]) == 3
    assert len([f for f in files if f.endswith(".json")]) == 3
    assert names == [f for f in files if f.endswith(".jpg")]
    with open(os.path.join(str(tmp_path), names[0].replace(".jpg", ".json"))) as f:
        meta = json.load(f)
    assert meta["fourcc"] == "MJPG" and meta["image"] == names[0]
    assert cv2.imread(os.path.join(str(tmp_path), names[0])).shape == (48, 64, 3)
    assert sleeps == [1.5, 1.5]


def test_capture_raises_when_the_camera_stops_delivering(tmp_path):
    with pytest.raises(CaptureError):
        capture_frames(
            FakeCap(frames=2), str(tmp_path), tag="t", count=5, interval=0, camera=0,
            requested=REQUESTED, warmup=0, sleep=lambda s: None, now=lambda: WHEN,
        )


def test_capture_refuses_to_overwrite_an_existing_photo(tmp_path):
    existing = tmp_path / frame_filename("t", 0, WHEN)
    existing.write_bytes(b"precious")

    with pytest.raises(CaptureError):
        capture_frames(
            FakeCap(), str(tmp_path), tag="t", count=2, interval=0, camera=0,
            requested=REQUESTED, warmup=0, sleep=lambda s: None, now=lambda: WHEN,
        )

    assert existing.read_bytes() == b"precious"


@pytest.mark.parametrize("flag,value", [
    ("--count", "0"), ("--count", "-3"), ("--interval", "-1"), ("--warmup", "-1"),
    ("--width", "0"), ("--height", "-720"), ("--fps", "0"),
])
def test_main_rejects_invalid_numbers_before_touching_the_camera(tmp_path, flag, value):
    opened = []

    with pytest.raises(SystemExit) as exc:
        main(["--out", str(tmp_path), flag, value],
             open_camera=lambda *a, **k: opened.append(a))

    assert exc.value.code == 2
    assert opened == []


def test_main_returns_error_code_when_camera_cannot_open(tmp_path):
    rc = main(["--camera", "0", "--out", str(tmp_path), "--tag", "t"],
              open_camera=lambda *a, **k: None)

    assert rc == 1
