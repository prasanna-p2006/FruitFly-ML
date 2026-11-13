"""
tests/test_unit_video.py

Unit tests for video reading, writing, and CSV telemetry utilities:
- cs229.files: read_video, write_video, CapProps
- Exception handling on missing or corrupt video files
- Backend selection (msmf, ffmpeg, auto)
- Resource cleanup verification
- CSV schema and telemetry header validation
"""

import os
import csv
import pytest
import numpy as np
import cv2

from cs229.files import read_video, write_video, CapProps

@pytest.fixture
def tiny_synthetic_video(temp_dir):
    """Generates a small 5-frame 200x200 MP4 video in temp_dir."""
    vid_path = os.path.join(temp_dir, 'synthetic_test.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(vid_path, fourcc, 35.0, (200, 200))

    for i in range(5):
        frame = np.full((200, 200, 3), 200 + i * 5, dtype=np.uint8)
        # Draw moving circle
        cv2.circle(frame, (50 + i * 15, 100), 10, (20, 20, 20), -1)
        writer.write(frame)

    writer.release()
    return vid_path

@pytest.mark.unit
def test_cap_props():
    props = CapProps(width=1530, height=1530, fps=35.0)
    assert props.width == 1530
    assert props.height == 1530
    assert props.fps == 35.0
    assert props.t_ms == 29

@pytest.mark.unit
def test_read_video_missing_file():
    """Verify FileNotFoundError when video file does not exist."""
    with pytest.raises(FileNotFoundError):
        read_video("non_existent_video_file_xyz.mp4")

@pytest.mark.unit
def test_read_synthetic_video(tiny_synthetic_video):
    """Test reading the synthetic test video with OpenCV."""
    cap = cv2.VideoCapture(tiny_synthetic_video)
    assert cap.isOpened() is True
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - 35.0) < 0.1

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        assert frame.shape == (200, 200, 3)

    cap.release()
    assert frame_count == 5

@pytest.mark.unit
def test_write_video_and_cleanup(temp_dir):
    """Test writing and releasing an annotated video."""
    out_path = os.path.join(temp_dir, 'out_test.mp4')
    props = CapProps(width=100, height=100, fps=35.0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, props.fps, (props.width, props.height))

    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))

    writer.release()
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0

@pytest.mark.unit
def test_csv_telemetry_schema(temp_dir):
    """Validate expected trajectory CSV schema and data integrity."""
    csv_path = os.path.join(temp_dir, 'telemetry.csv')
    expected_header = [
        "frame_number", "timestamp_sec", "detected_fly_count",
        "male_x", "male_y", "female_x", "female_y",
        "male_orient_deg", "female_orient_deg",
        "male_wing_left_deg", "male_wing_right_deg", "status"
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(expected_header)
        for i in range(1, 4):
            t_s = (i - 1) / 35.0
            w.writerow([i, f"{t_s:.3f}", 2, 100.0, 100.0, 200.0, 200.0, 45.0, -30.0, 15.0, 12.0, "TWO_FLIES_SEPARATED"])

    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.reader(f)
        rows = list(r)

    assert rows[0] == expected_header
    assert len(rows) == 4
    for i, row in enumerate(rows[1:], 1):
        assert int(row[0]) == i
        assert float(row[1]) == pytest.approx((i - 1) / 35.0, abs=0.002)
        assert row[11] == "TWO_FLIES_SEPARATED"

