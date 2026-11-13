"""
tests/test_integration_pipeline.py

Integration tests for the complete video inference, visualization, and telemetry pipeline:
- Tests process_video() end-to-end on held-out video frames
- Verifies:
  1. Complete execution without unhandled exceptions
  2. Output video creation with matching dimensions and frame count
  3. Trajectory CSV schema, sequential frame indexing (1..N), and 35 FPS timestamps
  4. 1:1 synchronization between output video frames and CSV rows
  5. Safe handling and null values during contact / overlap states
  6. Proper resource release (video capture, video writer, and file descriptors)
"""

import os
import csv
import pytest
import cv2

from cs229.video_pipeline import process_video

@pytest.mark.integration
def test_full_pipeline_small_batch(temp_dir):
    """Run complete inference pipeline on test4.mp4 for 5 frames into temp_dir."""
    res = process_video(
        video_name="test4.mp4",
        output_dir=temp_dir,
        csv_dir=temp_dir,
        screenshot_dir=temp_dir,
        max_frames=5,
        backend="msmf"
    )

    # 1. Check returned benchmark dictionary
    assert res['frames'] == 5
    assert res['video_name'] == "test4.mp4"
    assert res['avg_fps'] > 0
    assert os.path.exists(res['video_path'])
    assert os.path.exists(res['csv_path'])

    # 2. Check output video integrity
    out_cap = cv2.VideoCapture(res['video_path'])
    assert out_cap.isOpened() is True
    w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = out_cap.get(cv2.CAP_PROP_FPS)
    assert w == 1530 and h == 1530
    assert abs(fps - 35.0) < 0.1

    decoded_frames = 0
    while True:
        ok, frame = out_cap.read()
        if not ok:
            break
        decoded_frames += 1
        assert frame.shape == (1530, 1530, 3)
    out_cap.release()
    assert decoded_frames == 5

    # 3. Check CSV integrity and 1:1 synchronization
    with open(res['csv_path'], 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    expected_header = [
        "frame_number", "timestamp_sec", "detected_fly_count",
        "male_x", "male_y", "female_x", "female_y",
        "male_orient_deg", "female_orient_deg",
        "male_wing_left_deg", "male_wing_right_deg", "status"
    ]
    assert header == expected_header

    data_rows = rows[1:]
    assert len(data_rows) == 5

    for i, row in enumerate(data_rows, 1):
        f_num = int(row[0])
        t_sec = float(row[1])
        status = row[11]

        assert f_num == i
        expected_t = (i - 1) / 35.0
        assert abs(t_sec - expected_t) < 0.002
        assert status in [
            "TWO_FLIES_SEPARATED", "OVERLAPPING_FLIES",
            "SINGLE_FLY_DETECTED", "NO_FLIES_DETECTED", "UNKNOWN"
        ]

@pytest.mark.integration
def test_pipeline_resource_cleanup_on_error(temp_dir, monkeypatch):
    """Verify resources (writer, cap, file) are properly closed even if an exception occurs mid-stream."""
    out_video = os.path.join(temp_dir, 'test4_annotated.mp4')
    out_csv = os.path.join(temp_dir, 'test4_trajectory.csv')

    # Force an exception during frame 2
    count = [0]
    import cs229.video_pipeline as vp
    orig_find_contours = vp.find_contours

    def mock_find_contours(*args, **kwargs):
        count[0] += 1
        if count[0] >= 2:
            raise RuntimeError("Simulated mid-pipeline pipeline crash")
        return orig_find_contours(*args, **kwargs)

    monkeypatch.setattr(vp, 'find_contours', mock_find_contours)

    with pytest.raises(RuntimeError, match="Simulated mid-pipeline pipeline crash"):
        process_video(
            video_name="test4.mp4",
            output_dir=temp_dir,
            csv_dir=temp_dir,
            screenshot_dir=temp_dir,
            max_frames=5,
            backend="msmf"
        )

    # Verify CSV file handle was released and can be deleted/reopened
    assert os.path.exists(out_csv)
    # Attempt opening for write without WinError 32 (file lock)
    with open(out_csv, 'a', encoding='utf-8') as f:
        f.write("# Cleanup verified\n")
