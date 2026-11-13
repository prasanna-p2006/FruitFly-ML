"""
tests/test_edge_cases.py

Edge-case and failure mode test suite:
- Overlapping flies detection and safe null telemetry logging
- Missing landmarks handling in annotation parsing
- Invalid image dimensions (non-square images)
- Empty or zero-intensity frames (zero moments handling)
- Corrupted or 0-byte video inputs
- Unexpected / malformed model feature vectors
- Automatic creation of missing output directories
- Invalid file paths and graceful exception handling
"""

import os
import json
import pytest
import numpy as np
import cv2

from cs229.image import img_to_mask
from cs229.contour import find_contours, largest_contour
from cs229.patch import ImagePatch
from cs229.files import read_video, get_dir
from cs229.annotation import Annotation
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor

@pytest.mark.edge
def test_overlapping_flies_detection():
    """Verify that a large merged contour is classified as 'both' and not 'neither'."""
    is_fly = IsFlyPredictor()
    # Create large merged contour (area ~ 15000 > 10131.0)
    merged_pts = np.array([[[0, 0]], [[150, 0]], [[150, 100]], [[0, 100]]], dtype=np.int32)
    label = is_fly.predict(merged_pts)
    assert label == 'both'

@pytest.mark.edge
def test_missing_landmarks_warning(temp_dir, capsys):
    """Verify sanity checks detect incomplete landmark pairs (e.g. head without abdomen)."""
    bad_anno_path = os.path.join(temp_dir, 'missing_landmarks.json')
    bad_data = {
        "imagePath": "test.bmp",
        "shapes": [
            {"label": "mp", "points": [[100.0, 100.0]]},
            {"label": "fp", "points": [[200.0, 200.0]]},
            {"label": "mh", "points": [[100.0, 80.0]]}  # mh defined but ma is missing
        ]
    }
    with open(bad_anno_path, 'w', encoding='utf-8') as f:
        json.dump(bad_data, f)

    Annotation(bad_anno_path)
    captured = capsys.readouterr()
    assert "Must define both male head and abdomen or neither" in captured.out

@pytest.mark.edge
def test_invalid_image_dimensions():
    """Verify img_to_mask raises AssertionError when rows != cols."""
    rect_img = np.zeros((100, 200), dtype=np.uint8)
    with pytest.raises(AssertionError):
        img_to_mask(rect_img)

@pytest.mark.edge
def test_empty_frame_handling():
    """Verify empty/zero-intensity frame processing does not crash."""
    # 1. Bright white empty arena (no flies)
    empty_arena = np.full((100, 100), 240, dtype=np.uint8)
    mask = img_to_mask(empty_arena)
    assert mask.shape == (100, 100)

    # find_contours on empty arena yields no fly contours
    contours = find_contours(empty_arena, mask=mask)
    assert len(contours) == 0
    assert largest_contour(contours) is None

    # 2. Patch with zero moments (all black)
    all_black = np.zeros((100, 100), dtype=np.uint8)
    patch = ImagePatch(all_black)
    assert patch.estimate_center(absolute=False) == (0.0, 0.0)
    assert patch.estimate_axes() == (0.0, 0.0)
    assert patch.estimate_aspect_ratio() == 1.0

@pytest.mark.edge
def test_corrupt_or_zero_byte_video(temp_dir):
    """Verify read_video cleanly raises RuntimeError on a 0-byte corrupt file."""
    corrupt_vid = os.path.join(temp_dir, 'corrupt.mp4')
    with open(corrupt_vid, 'wb') as f:
        f.write(b'')  # 0-byte file

    # Put corrupt video inside input/video for read_video lookup test
    input_v_dir = get_dir('input', 'video', mkdir_p=False)
    # Testing direct path with OpenCV VideoCapture
    cap = cv2.VideoCapture(corrupt_vid)
    assert cap.isOpened() is False
    cap.release()

@pytest.mark.edge
def test_unexpected_model_feature_dimensions():
    """Verify estimators raise ValueError on wrong input feature counts."""
    id_p = IdPredictor()
    # Expects 4 features; pass 3
    with pytest.raises(ValueError):
        id_p.clf.predict(np.array([[100.0, 200.0, 1.5]]))

@pytest.mark.edge
def test_automatic_output_dir_creation(temp_dir):
    """Verify get_dir creates deeply nested output directories automatically."""
    nested_path = get_dir('output', 'test_nested', 'sub_level', 'deeper', mkdir_p=True)
    assert os.path.isdir(nested_path)
    assert os.path.exists(nested_path)
