"""
tests/test_unit_image_contour.py

Unit tests for image and contour processing modules:
- cs229.image: bound_point, img_to_mask (square and non-square handling)
- cs229.contour: threshold_core, threshold_wings, find_contours, largest_contour,
                 contour_roi, in_contour, contour_label
- cs229.patch: crop_to_contour, moments_to_angle, moments_to_center, ImagePatch
"""

import pytest
import numpy as np
import cv2

from cs229.image import bound_point, img_to_mask
from cs229.contour import (
    find_contours, largest_contour, contour_roi, in_contour,
    threshold_core, threshold_wings, contour_label
)
from cs229.patch import (
    crop_to_contour, moments_to_angle, moments_to_center, ImagePatch
)

@pytest.mark.unit
def test_bound_point():
    img = np.zeros((100, 200), dtype=np.uint8) # height=100 (rows), width=200 (cols)
    # Point within bounds
    pt = bound_point((50, 40), img)
    assert pt == (50, 40)
    # Point outside left/top
    pt_neg = bound_point((-10, -5), img)
    assert pt_neg == (0, 0)
    # Point outside right/bottom
    pt_over = bound_point((250, 150), img)
    assert pt_over == (199, 99)

@pytest.mark.unit
def test_img_to_mask():
    # Square image
    square_img = np.zeros((400, 400), dtype=np.uint8)
    mask = img_to_mask(square_img)
    assert mask.shape == (400, 400)
    assert mask.dtype == np.uint8
    assert mask[200, 200] == 255  # Center inside circle
    assert mask[0, 0] == 0        # Corner outside circle

    # Non-square image must raise AssertionError
    non_square_img = np.zeros((300, 400), dtype=np.uint8)
    with pytest.raises(AssertionError):
        img_to_mask(non_square_img)

@pytest.mark.unit
def test_threshold_core_with_and_without_mask():
    img = np.full((100, 100), 200, dtype=np.uint8)
    # Put a dark object in the center
    cv2.circle(img, (50, 50), 10, 30, -1)

    # With mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 40, 255, -1)
    bw = threshold_core(img, mask=mask)
    assert bw.shape == (100, 100)
    assert bw[50, 50] == 255
    assert bw[0, 0] == 0

    # Without mask (mask=None must not crash)
    bw_nomask = threshold_core(img, mask=None)
    assert bw_nomask.shape == (100, 100)
    assert bw_nomask[50, 50] == 255

@pytest.mark.unit
def test_threshold_wings_with_and_without_mask():
    img = np.full((100, 100), 220, dtype=np.uint8)
    cv2.circle(img, (50, 50), 8, 40, -1)

    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 45, 255, -1)

    bw = threshold_wings(img, mask=mask)
    assert bw.shape == (100, 100)

    # mask=None must execute cleanly without crash
    bw_nomask = threshold_wings(img, mask=None)
    assert bw_nomask.shape == (100, 100)

@pytest.mark.unit
def test_find_contours_and_largest():
    # Empty image (all white, no flies)
    img_empty = np.full((200, 200), 240, dtype=np.uint8)
    mask = img_to_mask(img_empty)
    contours_empty = find_contours(img_empty, mask=mask)
    assert len(contours_empty) == 0
    assert largest_contour(contours_empty) is None

    # Single fly
    img_single = img_empty.copy()
    cv2.circle(img_single, (100, 100), 10, 20, -1)
    contours_single = find_contours(img_single, mask=mask)
    assert len(contours_single) == 1
    lc = largest_contour(contours_single)
    assert lc is not None
    assert cv2.contourArea(lc) > 200

    # Two separated flies
    img_two = img_empty.copy()
    cv2.circle(img_two, (70, 70), 8, 20, -1)
    cv2.circle(img_two, (130, 130), 12, 20, -1)
    contours_two = find_contours(img_two, mask=mask)
    assert len(contours_two) == 2

@pytest.mark.unit
def test_contour_roi_and_in_contour():
    # Create simple rectangular contour
    c = np.array([[[10, 20]], [[30, 20]], [[30, 50]], [[10, 50]]], dtype=np.int32)
    roi_y, roi_x = contour_roi(c)
    assert roi_x == slice(10, 31)
    assert roi_y == slice(20, 51)

    # In contour check
    assert in_contour((20, 35), c) is True
    assert in_contour((5, 5), c) is False

@pytest.mark.unit
def test_crop_to_contour_and_image_patch():
    img = np.full((100, 100), 240, dtype=np.uint8)
    cv2.ellipse(img, (50, 50), (15, 8), 0, 0, 360, 20, -1)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    contours = find_contours(img, mask=mask)
    assert len(contours) >= 1

    patch = crop_to_contour(img, contours[0])
    assert isinstance(patch, ImagePatch)
    assert patch.img.shape[0] > 0 and patch.img.shape[1] > 0

    # Test center estimation
    cx, cy = patch.estimate_center(absolute=True)
    assert 45 <= cx <= 55
    assert 45 <= cy <= 55

    # Test axes and aspect ratio
    MA, ma = patch.estimate_axes()
    assert MA > ma > 0
    aspect = patch.estimate_aspect_ratio()
    assert aspect > 1.0

    # Test rotation
    p_rot = patch.rotate(np.pi / 2)
    assert isinstance(p_rot, ImagePatch)
    p_180 = patch.rotate180()
    assert isinstance(p_180, ImagePatch)

@pytest.mark.unit
def test_image_patch_zero_moments_safety():
    """Verify zero-intensity / blank patch does not crash with ZeroDivisionError."""
    blank = np.zeros((20, 20), dtype=np.uint8)
    patch = ImagePatch(img=blank)

    # Must return (0, 0) without ZeroDivisionError
    assert patch.estimate_center(absolute=False) == (0.0, 0.0)
    assert patch.estimate_axes() == (0.0, 0.0)
    assert patch.estimate_aspect_ratio() == 1.0
    assert patch.estimate_angle() == 0.0

