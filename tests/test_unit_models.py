"""
tests/test_unit_models.py

Unit tests for trained machine learning models and predictor wrappers:
- IsFlyPredictor (Decision Tree on contour area)
- IdPredictor (StandardScaler + Logistic Regression on pair geometry)
- PosePredictor (HOG + PCA-15 + Logistic Regression for male and female)
- WingPredictor (HOG + PCA-40 + Linear Regression for male wings)
- Input dimensionality verification and error handling
"""

import pytest
import numpy as np
import cv2

from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor
from cs229.train_wing import WingPredictor
from cs229.patch import ImagePatch

@pytest.fixture(scope="module")
def predictors():
    """Load all 5 predictors once for module tests."""
    return {
        'is_fly': IsFlyPredictor(),
        'id': IdPredictor(),
        'pose_m': PosePredictor('male'),
        'pose_f': PosePredictor('female'),
        'wing': WingPredictor()
    }

@pytest.mark.unit
def test_is_fly_predictor(predictors):
    is_fly = predictors['is_fly']

    # 1. Tiny contour (area < 1413.5) -> 'neither'
    pts_small = np.array([[[10, 10]], [[30, 10]], [[30, 30]], [[10, 30]]], dtype=np.int32) # area 400
    assert is_fly.predict(pts_small) == 'neither'

    # 2. Single fly contour (1413.5 < area < 10131) -> 'one'
    pts_one = np.array([[[10, 10]], [[70, 10]], [[70, 70]], [[10, 70]]], dtype=np.int32) # area 3600
    assert is_fly.predict(pts_one) == 'one'

    # 3. Merged / two flies contour (area > 10131) -> 'both'
    pts_both = np.array([[[10, 10]], [[160, 10]], [[160, 110]], [[10, 110]]], dtype=np.int32) # area 15000
    assert is_fly.predict(pts_both) == 'both'

@pytest.mark.unit
def test_id_predictor(predictors):
    id_pred = predictors['id']

    # Male-like patch (smaller area, typical aspect ratio)
    c_m = np.array([[[10, 10]], [[35, 10]], [[35, 30]], [[10, 30]]], dtype=np.int32)
    img_m = np.zeros((50, 50), dtype=np.uint8)
    cv2.drawContours(img_m, [c_m], -1, 255, -1)
    p_m = ImagePatch(img_m)

    # Female-like patch (larger area)
    c_f = np.array([[[10, 10]], [[45, 10]], [[45, 40]], [[10, 40]]], dtype=np.int32)
    img_f = np.zeros((60, 60), dtype=np.uint8)
    cv2.drawContours(img_f, [c_f], -1, 255, -1)
    p_f = ImagePatch(img_f)

    # Predict pair (m, f)
    res_mf = id_pred.predict(c_m, c_f, p_m, p_f)
    assert res_mf in ['mf', 'fm']

    # Swapped pair (f, m)
    res_fm = id_pred.predict(c_f, c_m, p_f, p_m)
    assert res_fm in ['mf', 'fm']

@pytest.mark.unit
def test_pose_predictor(predictors):
    pose_m = predictors['pose_m']
    pose_f = predictors['pose_f']

    # Create synthetic fly patch (oriented ellipse: 40x20 in 80x80 canvas)
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(img, (50, 50), (20, 10), 30, 0, 360, 200, -1)
    patch = ImagePatch(img, ulx=100, uly=150)

    # Test male pose
    (cx_m, cy_m), angle_m = pose_m.predict(patch)
    assert isinstance(cx_m, (float, int)) and isinstance(cy_m, (float, int))
    assert isinstance(angle_m, float)
    assert not np.isnan(angle_m) and not np.isinf(angle_m)
    # Check absolute coordinates offset
    assert cx_m > 100 and cy_m > 150

    # Test female pose
    (cx_f, cy_f), angle_f = pose_f.predict(patch)
    assert isinstance(angle_f, float)
    assert not np.isnan(angle_f) and not np.isinf(angle_f)

@pytest.mark.unit
def test_wing_predictor(predictors):
    wing_pred = predictors['wing']

    # Synthetic male fly patch with standard crop_size=400 (as used in male_fly_patch)
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(img, (200, 200), (50, 25), 0, 0, 360, 220, -1)
    # Add small wing display on side
    cv2.ellipse(img, (240, 180), (30, 10), 45, 0, 360, 180, -1)
    mask = np.full((400, 400), 255, dtype=np.uint8)
    patch = ImagePatch(img, mask=mask)

    w_right, w_left = wing_pred.predict(patch)
    assert isinstance(w_right, (float, np.floating))
    assert isinstance(w_left, (float, np.floating))
    assert not np.isnan(w_right) and not np.isnan(w_left)
    assert not np.isinf(w_right) and not np.isinf(w_left)

@pytest.mark.unit
def test_model_invalid_inputs(predictors):
    """Verify models raise appropriate ValueError on wrong feature dimensions."""
    is_fly = predictors['is_fly']
    # Decision tree expects 1 feature; passing 5 features should fail
    with pytest.raises(ValueError):
        is_fly.clf.predict(np.zeros((1, 5)))

    id_pred = predictors['id']
    # Id model expects 4 features; passing 2 should fail
    with pytest.raises(ValueError):
        id_pred.clf.predict(np.zeros((1, 2)))

    wing_pred = predictors['wing']
    # Wing model expects 3564 HOG features; passing 10 should fail
    with pytest.raises(ValueError):
        wing_pred.clf.predict(np.zeros((1, 10)))
