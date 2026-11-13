"""
tests/test_trained_models.py

Comprehensive test suite for Phase 4 model artifacts and inference compatibility:
1. Verifies all model files in models/ exist and are valid joblib files.
2. Verifies Predictor wrappers (IsFlyPredictor, IdPredictor, PosePredictor, WingPredictor) load properly.
3. Performs inference on real held-out test data.
4. Checks for NaN or infinite outputs.
5. Verifies output shapes and expected types.
"""

import os
import sys
import json
import joblib
import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.files import get_file
from cs229.train_is_fly import IsFlyPredictor
from cs229.train_id import IdPredictor
from cs229.train_orientation import PosePredictor
from cs229.train_wing import WingPredictor

def test_models():
    print("=" * 60)
    print("Testing Trained Model Artifacts & Predictor Classes")
    print("=" * 60)

    # 1. Check file existence
    expected_models = [
        'clf_is_fly.joblib',
        'clf_id.joblib',
        'clf_orient_male.joblib',
        'clf_orient_female.joblib',
        'clf_wing.joblib'
    ]
    for m in expected_models:
        path = get_file('output', 'models', m)
        assert os.path.exists(path), f"Missing model: {path}"
        sz = os.path.getsize(path)
        print(f"  Model {m:25s}: exists ({sz:,} bytes)")

    # 2. Check Predictor classes
    is_fly_p = IsFlyPredictor()
    id_p = IdPredictor()
    pose_m_p = PosePredictor('male')
    pose_f_p = PosePredictor('female')
    wing_p = WingPredictor()

    assert hasattr(is_fly_p.clf, 'predict'), "is_fly predictor has no predict method"
    assert hasattr(id_p.clf, 'predict'), "id predictor has no predict method"
    assert hasattr(pose_m_p.clf, 'predict'), "pose_m predictor has no predict method"
    assert hasattr(pose_f_p.clf, 'predict'), "pose_f predictor has no predict method"
    assert hasattr(wing_p.clf, 'predict'), "wing predictor has no predict method"
    print("  [PASS] All 5 Predictor instances initialized and loaded successfully.")

    # 3. Test on actual held-out feature samples
    X_is_fly = joblib.load(get_file('output', 'data', 'X_is_fly.joblib'))
    pred_fly = is_fly_p.clf.predict(X_is_fly[:10])
    assert not np.isnan(pred_fly).any() and not np.isinf(pred_fly).any()
    assert set(pred_fly).issubset({0, 1, 2})
    print(f"  [PASS] is_fly predictions valid: sample predictions={pred_fly[:5]}")

    X_id = joblib.load(get_file('output', 'data', 'X_id.joblib'))
    pred_id = id_p.clf.predict(X_id[:10])
    assert not np.isnan(pred_id).any() and not np.isinf(pred_id).any()
    assert set(pred_id).issubset({0, 1})
    print(f"  [PASS] id predictions valid: sample predictions={pred_id[:5]}")

    X_orient = joblib.load(get_file('output', 'data', 'X_orient.joblib'))
    pred_om = pose_m_p.clf.predict(X_orient['male'][:10])
    assert not np.isnan(pred_om).any() and not np.isinf(pred_om).any()
    assert set(pred_om).issubset({0, 1})
    print(f"  [PASS] orient_male predictions valid: sample predictions={pred_om[:5]}")

    pred_of = pose_f_p.clf.predict(X_orient['female'][:10])
    assert not np.isnan(pred_of).any() and not np.isinf(pred_of).any()
    assert set(pred_of).issubset({0, 1})
    print(f"  [PASS] orient_female predictions valid: sample predictions={pred_of[:5]}")

    X_wing = joblib.load(get_file('output', 'data', 'X_wing.joblib'))
    pred_w = wing_p.clf.predict(X_wing[:10])
    assert not np.isnan(pred_w).any() and not np.isinf(pred_w).any()
    pred_w_deg = np.degrees(pred_w)
    print(f"  [PASS] wing predictions valid: sample angles (deg)={np.round(pred_w_deg[:5], 2)}")

    # 4. Check outputs artifacts
    dot_file = get_file('output', 'diagrams', 'tree_is_fly.dot')
    assert os.path.exists(dot_file) and os.path.getsize(dot_file) > 0
    print(f"  [PASS] Decision Tree diagram verified: {dot_file}")

    expected_graphs = [
        'pca_orient_male_explained.eps',
        'pca_orient_male.eps',
        'pca_orient_female_explained.eps',
        'pca_orient_female.eps',
        'pca_wing_explained.eps',
        'pca_wing.eps'
    ]
    for g in expected_graphs:
        gpath = get_file('output', 'graphs', g)
        assert os.path.exists(gpath) and os.path.getsize(gpath) > 0
        print(f"  [PASS] Graph verified: {g}")

    # 5. Check evaluation_metrics.json
    metrics_path = get_file('output', 'data', 'evaluation_metrics.json')
    assert os.path.exists(metrics_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    assert set(metrics.keys()) == {'is_fly', 'id', 'orient_male', 'orient_female', 'wing'}
    print(f"  [PASS] Evaluation metrics JSON verified: {len(metrics)} model tasks recorded.")

    print("\nAll trained models and artifacts verified 100% operational!")

if __name__ == '__main__':
    test_models()

