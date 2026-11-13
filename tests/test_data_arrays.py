"""
tests/test_data_arrays.py

Validation test suite for generated feature and label datasets:
1. is_fly dataset (X_is_fly.joblib, y_is_fly.joblib)
2. id dataset (X_id.joblib, y_id.joblib)
3. orientation dataset (X_orient.joblib, y_orient.joblib)
4. wing dataset (X_wing.joblib, y_wing.joblib)

Validates:
- Correct dimensions & shapes
- Non-empty datasets
- No unexpected NaN or infinite values
- Label types and valid ranges
- Sample-feature alignment
"""

import os
import sys
import joblib
import numpy as np

# Ensure src is on sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.files import get_file

def validate_is_fly():
    print("=" * 60)
    print("Validating: is_fly (FlyCount)")
    print("=" * 60)
    x_path = get_file('output', 'data', 'X_is_fly.joblib')
    y_path = get_file('output', 'data', 'y_is_fly.joblib')
    
    assert os.path.exists(x_path), f"File not found: {x_path}"
    assert os.path.exists(y_path), f"File not found: {y_path}"
    
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    
    assert len(X) > 0, "X is empty!"
    assert len(X) == len(y), f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}"
    assert not np.isnan(X).any(), "X contains NaN values!"
    assert not np.isinf(X).any(), "X contains infinite values!"
    assert not np.isnan(y).any(), "y contains NaN values!"
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    assert set(unique_labels).issubset({0, 1, 2}), f"Unexpected labels in y: {unique_labels}"
    print("  [PASS] is_fly validation passed.")
    return len(X)

def validate_id():
    print("=" * 60)
    print("Validating: id (Male/Female sex identification)")
    print("=" * 60)
    x_path = get_file('output', 'data', 'X_id.joblib')
    y_path = get_file('output', 'data', 'y_id.joblib')
    
    assert os.path.exists(x_path), f"File not found: {x_path}"
    assert os.path.exists(y_path), f"File not found: {y_path}"
    
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    
    assert len(X) > 0, "X is empty!"
    assert len(X) == len(y), f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}"
    assert X.shape[1] == 4, f"Expected 4 features (area_1, area_2, aspect_1, aspect_2), got {X.shape[1]}"
    assert not np.isnan(X).any(), "X contains NaN values!"
    assert not np.isinf(X).any(), "X contains infinite values!"
    assert not np.isnan(y).any(), "y contains NaN values!"
    
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
    assert set(unique_labels).issubset({0, 1}), f"Unexpected labels in y: {unique_labels}"
    print("  [PASS] id validation passed.")
    return len(X)

def validate_orient():
    print("=" * 60)
    print("Validating: orientation (Male/Female body orientation)")
    print("=" * 60)
    x_path = get_file('output', 'data', 'X_orient.joblib')
    y_path = get_file('output', 'data', 'y_orient.joblib')
    
    assert os.path.exists(x_path), f"File not found: {x_path}"
    assert os.path.exists(y_path), f"File not found: {y_path}"
    
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    
    assert isinstance(X, dict) and isinstance(y, dict), "X and y must be dictionaries with 'male' and 'female' keys"
    assert 'male' in X and 'female' in X, "Missing 'male' or 'female' in X"
    assert 'male' in y and 'female' in y, "Missing 'male' or 'female' in y"
    
    for sex in ['male', 'female']:
        X_sub = X[sex]
        y_sub = y[sex]
        print(f"  [{sex}] X shape: {X_sub.shape}, dtype: {X_sub.dtype}")
        print(f"  [{sex}] y shape: {y_sub.shape}, dtype: {y_sub.dtype}")
        
        assert len(X_sub) > 0, f"X[{sex}] is empty!"
        assert len(X_sub) == len(y_sub), f"Length mismatch for {sex}: len(X)={len(X_sub)} != len(y)={len(y_sub)}"
        assert not np.isnan(X_sub).any(), f"X[{sex}] contains NaN values!"
        assert not np.isinf(X_sub).any(), f"X[{sex}] contains infinite values!"
        assert not np.isnan(y_sub).any(), f"y[{sex}] contains NaN values!"
        
        unique_labels, counts = np.unique(y_sub, return_counts=True)
        print(f"  [{sex}] Class distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
        assert set(unique_labels).issubset({0, 1}), f"Unexpected labels in y[{sex}]: {unique_labels}"
        assert X_sub.shape[0] >= 15, f"Sample count ({X_sub.shape[0]}) must exceed 15 PCA components!"
        
    print("  [PASS] orientation validation passed.")
    return len(X['male']) + len(X['female'])

def validate_wing():
    print("=" * 60)
    print("Validating: wing (Male wing angle estimation)")
    print("=" * 60)
    x_path = get_file('output', 'data', 'X_wing.joblib')
    y_path = get_file('output', 'data', 'y_wing.joblib')
    
    assert os.path.exists(x_path), f"File not found: {x_path}"
    assert os.path.exists(y_path), f"File not found: {y_path}"
    
    X = joblib.load(x_path)
    y = joblib.load(y_path)
    
    print(f"  X shape: {X.shape}, dtype: {X.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    
    assert len(X) > 0, "X is empty!"
    assert len(X) == len(y), f"Length mismatch: len(X)={len(X)} != len(y)={len(y)}"
    assert not np.isnan(X).any(), "X contains NaN values!"
    assert not np.isinf(X).any(), "X contains infinite values!"
    assert not np.isnan(y).any(), "y contains NaN values!"
    assert not np.isinf(y).any(), "y contains infinite values!"
    
    deg = np.degrees(y)
    print(f"  Wing angles: min={deg.min():.2f} deg, max={deg.max():.2f} deg, mean={deg.mean():.2f} deg, std={deg.std():.2f} deg")
    assert 0 <= deg.min() and deg.max() <= 180, f"Angle out of bounds: min={deg.min()}, max={deg.max()}"
    assert X.shape[0] >= 40, f"Sample count ({X.shape[0]}) must exceed 40 PCA components!"
    
    eps_path = get_file('output', 'graphs', 'labels_wing.eps')
    print(f"  Checking wing distribution graph: {eps_path} (exists={os.path.exists(eps_path)})")
    
    print("  [PASS] wing validation passed.")
    return len(X)

def main():
    print("Starting full validation of generated feature/label arrays...\n")
    validate_is_fly()
    validate_id()
    validate_orient()
    if os.path.exists(get_file('output', 'data', 'X_wing.joblib')):
        validate_wing()
    else:
        print("\nNote: wing dataset not yet generated.")
    print("\nAll available datasets validated successfully!")

if __name__ == '__main__':
    main()
