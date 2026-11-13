"""
tests/verify_deps_and_stats.py

1. Check dependency consistency between requirements.txt, .venv pip list, and runtime imports.
2. Confirm cv2.HOGDescriptor works.
3. Compute exact mathematical statistics of wing angles in y_wing.joblib.
"""

import os
import sys
import importlib.metadata
import joblib
import numpy as np

def check_dependencies():
    print("=" * 60)
    print("Dependency & Runtime Verification")
    print("=" * 60)
    
    # Read requirements.txt
    req_file = "requirements.txt"
    req_versions = {}
    with open(req_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "==" in line:
                    pkg, ver = line.split("==")
                    req_versions[pkg.lower()] = ver
    print(f"requirements.txt packages: {req_versions}")
    
    # Check installed packages in .venv
    installed_versions = {}
    for pkg in req_versions:
        try:
            ver = importlib.metadata.version(pkg)
            installed_versions[pkg] = ver
        except Exception as e:
            installed_versions[pkg] = f"Error: {e}"
            
    print(f"Installed versions in .venv: {installed_versions}")
    
    # Verify imports and runtime versions
    import cv2
    import numpy
    import sklearn
    import matplotlib
    import joblib as jb
    import imutils
    import tqdm
    
    runtime_versions = {
        "opencv-python": cv2.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": jb.__version__,
        "imutils": imutils.__version__,
        "tqdm": tqdm.__version__
    }
    print(f"Runtime imported versions: {runtime_versions}")
    
    # Check agreements
    all_match = True
    for pkg, req_v in req_versions.items():
        inst_v = installed_versions.get(pkg)
        runt_v = runtime_versions.get(pkg)
        match = (req_v == inst_v == runt_v)
        print(f"  {pkg:15s}: req={req_v:12s} inst={inst_v:12s} runtime={runt_v:12s} -> Match={match}")
        if not match:
            all_match = False
            
    print(f"All pinned versions agree: {all_match}")
    
    # Verify cv2.HOGDescriptor
    print("\nTesting cv2.HOGDescriptor:")
    assert hasattr(cv2, 'HOGDescriptor'), "cv2 has NO attribute HOGDescriptor!"
    hog = cv2.HOGDescriptor((80, 96), (16, 16), (8, 8), (8, 8), 9)
    test_patch = np.zeros((96, 80), dtype=np.uint8)
    feat = hog.compute(test_patch).flatten()
    print(f"  cv2.HOGDescriptor instantiated successfully!")
    print(f"  Sample feature computed on (96, 80) patch: shape={feat.shape}, dtype={feat.dtype}")
    assert feat.shape[0] == 3564, f"Expected 3564 features, got {feat.shape[0]}"
    print("  [PASS] cv2.HOGDescriptor verified and operational.")

def check_wing_stats():
    print("\n" + "=" * 60)
    print("Wing Angle Mathematical Verification")
    print("=" * 60)
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    y_path = os.path.join(repo_root, "outputs", "data", "y_wing.joblib")
    assert os.path.exists(y_path), f"File not found: {y_path}"
    
    y = joblib.load(y_path)
    print(f"Total samples: {len(y)}")
    
    # Radians
    r_min = float(np.min(y))
    r_max = float(np.max(y))
    r_mean = float(np.mean(y))
    r_std = float(np.std(y))
    
    # Degrees
    deg = np.degrees(y)
    d_min = float(np.min(deg))
    d_max = float(np.max(deg))
    d_mean = float(np.mean(deg))
    d_std = float(np.std(deg))
    
    print("\nIn Radians:")
    print(f"  Min:  {r_min:.6f} rad")
    print(f"  Max:  {r_max:.6f} rad")
    print(f"  Mean: {r_mean:.6f} rad")
    print(f"  Std:  {r_std:.6f} rad")
    
    print("\nIn Degrees:")
    print(f"  Min:  {d_min:.4f}°")
    print(f"  Max:  {d_max:.4f}°")
    print(f"  Mean: {d_mean:.4f}°")
    print(f"  Std:  {d_std:.4f}°")
    
    # Theoretical check: formula is arctan(|x| / |y|) >= 0 and < pi/2 (90 deg)
    theoretical_min_rad = 0.0
    theoretical_max_rad = np.pi / 2  # 1.5707963 rad = 90 deg
    
    out_of_bounds = np.sum((y < theoretical_min_rad) | (y >= theoretical_max_rad))
    print(f"\nSamples outside theoretical [0, pi/2) rad ([0°, 90°)): {out_of_bounds}")
    assert out_of_bounds == 0, f"Found {out_of_bounds} samples out of bounds!"
    print("  [PASS] All wing angle values strictly fall within theoretical valid range [0, 90°).")

if __name__ == '__main__':
    check_dependencies()
    check_wing_stats()

