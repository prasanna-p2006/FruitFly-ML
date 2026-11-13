"""
tests/conftest.py

Global pytest configuration and fixtures for FruitFly-ML test suite.
Enforces non-interactive headless Matplotlib Agg backend and manages test paths.
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
import cv2

# 1. Enforce headless Matplotlib before any modules import pyplot
import matplotlib
matplotlib.use('Agg')
os.environ['MPLBACKEND'] = 'Agg'
# Ensure matplotlib config dir is writable to avoid WinError 5 in AppData
mpl_temp = os.path.join(tempfile.gettempdir(), 'mpl_test_cache')
os.makedirs(mpl_temp, exist_ok=True)
os.environ['MPLCONFIGDIR'] = mpl_temp

# 2. Add repo root and src/ to sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

@pytest.fixture
def temp_dir():
    """Provides a fresh isolated temporary directory for test output artifacts."""
    d = tempfile.mkdtemp(prefix='fruitfly_test_')
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def synthetic_arena_frame():
    """Creates a synthetic 500x500 circular arena image with two small dark fly-like ellipses."""
    img = np.full((500, 500), 230, dtype=np.uint8)
    # Mask background outside circle
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 240, 255, -1)
    img[mask == 0] = 50
    # Fly 1 (male-like: smaller core, at (200, 200))
    cv2.ellipse(img, (200, 200), (12, 6), 45, 0, 360, 30, -1)
    # Fly 2 (female-like: larger core, at (300, 300))
    cv2.ellipse(img, (300, 300), (18, 9), -30, 0, 360, 30, -1)
    return img

