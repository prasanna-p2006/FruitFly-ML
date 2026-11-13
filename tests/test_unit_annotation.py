"""
tests/test_unit_annotation.py

Unit tests for cs229.annotation module:
- Valid LabelMe annotation JSON loading
- Missing required JSON fields (e.g. imagePath, shapes, points)
- Missing referenced image file
- Malformed / invalid JSON syntax
- Missing or extra landmarks
- Sanity check warning behavior
- Feature query helper (check_features)
"""

import os
import json
import pytest
import numpy as np

from cs229.annotation import Annotation, project

@pytest.mark.unit
def test_project_mathematics():
    """Verify vector projection coefficient math."""
    a = (0, 0)
    b = (10, 0)
    c = (5, 0)
    assert project(a, b, c) == pytest.approx(0.5)

    c_outside = (15, 0)
    assert project(a, b, c_outside) == pytest.approx(1.5)

    c_negative = (-5, 0)
    assert project(a, b, c_negative) == pytest.approx(-0.5)

@pytest.mark.unit
def test_valid_annotation_loading(temp_dir):
    """Test loading a fully compliant LabelMe annotation JSON."""
    json_path = os.path.join(temp_dir, 'sample.json')
    anno_data = {
        "version": "4.5.6",
        "imagePath": "sample.bmp",
        "shapes": [
            {"label": "mp", "points": [[100.0, 100.0]]},
            {"label": "mh", "points": [[100.0, 80.0]]},
            {"label": "ma", "points": [[100.0, 120.0]]},
            {"label": "fp", "points": [[200.0, 200.0]]},
            {"label": "fh", "points": [[200.0, 180.0]]},
            {"label": "fa", "points": [[200.0, 220.0]]},
        ]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(anno_data, f)

    anno = Annotation(json_path)
    assert anno.json_path == json_path
    assert anno.image_path == os.path.join(temp_dir, 'sample.bmp')
    assert anno.count('mp') == 1
    assert anno.count('fp') == 1
    assert anno.has('mh') is True
    assert anno.has('mw') is False
    assert anno.count('mw') == 0
    assert anno.get('mp') == [[100.0, 100.0]]

@pytest.mark.unit
def test_annotation_check_features(temp_dir):
    """Verify check_features handles both present and missing feature keys without KeyError."""
    json_path = os.path.join(temp_dir, 'feat_test.json')
    anno_data = {
        "imagePath": "feat.bmp",
        "shapes": [
            {"label": "mp", "points": [[50.0, 50.0]]},
            {"label": "fp", "points": [[60.0, 60.0]]},
        ]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(anno_data, f)

    anno = Annotation(json_path)
    # Present key
    assert anno.check_features(mp=1, fp=1) is True
    assert anno.check_features(mp=2) is False
    # Missing key must return True when expected count is 0, False otherwise (no KeyError)
    assert anno.check_features(mw=0) is True
    assert anno.check_features(mw=2) is False

@pytest.mark.unit
def test_annotation_missing_keys(temp_dir):
    """Verify handling of missing top-level keys in JSON."""
    bad_json = os.path.join(temp_dir, 'missing_imgpath.json')
    with open(bad_json, 'w', encoding='utf-8') as f:
        json.dump({"shapes": []}, f)

    with pytest.raises(KeyError):
        Annotation(bad_json)

@pytest.mark.unit
def test_annotation_malformed_json(temp_dir):
    """Verify JSONDecodeError on corrupted JSON syntax."""
    bad_json = os.path.join(temp_dir, 'corrupt.json')
    with open(bad_json, 'w', encoding='utf-8') as f:
        f.write("{ invalid json content ...")

    with pytest.raises(json.JSONDecodeError):
        Annotation(bad_json)

@pytest.mark.unit
def test_annotation_missing_file():
    """Verify FileNotFoundError on non-existent JSON path."""
    with pytest.raises(FileNotFoundError):
        Annotation("non_existent_file_12345.json")

@pytest.mark.unit
def test_annotation_sanity_check_warnings(temp_dir, capsys):
    """Verify sanity_check produces warnings when fly count or landmarks violate constraints."""
    # Only 1 fly (missing fp)
    json_path = os.path.join(temp_dir, 'one_fly.json')
    anno_data = {
        "imagePath": "one_fly.bmp",
        "shapes": [{"label": "mp", "points": [[10.0, 10.0]]}]
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(anno_data, f)

    Annotation(json_path)
    captured = capsys.readouterr()
    assert "Must have one male and one female fly" in captured.out

