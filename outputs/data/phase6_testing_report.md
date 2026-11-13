# Phase 6 Final Report: Comprehensive Testing, Refactoring & Code Quality

**Project Title:** Real-Time Detailed Video Analysis of Fruit Flies  
**Project Root:** `D:\ML-Projects\FruitFly\FruitFly-ML`  
**Date:** November 1, 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

Phase 6 focused on establishing automated software engineering rigor, comprehensive unit and integration testing, code quality hardening, and empirical verification for the `FruitFly-ML` repository. 

Building on the verified model artifacts from Phase 4 and the 2,918-frame video inference outputs from Phase 5, Phase 6 accomplished:
1. **Automated Testing Framework:** Deployed `pytest 9.2.0` and `pytest-cov 7.1.0` configured for headless execution (`Agg` backend), deterministic fixtures, and automated XML/JSON reporting.
2. **Defensive Code Hardening:** Implemented 7 safe, non-breaking bug fixes across core modules (`annotation.py`, `contour.py`, `patch.py`, `files.py`, `display.py`) to eliminate runtime vulnerabilities (`ZeroDivisionError`, `KeyError`, `ValueError`, unclosed file descriptors, and bare exceptions).
3. **Automated Test Suite:** Created 35 automated test cases spanning unit tests, end-to-end pipeline integration tests, and defensive edge-case handlers. All 35 tests pass with a 100% success rate.
4. **Code Coverage:** Generated module-by-module line coverage profiles showing an overall codebase coverage of 46.46% (768 / 1,653 statements). Primary execution pipeline and geometry modules achieve 82%–100% coverage (`image.py` 100%, `patch.py` 86%, `files.py` 84%, `video_pipeline.py` 82%), while contour and inference wrapper modules range from 51% to 69% (`annotation.py` 69%, `contour.py` 65%, `train_id.py` 65%, `train_is_fly.py` 60%, `train_wing.py` 52%, `train_orientation.py` 51%) because their offline training routines were preserved uncalled. Interactive and legacy GUI modules (`demo.py`, `display.py`) remain at 0% for headless testing.
5. **Multi-Stream Video & Telemetry Audit:** Re-verified all 5 output videos and CSV trajectories with 0 missing frames, 0 NaN values, and 100% 1:1 synchronization across 2,918 processed frames.

---

## 2. Code Inspection & Bug Fixes

A thorough inspection of `src/cs229/` identified several potential runtime failure points when operating on real-world noisy video frames, corrupted annotations, or headless environments. All fixes preserve the exact mathematical formulas and algorithm logic of the original Stanford CS229 implementation while adding safe guards.

### Summary of Discovered Bugs and Applied Fixes

| File | Function / Scope | Bug Description | Fix Applied | Result |
| :--- | :--- | :--- | :--- | :--- |
| `src/cs229/annotation.py` | `check_features()` | Directly accessed `self.labels[key]` without checking existence, raising `KeyError` if a requested feature was absent. | Updated to query `self.count(key) != val`, safely returning 0 when key is absent. | Prevents crash on partial landmark queries. |
| `src/cs229/annotation.py` | `Annotation.__init__()` | Used unmanaged `open()` without specifying encoding, risking handle leaks and OS-specific character encoding errors. | Wrapped in `with open(f, 'r', encoding='utf-8') as fh:`. | Safe resource management and UTF-8 encoding compliance. |
| `src/cs229/contour.py` | `largest_contour()` | Called `max(contours, key=cv2.contourArea)` directly. An empty list raised `ValueError: max() arg is an empty sequence`. | Added check: `if not contours: return None`. | Graceful fallback when no flies are detected in frame. |
| `src/cs229/contour.py` | `threshold_core()`, `threshold_wings()` | Passed `mask` directly to `cv2.bitwise_and`. When `mask=None`, OpenCV crashed with invalid argument type. | Added `if mask is not None:` branching before bitwise operations. | Supports standalone thresholding without precomputed masks. |
| `src/cs229/patch.py` | `moments_to_angle()`, `moments_to_center()` | Divided by raw spatial moment $m_{00}$ without checking if $m_{00} == 0$, causing `ZeroDivisionError` on blank patches. | Added defensive guards: `if m['m00'] == 0: return 0.0` and `return (0.0, 0.0)`. | Safe processing of edge-boundary or fully occluded patches. |
| `src/cs229/patch.py` | `estimate_axes()`, `estimate_aspect_ratio()` | Derived central moments directly from raw moments assuming $m_{00} > 0$. | Added guards against non-positive moments and zero-division in aspect ratio calculation. | Eliminates math domain errors on abnormal contours. |
| `src/cs229/files.py` | `read_video()` | Opened video captures without checking file existence, capture initialization, or valid FPS. | Added `os.path.exists()` check, `cap.isOpened()` assertion, and fallback to 35.0 FPS if metadata FPS is invalid. | Clear, actionable `FileNotFoundError` and `RuntimeError`. |
| `src/cs229/display.py` | `MouseData.last_click()` | Used bare `except:` block to catch list indexing errors. | Replaced with explicit condition: `if self.clicks: return self.clicks[-1] else: return None`. | Complies with PEP 8 and prevents masking critical exceptions. |

### Representative Code Changes

#### 1. Zero-Moment Division Guard in `src/cs229/patch.py`
```python
# Before
def moments_to_angle(m):
    u20 = m['m20']/m['m00'] - (m['m10']/m['m00'])**2
    u02 = m['m02']/m['m00'] - (m['m01']/m['m00'])**2
    u11 = m['m11']/m['m00'] - (m['m10']/m['m00'])*(m['m01']/m['m00'])
    return 0.5 * np.arctan2(2*u11, u20-u02)

# After
def moments_to_angle(m):
    if m['m00'] == 0:
        return 0.0
    u20 = m['m20']/m['m00'] - (m['m10']/m['m00'])**2
    u02 = m['m02']/m['m00'] - (m['m01']/m['m00'])**2
    u11 = m['m11']/m['m00'] - (m['m10']/m['m00'])*(m['m01']/m['m00'])
    return 0.5 * np.arctan2(2*u11, u20-u02)
```

#### 2. Safe Feature Checking in `src/cs229/annotation.py`
```python
# Before
def check_features(self, **kwargs):
    for key, val in kwargs.items():
        if len(self.labels[key]) != val:
            return False
    return True

# After
def check_features(self, **kwargs):
    for key, val in kwargs.items():
        if self.count(key) != val:
            return False
    return True
```

---

## 3. Test Suite Architecture & Implementation

A modern test structure was created under `tests/` using standard pytest conventions.

```text
FruitFly-ML/
├── pytest.ini                         # Pytest configuration with markers & testpaths
└── tests/
    ├── conftest.py                    # Global fixtures (headless Agg, mock arena frames, temp dirs)
    ├── test_unit_annotation.py        # 7 Unit tests: JSON parsing, math, landmark sanity checks
    ├── test_unit_image_contour.py     # 8 Unit tests: thresholding, contour finding, patch moments
    ├── test_unit_models.py            # 5 Unit tests: DecisionTree, LogisticRegression, PCA, Predictors
    ├── test_unit_video.py             # 5 Unit tests: video I/O, properties, CSV schema, cleanup
    ├── test_integration_pipeline.py   # 2 Integration tests: end-to-end small batch, error handling
    ├── test_edge_cases.py             # 7 Edge tests: merged flies, missing landmarks, 0-byte files
    └── test_trained_models.py         # 1 Integration test: verified artifact loading on disk
```

### Complete Test Inventory (35 Tests)

1. `tests/test_unit_annotation.py::test_project_mathematics`: Validates vector projection normalization equation.
2. `tests/test_unit_annotation.py::test_valid_annotation_loading`: Validates compliant LabelMe JSON loading and coordinate parsing.
3. `tests/test_unit_annotation.py::test_annotation_check_features`: Verifies feature queries on present, missing, and zero-count keys.
4. `tests/test_unit_annotation.py::test_annotation_missing_keys`: Verifies error handling when essential JSON keys are omitted.
5. `tests/test_unit_annotation.py::test_annotation_malformed_json`: Validates `json.JSONDecodeError` on syntax errors.
6. `tests/test_unit_annotation.py::test_annotation_missing_file`: Validates `FileNotFoundError` on non-existent JSON paths.
7. `tests/test_unit_annotation.py::test_annotation_sanity_check_warnings`: Validates constraint checking (e.g. fly count, landmark pairs).
8. `tests/test_unit_image_contour.py::test_bound_point`: Validates integer point coordinate clipping to image boundaries.
9. `tests/test_unit_image_contour.py::test_img_to_mask`: Validates circular arena mask generation and dimension assertion.
10. `tests/test_unit_image_contour.py::test_threshold_core_with_and_without_mask`: Verifies adaptive core thresholding with optional mask.
11. `tests/test_unit_image_contour.py::test_threshold_wings_with_and_without_mask`: Verifies wing thresholding and morphological filtering.
12. `tests/test_unit_image_contour.py::test_find_contours_and_largest`: Validates OpenCV 5 compatible contour retrieval and empty-contour guard.
13. `tests/test_unit_image_contour.py::test_contour_roi_and_in_contour`: Tests point-in-polygon and bounding ROI extraction.
14. `tests/test_unit_image_contour.py::test_crop_to_contour_and_image_patch`: Verifies image patch extraction, centroid translation, and rotation.
15. `tests/test_unit_image_contour.py::test_image_patch_zero_moments_safety`: Confirms zero-intensity patches do not raise `ZeroDivisionError`.
16. `tests/test_unit_models.py::test_is_fly_predictor`: Verifies Decision Tree split boundaries on contour area (`neither`, `one`, `both`).
17. `tests/test_unit_models.py::test_id_predictor`: Verifies sex identification classifier outputs on feature tuples.
18. `tests/test_unit_models.py::test_pose_predictor`: Verifies HOG + PCA-15 + Logistic Regression orientation predictions.
19. `tests/test_unit_models.py::test_wing_predictor`: Verifies HOG + PCA-40 + Linear Regression angle estimations.
20. `tests/test_unit_models.py::test_model_invalid_inputs`: Asserts estimators throw `ValueError` on mismatched feature dimensions.
21. `tests/test_unit_video.py::test_cap_props`: Validates `CapProps` frame dimensions, FPS, and millisecond frame timing.
22. `tests/test_unit_video.py::test_read_video_missing_file`: Asserts `FileNotFoundError` when video file does not exist.
23. `tests/test_unit_video.py::test_read_synthetic_video`: Validates sequential reading of video frames via OpenCV.
24. `tests/test_unit_video.py::test_write_video_and_cleanup`: Tests `VideoWriter` initialization, frame output, and clean resource release.
25. `tests/test_unit_video.py::test_csv_telemetry_schema`: Validates complete 10-column trajectory CSV schema and header compliance.
26. `tests/test_integration_pipeline.py::test_full_pipeline_small_batch`: Runs end-to-end `process_video` on 5 frames, validating video and CSV generation.
27. `tests/test_integration_pipeline.py::test_pipeline_resource_cleanup_on_error`: Injects mid-stream fault and asserts writer and capture release.
28. `tests/test_edge_cases.py::test_overlapping_flies_detection`: Tests classification of merged contours into `both` with null telemetry.
29. `tests/test_edge_cases.py::test_missing_landmarks_warning`: Tests warning generation when landmarks are asymmetric (e.g. head without abdomen).
30. `tests/test_edge_cases.py::test_invalid_image_dimensions`: Confirms `img_to_mask` enforces square chamber dimensions ($rows == cols$).
31. `tests/test_edge_cases.py::test_empty_frame_handling`: Confirms blank frames return `NO_FLIES_DETECTED` with clean null output.
32. `tests/test_edge_cases.py::test_corrupt_or_zero_byte_video`: Asserts `read_video` rejects 0-byte corrupt files with `RuntimeError`.
33. `tests/test_edge_cases.py::test_unexpected_model_feature_dimensions`: Validates dimensional consistency checks in scikit-learn models.
34. `tests/test_edge_cases.py::test_automatic_output_dir_creation`: Verifies `get_dir()` creates missing parent directories on demand.
35. `tests/test_trained_models.py::test_models`: Performs disk verification on all 5 serialized `.joblib` model artifacts.

---

## 4. Test Execution & Verification Results

### Automated Pytest Suite
```text
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\ML-Projects\FruitFly\FruitFly-ML
configfile: pytest.ini
testpaths: tests
plugins: cov-7.1.0
collected 35 items

35 passed in 6.97s (100% success rate)
```

Machine-readable test artifacts generated:
- **JUnit XML:** `outputs/data/pytest_results.xml`
- **Coverage JSON:** `outputs/data/coverage.json`

### Verification Script Execution Summary

| Script | Purpose | Result | Key Metrics Verified |
| :--- | :--- | :---: | :--- |
| `tests/test_annotations.py` | Verify 326 LabelMe annotations and referenced images | **PASSED** | 326 JSON files parsed; 326 / 326 images verified; 0 missing; 0 invalid. |
| `tests/test_data_arrays.py` | Validate training/test feature arrays in `outputs/data/` | **PASSED** | `X_is_fly` (1012, 1); `X_id` (564, 4); `X_orient` (478 male, 348 female); `X_wing` (472, 3564); 0 NaNs; 0 Infs. |
| `tests/test_trained_models.py` | Verify 5 saved models in `models/` and Predictor wrappers | **PASSED** | All 5 `.joblib` files verified; valid predictions and angles produced; diagnostic plots verified. |
| `tests/verify_deps_and_stats.py` | Verify runtime dependencies and mathematical ranges | **PASSED** | `cv2.HOGDescriptor` functional; wing angles strictly within $[0^\circ, 90^\circ)$ (mean $21.27^\circ$). |
| `tests/verify_video_integrity.py` | Verify stream integrity across all 5 test videos | **PASSED** | 2,918 source frames $\to$ 2,918 output frames $\to$ 2,918 CSV rows (100% 1:1 match). |

---

## 5. Code Coverage Analysis

Measured via `pytest --cov=src/cs229 --cov-report=term-missing`:

```text
Name                                 Stmts   Miss  Cover   Missing
------------------------------------------------------------------
src\cs229\image.py                      13      0   100%   
src\cs229\patch.py                     145     21    86%   20-24, 159-162, 222-225, 236-255, 258
src\cs229\files.py                      58      9    84%   27, 76-77, 81, 85, 95-100
src\cs229\video_pipeline.py            256     47    82%   29, 87-91, 96-97, 179, 183-184, 224-225, 302-325, 367-369, 379, 424-438, 441
src\cs229\__init__.py                    5      1    80%   7
src\cs229\annotation.py                101     31    69%   88-89, 102-103, 111-112, 116-131, 135-142, 155-160, 164-165, 168
src\cs229\contour.py                    80     28    65%   36-48, 61, 79-82, 107-110, 126-144, 147
src\cs229\train_id.py                   31     11    65%   24-29, 32-35, 38-41, 44
src\cs229\train_is_fly.py               30     12    60%   21-26, 29-35, 38-41, 44
src\cs229\train_wing.py                 58     28    52%   39-60, 63-71, 74-78, 81-84, 87
src\cs229\train_orientation.py          68     33    51%   45, 52-76, 79-88, 91-94, 97-102, 105
src\cs229\load_data_is_fly.py           42     21    50%   28-53, 56-59, 62
src\cs229\load_data_wing.py            127     75    41%   26-27, 53, 88-199, 202-205, 208
src\cs229\load_data_id.py               69     41    41%   39-93, 96-99, 102
src\cs229\load_data_orientation.py      89     56    37%   43-129, 132-135, 138
src\cs229\util.py                       42     32    24%   9, 13-19, 23-38, 41-51, 54-67
src\cs229\timing.py                     17     17     0%   1-24
src\cs229\train_all.py                 255    255     0%   31-476
src\cs229\display.py                    36     36     0%   1-49
src\cs229\demo.py                      131    131     0%   1-197
------------------------------------------------------------------
TOTAL                                 1653    885    46%
```

### Analysis of Uncovered Lines
1. **Interactive UI Modules (`demo.py`, `display.py` - 0%):** These scripts implement OpenCV GUI windows (`cv2.imshow`, `cv2.setMouseCallback`, `cv2.createTrackbar`). Because tests run in headless automated mode without desktop windowing, interactive loops are deliberately excluded from automated execution.
2. **Training Orchestration Scripts (`train_all.py` - 0%, `train_*.py` - 51%–65%):** These files contain the full offline training loops and cross-validation routines used in Phase 4. Their predictor classes (`IsFlyPredictor`, `IdPredictor`, `PosePredictor`, `WingPredictor`) are 100% covered by unit tests, but re-training the full models during test execution was deliberately avoided to maintain model stability.
3. **Data Preprocessing CLI Scripts (`load_data_*.py` - 37%–50%):** The `main()` sections that rebuild `.joblib` datasets from raw disk images are not invoked during inference tests; feature transformation helpers (`make_features`, `make_hog`, `male_fly_patch`) are fully exercised.
4. **Core Operational Pipeline & Vision Modules Breakdown:**
   - `image.py`: **100%** (13/13 statements)
   - `patch.py`: **86%** (124/145 statements; uncovered lines are standalone visualization and diagnostic main)
   - `files.py`: **84%** (49/58 statements; path resolution and video capture verified)
   - `video_pipeline.py`: **82%** (209/256 statements; complete video inference, HUD rendering, and telemetry logging covered)
   - `annotation.py`: **69%** (70/101 statements; annotation loading, coordinate parsing, and geometry covered; uncovered lines are standalone CLI print routines and rare landmark warning branches)
   - `contour.py`: **65%** (52/80 statements; adaptive core/wing thresholding and contour extraction covered; uncovered lines are standalone CLI main and label utilities)
   - **Predictor Modules (`train_*.py`):** **51%–65%** (Predictor inference classes `IsFlyPredictor`, `IdPredictor`, `PosePredictor`, `WingPredictor` are 100% covered; full dataset training functions were intentionally left uncalled to preserve trained models)

---

## 6. End-to-End Pipeline & Integrity Verification

The full 5-video verification confirmed absolute stream and telemetry consistency:

```text
================================================================================
Video      | Meta Frames | Decoded (FF) | MSMF Raw  | Output Frames | CSV Rows | Src Dur  | Out Dur  | Diff 
------------------------------------------------------------------------------------------------------------
test1.mp4  |         368 |          351 |       369 |           368 |      368 |  10.52s  |  10.51s  |     0
test2.mp4  |         413 |          352 |       413 |           413 |      413 |  11.80s  |  11.80s  |     0
test3.mp4  |         762 |          736 |       765 |           762 |      762 |  21.77s  |  21.77s  |     0
test4.mp4  |         680 |          632 |       680 |           680 |      680 |  19.43s  |  19.43s  |     0
test5.mp4  |         695 |          632 |       696 |           695 |      695 |  19.86s  |  19.86s  |     0
================================================================================
Total      |       2,918 |        2,703 |     2,923 |         2,918 |    2,918 |  83.38s  |  83.37s  |     0
```

- **Zero Frame Loss:** Every declared frame from the MP4 container sample tables ($stsz$) is decoded, processed, and written to the output MP4 and CSV.
- **Deduplication:** The Media Foundation padding artifact ($+1$ to $+3$ repeated frames at EOF) is cleanly filtered by the pipeline's duplicate detection filter.
- **CSV Data Integrity:** 2,918 data rows generated with strictly sequential frame numbers ($1 \dots N$), correct 35 FPS timestamps, and zero invalid/corrupted records.

---

## 7. Known Behaviors & Operational Notes

1. **Merged Contours during Physical Interaction:** In classical computer vision thresholding, when two flies come into close physical contact, their pixels merge into one contiguous contour. The pipeline classifies this contour as `both` (touching state) and logs `OVERLAPPING_FLIES` with null telemetry coordinates. This is by design to prevent false landmark predictions.
2. **Headless Execution Environment:** All automated test suites run using Matplotlib's `Agg` non-interactive backend. Any future extensions to the interactive `demo.py` tool should maintain headless compatibility flags.
3. **Platform-Specific Video Backends:** On Windows, OpenCV's default FFmpeg backend stops decoding before trailer GOPs on these files due to timestamp discontinuities. The production pipeline explicitly utilizes `cv2.CAP_MSMF` with timestamp-based frame validation to guarantee 100% complete decoding.

---

## 8. Final Certification

Phase 6 has verified that the `FruitFly-ML` repository satisfies all code quality, testing, and reliability criteria:
- **Automated Tests:** 35 passing automated tests (100% pass rate).
- **Regression Scripts:** 5 standalone verification scripts passing.
- **Bug Fixes:** 7 defensive bug fixes applied without altering original Stanford CS229 algorithms or model weights.
- **Telemetry & Video Verification:** 2,918 / 2,918 frames verified with 0 dropped frames and 0 data discrepancies.
- **Artifacts:** Full test reports and coverage profiles committed under `outputs/data/`.

