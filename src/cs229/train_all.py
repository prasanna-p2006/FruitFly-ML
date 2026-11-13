"""
src/cs229/train_all.py

Comprehensive training, evaluation, and serialization pipeline for FruitFly-ML.

Implements:
1. Leakage-safe grouped train/test split (grouping by source frame/fly).
2. Side-by-side comparison with original unconstrained random split.
3. Strict fitting of scalers and PCA transforms ONLY on training sets.
4. Comprehensive metric calculation:
   - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix
   - Regression: MAE (deg), RMSE (deg), Residual Std Sigma (deg & rad), R^2
   - Execution benchmarks: Training duration, Inference latency (ms/sample)
5. Model artifact serialization into models/:
   - clf_is_fly.joblib
   - clf_id.joblib
   - clf_orient_male.joblib
   - clf_orient_female.joblib
   - clf_wing.joblib
6. Visualizations and diagrams:
   - outputs/diagrams/tree_is_fly.dot
   - outputs/graphs/pca_orient_male_explained.eps, pca_orient_male.eps
   - outputs/graphs/pca_orient_female_explained.eps, pca_orient_female.eps
   - outputs/graphs/pca_wing_explained.eps, pca_wing.eps
7. Evaluation artifacts:
   - outputs/data/evaluation_metrics.json
   - outputs/data/test_predictions.joblib
8. Validation of loaded models for Phase 5 demo.py integration.
"""

import os
import sys
import time
import json
import cv2
import joblib
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

# Ensure cs229 package is accessible
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_dir = os.path.join(repo_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from cs229.files import get_file
from cs229.load_data_is_fly import CATEGORIES as IS_FLY_CATS, FEATURES as IS_FLY_FEATS
from cs229.load_data_id import CATEGORIES as ID_CATS
from cs229.load_data_orientation import CATEGORIES as ORIENT_CATS

RANDOM_STATE = 42

def plot_pca_orientation(pca, X_train_pca, y_train, sex):
    """Plot PCA explained variance and first two components for orientation."""
    eps_var = get_file('output', 'graphs', f'pca_orient_{sex}_explained.eps')
    eps_scat = get_file('output', 'graphs', f'pca_orient_{sex}.eps')

    # Cumulative explained variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of PCA components')
    plt.ylabel('Total explained variance ratio')
    plt.grid(True)
    plt.savefig(eps_var, bbox_inches='tight')
    plt.close()

    # 2D scatter plot
    plt.figure()
    for l, c, m in zip(range(2), ('blue', 'red'), ('o', 'x')):
        mask = (y_train == l)
        plt.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], color=c, label=ORIENT_CATS[l], alpha=0.6, marker=m)
    plt.legend(loc='upper right')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True)
    plt.savefig(eps_scat, bbox_inches='tight')
    plt.close()

def plot_pca_wing(pca, X_train_pca, y_train_deg):
    """Plot PCA explained variance and first component vs wing angle."""
    eps_var = get_file('output', 'graphs', 'pca_wing_explained.eps')
    eps_scat = get_file('output', 'graphs', 'pca_wing.eps')

    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of PCA components')
    plt.ylabel('Total explained variance ratio')
    plt.grid(True)
    plt.savefig(eps_var, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(X_train_pca[:, 0], y_train_deg, alpha=0.6)
    plt.xlabel('PCA Component 1')
    plt.ylabel('Wing angle (deg)')
    plt.grid(True)
    plt.savefig(eps_scat, bbox_inches='tight')
    plt.close()

def run_training_and_evaluation():
    print("=" * 80)
    print("PHASE 4: FRUITFLY-ML COMPREHENSIVE TRAINING & EVALUATION")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading serialized feature & label arrays...")
    X_is_fly = joblib.load(get_file('output', 'data', 'X_is_fly.joblib'))
    y_is_fly = joblib.load(get_file('output', 'data', 'y_is_fly.joblib'))
    groups_is_fly = joblib.load(get_file('output', 'data', 'groups_is_fly.joblib'))

    X_id = joblib.load(get_file('output', 'data', 'X_id.joblib'))
    y_id = joblib.load(get_file('output', 'data', 'y_id.joblib'))
    groups_id = np.repeat(np.arange(len(y_id) // 2), 2)

    X_orient = joblib.load(get_file('output', 'data', 'X_orient.joblib'))
    y_orient = joblib.load(get_file('output', 'data', 'y_orient.joblib'))
    groups_orient_m = np.repeat(np.arange(len(y_orient['male']) // 2), 2)
    groups_orient_f = np.repeat(np.arange(len(y_orient['female']) // 2), 2)

    X_wing = joblib.load(get_file('output', 'data', 'X_wing.joblib'))
    y_wing = joblib.load(get_file('output', 'data', 'y_wing.joblib'))
    groups_wing = np.repeat(np.arange(len(y_wing) // 2), 2)

    print(f"  Loaded is_fly:      X={X_is_fly.shape}, y={y_is_fly.shape}, groups={len(np.unique(groups_is_fly))}")
    print(f"  Loaded id:          X={X_id.shape}, y={y_id.shape}, groups={len(np.unique(groups_id))}")
    print(f"  Loaded orient_male: X={X_orient['male'].shape}, y={y_orient['male'].shape}, groups={len(np.unique(groups_orient_m))}")
    print(f"  Loaded orient_fem:  X={X_orient['female'].shape}, y={y_orient['female'].shape}, groups={len(np.unique(groups_orient_f))}")
    print(f"  Loaded wing:        X={X_wing.shape}, y={y_wing.shape}, groups={len(np.unique(groups_wing))}")

    all_metrics = {}
    test_preds = {}

    # -------------------------------------------------------------
    # Helper to train and evaluate a classification model
    # -------------------------------------------------------------
    def eval_classifier(name, model_fn, X, y, groups, categories):
        print(f"\n--- Training & Evaluating: {name} ---")

        # 1. Leakage-safe Grouped Split
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Fit model on training split
        clf = model_fn()
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t_fit = time.perf_counter() - t0

        # Predict
        t0 = time.perf_counter()
        y_test_pred = clf.predict(X_test)
        t_pred = time.perf_counter() - t0

        y_train_pred = clf.predict(X_train)

        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        prec = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        cm = confusion_matrix(y_test, y_test_pred).tolist()

        latency_ms = (t_pred / len(X_test)) * 1000.0

        print(f"  [Leakage-Safe Grouped Split]")
        print(f"    Train: {len(X_train)} samples ({len(np.unique(groups[train_idx]))} groups)")
        print(f"    Test:  {len(X_test)} samples ({len(np.unique(groups[test_idx]))} groups)")
        print(f"    Train Accuracy: {acc_train * 100:.2f}% (Error: {(1 - acc_train) * 100:.2f}%)")
        print(f"    Test Accuracy:  {acc_test * 100:.2f}% (Error: {(1 - acc_test) * 100:.2f}%)")
        print(f"    Macro Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print(f"    Confusion Matrix:\n{np.array(cm)}")
        print(f"    Fit time: {t_fit * 1000:.2f} ms, Latency: {latency_ms:.4f} ms/sample")

        # 2. Original Stanford Protocol (Naive unconstrained random split)
        X_tr_n, X_te_n, y_tr_n, y_te_n = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
        clf_naive = model_fn()
        clf_naive.fit(X_tr_n, y_tr_n)
        acc_train_n = accuracy_score(y_tr_n, clf_naive.predict(X_tr_n))
        acc_test_n = accuracy_score(y_te_n, clf_naive.predict(X_te_n))
        print(f"  [Original Stanford Naive Random Split]")
        print(f"    Train Accuracy: {acc_train_n * 100:.2f}%, Test Accuracy: {acc_test_n * 100:.2f}% (Error: {(1 - acc_test_n) * 100:.2f}%)")

        metrics = {
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'n_groups_train': int(len(np.unique(groups[train_idx]))),
            'n_groups_test': int(len(np.unique(groups[test_idx]))),
            'train_accuracy': float(acc_train),
            'test_accuracy': float(acc_test),
            'train_error_pct': float((1 - acc_train) * 100),
            'test_error_pct': float((1 - acc_test) * 100),
            'precision_macro': float(prec),
            'recall_macro': float(rec),
            'f1_macro': float(f1),
            'confusion_matrix': cm,
            'categories': categories,
            'fit_time_seconds': float(t_fit),
            'inference_latency_ms': float(latency_ms),
            'naive_split_test_accuracy': float(acc_test_n),
            'naive_split_test_error_pct': float((1 - acc_test_n) * 100)
        }

        return clf, metrics, (y_test, y_test_pred)

    # -------------------------------------------------------------
    # 1. FlyCount Model (Decision Tree)
    # -------------------------------------------------------------
    clf_is_fly, metrics_is_fly, preds_is_fly = eval_classifier(
        name='FlyCount (is_fly)',
        model_fn=lambda: tree.DecisionTreeClassifier(random_state=RANDOM_STATE),
        X=X_is_fly,
        y=y_is_fly,
        groups=groups_is_fly,
        categories=IS_FLY_CATS
    )
    all_metrics['is_fly'] = metrics_is_fly
    test_preds['is_fly'] = {'y_true': preds_is_fly[0], 'y_pred': preds_is_fly[1]}

    # Export Decision Tree dot diagram
    dot_path = get_file('output', 'diagrams', 'tree_is_fly.dot')
    tree.export_graphviz(
        clf_is_fly,
        out_file=dot_path,
        feature_names=IS_FLY_FEATS,
        class_names=IS_FLY_CATS,
        filled=True,
        rounded=True,
        special_characters=True
    )
    print(f"  Exported Decision Tree diagram: {dot_path}")

    # -------------------------------------------------------------
    # 2. Sex Identification Model (StandardScaler + Logistic Regression)
    # -------------------------------------------------------------
    clf_id, metrics_id, preds_id = eval_classifier(
        name='Sex Identification (id)',
        model_fn=lambda: make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)),
        X=X_id,
        y=y_id,
        groups=groups_id,
        categories=ID_CATS
    )
    all_metrics['id'] = metrics_id
    test_preds['id'] = {'y_true': preds_id[0], 'y_pred': preds_id[1]}

    # -------------------------------------------------------------
    # 3. Orientation Models (PCA(15) + Logistic Regression)
    # -------------------------------------------------------------
    # Male Orientation
    clf_orient_m, metrics_orient_m, preds_orient_m = eval_classifier(
        name='Male Orientation (orient_male)',
        model_fn=lambda: make_pipeline(PCA(n_components=15, random_state=RANDOM_STATE), LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)),
        X=X_orient['male'],
        y=y_orient['male'],
        groups=groups_orient_m,
        categories=ORIENT_CATS
    )
    all_metrics['orient_male'] = metrics_orient_m
    test_preds['orient_male'] = {'y_true': preds_orient_m[0], 'y_pred': preds_orient_m[1]}

    # Generate PCA plots for male
    gss_m = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx_m, _ = next(gss_m.split(X_orient['male'], y_orient['male'], groups_orient_m))
    pca_plot_m = PCA(n_components=min(200, len(tr_idx_m)), random_state=RANDOM_STATE)
    X_pca_tr_m = pca_plot_m.fit_transform(X_orient['male'][tr_idx_m])
    plot_pca_orientation(pca_plot_m, X_pca_tr_m, y_orient['male'][tr_idx_m], 'male')

    # Female Orientation
    clf_orient_f, metrics_orient_f, preds_orient_f = eval_classifier(
        name='Female Orientation (orient_female)',
        model_fn=lambda: make_pipeline(PCA(n_components=15, random_state=RANDOM_STATE), LogisticRegression(solver='lbfgs', random_state=RANDOM_STATE)),
        X=X_orient['female'],
        y=y_orient['female'],
        groups=groups_orient_f,
        categories=ORIENT_CATS
    )
    all_metrics['orient_female'] = metrics_orient_f
    test_preds['orient_female'] = {'y_true': preds_orient_f[0], 'y_pred': preds_orient_f[1]}

    # Generate PCA plots for female
    gss_f = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx_f, _ = next(gss_f.split(X_orient['female'], y_orient['female'], groups_orient_f))
    pca_plot_f = PCA(n_components=min(200, len(tr_idx_f)), random_state=RANDOM_STATE)
    X_pca_tr_f = pca_plot_f.fit_transform(X_orient['female'][tr_idx_f])
    plot_pca_orientation(pca_plot_f, X_pca_tr_f, y_orient['female'][tr_idx_f], 'female')

    # -------------------------------------------------------------
    # 4. Wing Angle Regression Model (PCA(40) + Linear Regression)
    # -------------------------------------------------------------
    print(f"\n--- Training & Evaluating: Male Wing Angle (wing) ---")
    gss_w = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    tr_idx_w, te_idx_w = next(gss_w.split(X_wing, y_wing, groups_wing))

    X_train_w, y_train_w = X_wing[tr_idx_w], y_wing[tr_idx_w]
    X_test_w, y_test_w = X_wing[te_idx_w], y_wing[te_idx_w]

    clf_wing = make_pipeline(PCA(n_components=40, random_state=RANDOM_STATE), LinearRegression())
    t0 = time.perf_counter()
    clf_wing.fit(X_train_w, y_train_w)
    t_fit_w = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_w = clf_wing.predict(X_test_w)
    t_pred_w = time.perf_counter() - t0
    latency_w = (t_pred_w / len(X_test_w)) * 1000.0

    y_train_pred_w = clf_wing.predict(X_train_w)

    # Calculate metrics in radians
    mae_rad = mean_absolute_error(y_test_w, y_pred_w)
    rmse_rad = np.sqrt(mean_squared_error(y_test_w, y_pred_w))
    sigma_rad = float(np.std(y_test_w - y_pred_w))
    train_sigma_rad = float(np.std(y_train_w - y_train_pred_w))
    r2 = r2_score(y_test_w, y_pred_w)

    # Convert to degrees
    y_test_deg = np.degrees(y_test_w)
    y_pred_deg = np.degrees(y_pred_w)
    mae_deg = float(mean_absolute_error(y_test_deg, y_pred_deg))
    rmse_deg = float(np.sqrt(mean_squared_error(y_test_deg, y_pred_deg)))
    sigma_deg = float(np.std(y_test_deg - y_pred_deg))
    train_sigma_deg = float(np.std(np.degrees(y_train_w) - np.degrees(y_train_pred_w)))

    print(f"  [Leakage-Safe Grouped Split]")
    print(f"    Train: {len(X_train_w)} samples ({len(np.unique(groups_wing[tr_idx_w]))} groups)")
    print(f"    Test:  {len(X_test_w)} samples ({len(np.unique(groups_wing[te_idx_w]))} groups)")
    print(f"    Train Residual Std (sigma): {train_sigma_deg:.2f}° ({train_sigma_rad:.4f} rad)")
    print(f"    Test Residual Std (sigma):  {sigma_deg:.2f}° ({sigma_rad:.4f} rad)")
    print(f"    Test MAE:                   {mae_deg:.2f}° ({mae_rad:.4f} rad)")
    print(f"    Test RMSE:                  {rmse_deg:.2f}° ({rmse_rad:.4f} rad)")
    print(f"    Test R^2 Score:             {r2:.4f}")
    print(f"    Fit time: {t_fit_w * 1000:.2f} ms, Latency: {latency_w:.4f} ms/sample")

    # Naive split comparison
    X_tr_wn, X_te_wn, y_tr_wn, y_te_wn = train_test_split(X_wing, y_wing, test_size=0.25, random_state=RANDOM_STATE)
    clf_wn = make_pipeline(PCA(n_components=40, random_state=RANDOM_STATE), LinearRegression())
    clf_wn.fit(X_tr_wn, y_tr_wn)
    y_pred_wn_deg = np.degrees(clf_wn.predict(X_te_wn))
    y_te_wn_deg = np.degrees(y_te_wn)
    mae_deg_n = float(mean_absolute_error(y_te_wn_deg, y_pred_wn_deg))
    rmse_deg_n = float(np.sqrt(mean_squared_error(y_te_wn_deg, y_pred_wn_deg)))
    sigma_deg_n = float(np.std(y_te_wn_deg - y_pred_wn_deg))
    print(f"  [Original Stanford Naive Random Split]")
    print(f"    Test MAE: {mae_deg_n:.2f}°, Test RMSE: {rmse_deg_n:.2f}°, Test Sigma: {sigma_deg_n:.2f}°")

    metrics_wing = {
        'n_train': int(len(X_train_w)),
        'n_test': int(len(X_test_w)),
        'n_groups_train': int(len(np.unique(groups_wing[tr_idx_w]))),
        'n_groups_test': int(len(np.unique(groups_wing[te_idx_w]))),
        'train_sigma_deg': train_sigma_deg,
        'test_sigma_deg': sigma_deg,
        'train_sigma_rad': train_sigma_rad,
        'test_sigma_rad': sigma_rad,
        'mae_deg': mae_deg,
        'rmse_deg': rmse_deg,
        'mae_rad': float(mae_rad),
        'rmse_rad': float(rmse_rad),
        'r2_score': float(r2),
        'fit_time_seconds': float(t_fit_w),
        'inference_latency_ms': float(latency_w),
        'naive_split_mae_deg': mae_deg_n,
        'naive_split_rmse_deg': rmse_deg_n,
        'naive_split_sigma_deg': sigma_deg_n
    }
    all_metrics['wing'] = metrics_wing
    test_preds['wing'] = {'y_true_deg': y_test_deg, 'y_pred_deg': y_pred_deg}

    # Generate PCA plots for wing
    pca_plot_w = PCA(n_components=min(200, len(tr_idx_w)), random_state=RANDOM_STATE)
    X_pca_tr_w = pca_plot_w.fit_transform(X_train_w)
    plot_pca_wing(pca_plot_w, X_pca_tr_w, np.degrees(y_train_w))

    # -------------------------------------------------------------
    # 5. Serialize Production Models to models/
    # -------------------------------------------------------------
    print("\n[Step 2] Serializing trained pipeline models to models/...")
    models_dir = os.path.join(repo_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_files = {
        'clf_is_fly.joblib': clf_is_fly,
        'clf_id.joblib': clf_id,
        'clf_orient_male.joblib': clf_orient_m,
        'clf_orient_female.joblib': clf_orient_f,
        'clf_wing.joblib': clf_wing
    }

    for filename, model_obj in model_files.items():
        dst_path = os.path.join(models_dir, filename)
        joblib.dump(model_obj, dst_path)
        sz = os.path.getsize(dst_path)
        print(f"  Saved: {filename:25s} Size: {sz:>10,d} bytes -> {dst_path}")

    # -------------------------------------------------------------
    # 6. Save Evaluation Metrics and Predictions
    # -------------------------------------------------------------
    metrics_path = get_file('output', 'data', 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved evaluation metrics JSON: {metrics_path}")

    preds_path = get_file('output', 'data', 'test_predictions.joblib')
    joblib.dump(test_preds, preds_path)
    print(f"  Saved test predictions: {preds_path}")

    # -------------------------------------------------------------
    # 7. Final Model Validation with Predictor Classes
    # -------------------------------------------------------------
    print("\n[Step 3] Reloading models from disk and validating integration with Predictor classes...")
    from cs229.train_is_fly import IsFlyPredictor
    from cs229.train_id import IdPredictor
    from cs229.train_orientation import PosePredictor
    from cs229.train_wing import WingPredictor
    from cs229.patch import ImagePatch

    is_fly_pred = IsFlyPredictor()
    id_pred = IdPredictor()
    pose_m_pred = PosePredictor('male')
    pose_f_pred = PosePredictor('female')
    wing_pred = WingPredictor()

    # Test is_fly predictor
    dummy_contour = np.array([[[100, 100]], [[100, 150]], [[150, 150]], [[150, 100]]])
    res_fly = is_fly_pred.predict(dummy_contour)
    assert res_fly in IS_FLY_CATS, f"Invalid is_fly prediction: {res_fly}"
    print(f"  IsFlyPredictor test: dummy contour predicted as '{res_fly}' -> [PASS]")

    # Test id predictor
    class DummyPatch:
        def estimate_aspect_ratio(self): return 1.5
    res_id = id_pred.predict(dummy_contour, dummy_contour, DummyPatch(), DummyPatch())
    assert res_id in ID_CATS, f"Invalid id prediction: {res_id}"
    print(f"  IdPredictor test: dummy pair predicted as '{res_id}' -> [PASS]")

    # Test PosePredictor and WingPredictor on 400x400 synthetic fly patch
    dummy_img = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(dummy_img, (200, 200), (60, 30), 45, 0, 360, 255, -1)
    dummy_mask = np.ones((400, 400), dtype=np.uint8) * 255
    patch = ImagePatch(dummy_img, dummy_mask)

    center_m, angle_m = pose_m_pred.predict(patch)
    assert not np.isnan(angle_m) and not np.isinf(angle_m)
    print(f"  PosePredictor(male) test: angle={np.degrees(angle_m):.2f}° -> [PASS]")

    center_f, angle_f = pose_f_pred.predict(patch)
    assert not np.isnan(angle_f) and not np.isinf(angle_f)
    print(f"  PosePredictor(female) test: angle={np.degrees(angle_f):.2f}° -> [PASS]")

    rw, lw = wing_pred.predict(patch)
    assert not np.isnan(rw) and not np.isnan(lw) and not np.isinf(rw) and not np.isinf(lw)
    print(f"  WingPredictor test: right_wing={np.degrees(rw):.2f}°, left_wing={np.degrees(lw):.2f}° -> [PASS]")

    print("\n" + "=" * 80)
    print("ALL MODELS TRAINED, EVALUATED, SERIALIZED, AND VALIDATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    run_training_and_evaluation()
