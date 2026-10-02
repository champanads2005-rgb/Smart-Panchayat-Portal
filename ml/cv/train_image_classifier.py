"""
Trains the Phase 7 pothole image classifier.

Requires ml/cv/data/manifest.csv (built by dataset_prep.py) AND the raw
source images on disk at the SAME --raw-dir path used to build that
manifest (the manifest stores relative paths + hashes, not pixels - see
ml/cv/data/README.md for why the dataset itself isn't shipped here).

What this does:
    1. Loads every manifest row, re-validates its content hash against
       the file actually on disk (catches a stale/mismatched raw-dir
       early with a clear error instead of silently training on the
       wrong images).
    2. Extracts features via ml/cv/preprocessing.extract_features() -
       the EXACT SAME function ml/inference/image_classify.py calls at
       serve time, so there is no train/serve skew.
    3. Trains three real, different classical classifiers on the
       manifest's own `train` split: Logistic Regression, Linear SVM
       (with Platt-scaling calibration so it can also expose
       predict_proba), and Random Forest.
    4. Selects the best by macro-F1 on the `val` split.
    5. Reports FINAL metrics on the untouched `test` split only, once -
       accuracy, per-class precision/recall/F1, macro-F1, confusion
       matrix, ROC-AUC. mAP is not computed: this is whole-image
       classification with no bounding boxes, so mAP does not apply
       (see ml/cv/data/README.md "Limitations").
    6. Saves the selected model (features pipeline is deterministic code,
       not a fitted transformer, so only the classifier + a StandardScaler
       fitted on train are saved) to ml/cv/models/pothole_classifier.joblib.

Run:
    python ml/cv/train_image_classifier.py --raw-dir "/path/to/My Dataset"
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from ml.cv.preprocessing import extract_features, validate_and_load_image  # noqa: E402

MANIFEST_PATH = Path(__file__).parent / "data" / "manifest.csv"
MODEL_PATH = Path(__file__).parent / "models" / "pothole_classifier.joblib"
METRICS_JSON_PATH = Path(__file__).parent / "evaluation" / "image_classifier_metrics.json"
METRICS_TXT_PATH = Path(__file__).parent / "evaluation" / "image_classifier_metrics.txt"

POSITIVE_LABEL = "Pothole"


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run dataset_prep.py first."
        )
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _load_split(rows, raw_dir: Path, split: str, verbose=True):
    X, y, skipped = [], [], 0
    split_rows = [r for r in rows if r["split"] == split]
    for i, row in enumerate(split_rows):
        path = raw_dir / row["relative_path"]
        try:
            raw_bytes = path.read_bytes()
        except OSError:
            skipped += 1
            continue
        actual_hash = hashlib.sha256(raw_bytes).hexdigest()
        if actual_hash != row["sha256"]:
            skipped += 1
            continue
        try:
            image = validate_and_load_image(raw_bytes)
            features = extract_features(image)
        except Exception:  # noqa: BLE001 - skip unreadable files, don't crash training
            skipped += 1
            continue
        X.append(features)
        y.append(row["label"])
        if verbose and (i + 1) % 100 == 0:
            print(f"  [{split}] loaded {i + 1}/{len(split_rows)}")
    if skipped:
        print(f"  [{split}] skipped {skipped} file(s) (hash mismatch or unreadable).")
    return np.array(X), np.array(y)


def train_and_evaluate(raw_dir: Path):
    rows = _load_manifest()

    print("Loading and featurizing images (this is the slow step)...")
    t0 = time.time()
    X_train, y_train = _load_split(rows, raw_dir, "train")
    X_val, y_val = _load_split(rows, raw_dir, "val")
    X_test, y_test = _load_split(rows, raw_dir, "test")
    print(f"Featurization took {time.time() - t0:.1f}s")
    print(f"train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    SEED = 42
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "linear_svm_calibrated": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=5000), cv=3
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced", random_state=SEED
        ),
    }

    val_scores = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        fitted[name] = model
        preds = model.predict(X_val_s)
        macro_f1 = f1_score(y_val, preds, average="macro")
        val_scores[name] = macro_f1
        print(f"  {name}: val macro-F1 = {macro_f1:.3f}")

    best_name = max(val_scores, key=val_scores.get)
    best_model = fitted[best_name]
    print(f"Selected model: {best_name} (val macro-F1={val_scores[best_name]:.3f})")

    test_preds = best_model.predict(X_test_s)
    test_proba = best_model.predict_proba(X_test_s)
    classes = [str(c) for c in best_model.classes_]
    pos_idx = classes.index(POSITIVE_LABEL)

    report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds, labels=classes)
    try:
        auc = roc_auc_score((y_test == POSITIVE_LABEL).astype(int), test_proba[:, pos_idx])
    except ValueError:
        auc = None

    metrics = {
        "model_name": best_name,
        "trained_on": "real_web_scraped",
        "dataset_source": "https://github.com/Somashekarbm/PotholeDetection (My Dataset folder)",
        "classes": classes,
        "supported_classes_for_ui": ["Pothole"],
        "feature_pipeline": "HOG (grayscale) + HSV color histogram, 128x128",
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "val_macro_f1_by_candidate": val_scores,
        "test_accuracy": accuracy_score(y_test, test_preds),
        "test_macro_f1": f1_score(y_test, test_preds, average="macro"),
        "test_roc_auc": auc,
        "test_classification_report": report,
        "test_confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        "note": (
            "mAP is not applicable: this is a whole-image binary classifier "
            "with no bounding boxes, not an object detector."
        ),
    }

    METRICS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(METRICS_TXT_PATH, "w") as f:
        f.write(f"Selected model: {best_name}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Train/val/test sizes: {len(y_train)}/{len(y_val)}/{len(y_test)}\n\n")
        f.write("Validation macro-F1 by candidate:\n")
        for name, score in val_scores.items():
            f.write(f"  {name}: {score:.4f}\n")
        f.write(f"\nHeld-out TEST accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"Held-out TEST macro-F1: {metrics['test_macro_f1']:.4f}\n")
        f.write(f"Held-out TEST ROC-AUC (Pothole positive class): {auc}\n\n")
        f.write("Per-class precision/recall/F1 (test):\n")
        for label in classes:
            r = report[label]
            f.write(
                f"  {label}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
                f"f1={r['f1-score']:.3f} support={int(r['support'])}\n"
            )
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={classes}:\n")
        for row in cm.tolist():
            f.write(f"  {row}\n")
        f.write(
            "\nNote: mAP is not applicable - this is a whole-image binary "
            "classifier, not an object detector with bounding boxes.\n"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "scaler": scaler,
            "classes": classes,
            "model_name": best_name,
            "trained_on": "real_web_scraped",
            "positive_label": POSITIVE_LABEL,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_JSON_PATH} and {METRICS_TXT_PATH}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Path to the same 'My Dataset' folder passed to dataset_prep.py",
    )
    args = parser.parse_args()
    train_and_evaluate(Path(args.raw_dir).expanduser().resolve())
