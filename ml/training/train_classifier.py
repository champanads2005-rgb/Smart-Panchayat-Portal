"""
Trains and evaluates complaint category classifiers on the synthetic dataset.

Compares:
  - TF-IDF + Logistic Regression
  - TF-IDF + Linear SVM (with probability calibration for confidence scores)

Selects the best model by macro F1 on a held-out test split, saves it with
joblib, and writes a JSON metrics report (accuracy, precision, recall, F1,
confusion matrix) plus a human-readable metrics.txt. No numbers here are
invented — everything is computed from this run.

Run:
    python ml/training/train_classifier.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "complaints_synthetic.csv"
MODEL_DIR = BASE_DIR / "models"
EVAL_DIR = BASE_DIR / "evaluation"
MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df["text"].tolist(), df["category"].tolist()


def build_candidates():
    return {
        "tfidf_logreg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "tfidf_linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE), cv=3
            )),
        ]),
    }


def evaluate(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels": labels,
    }


def main():
    X, y = load_data()
    labels = sorted(set(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = build_candidates()
    results = {}
    fitted_models = {}

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        results[name] = evaluate(pipeline, X_test, y_test, labels)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {results[name]['accuracy']:.4f}")
        print(f"Macro F1: {results[name]['macro_f1']:.4f}")

    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = fitted_models[best_name]
    best_metrics = results[best_name]

    print(f"\nSelected best model: {best_name} (macro F1 = {best_metrics['macro_f1']:.4f})")

    # Save best model
    model_path = MODEL_DIR / "complaint_classifier.joblib"
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "labels": labels,
        "trained_on": "synthetic",
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, model_path)
    print(f"Saved model to {model_path}")

    # Save full comparison report
    report_path = EVAL_DIR / "classifier_metrics.json"
    with open(report_path, "w") as f:
        json.dump({
            "best_model": best_name,
            "all_results": results,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "dataset": "synthetic (ml/data/complaints_synthetic.csv)",
        }, f, indent=2)
    print(f"Saved metrics report to {report_path}")

    # Human-readable summary
    summary_path = EVAL_DIR / "classifier_metrics.txt"
    with open(summary_path, "w") as f:
        f.write("Complaint Classification — Model Comparison\n")
        f.write("=" * 50 + "\n")
        f.write("Dataset: SYNTHETIC (bootstrap, see ml/data/generate_synthetic_data.py)\n")
        f.write(f"Train size: {len(X_train)}  Test size: {len(X_test)}\n\n")
        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {res['macro_f1']:.4f}\n")
            for label in labels:
                r = res["classification_report"][label]
                f.write(f"    {label:15s} precision={r['precision']:.2f} recall={r['recall']:.2f} f1={r['f1-score']:.2f}\n")
            f.write("\n")
        f.write(f"SELECTED MODEL: {best_name}\n")
        f.write(f"Confusion matrix (rows=actual, cols=predicted), labels={labels}:\n")
        for row in best_metrics["confusion_matrix"]:
            f.write("  " + " ".join(f"{v:3d}" for v in row) + "\n")
    print(f"Saved human-readable summary to {summary_path}")


if __name__ == "__main__":
    main()
