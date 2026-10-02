"""
Inference wrapper around the trained complaint classifier.

Loads the saved model ONCE at import time (not per-request) and exposes a
single `classify_complaint(text)` function used by the Flask API.

If the model file is missing (e.g. training hasn't been run yet), this
degrades gracefully instead of crashing the whole app - the complaint form
still works with manual category selection.
"""

from pathlib import Path
from typing import Optional

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "complaint_classifier.joblib"

_bundle = None
_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    else:
        _load_error = f"Model file not found at {MODEL_PATH}. Run ml/training/train_classifier.py first."
except Exception as exc:  # noqa: BLE001 - we want any load failure to degrade gracefully
    _load_error = f"Failed to load classifier: {exc}"


def is_available() -> bool:
    return _bundle is not None


def get_load_error() -> Optional[str]:
    return _load_error


def top_contributing_terms(text: str, n: int = 5):
    """Simple explainability: which TF-IDF terms in this text had the
    highest weight for the predicted class, using the linear model's
    coefficients where available. Falls back to an empty list otherwise."""
    if not is_available():
        return []
    model = _bundle["model"]
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        # Works for LogisticRegression directly; for calibrated SVM, the
        # underlying estimator's coefficients aren't directly exposed the
        # same way, so we skip gracefully.
        if not hasattr(clf, "coef_"):
            return []
        vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        pred_idx = list(clf.classes_).index(model.predict([text])[0])
        coefs = clf.coef_[pred_idx]
        row = vec.tocoo()
        scored = [(feature_names[j], row.data[i] * coefs[j]) for i, j in enumerate(row.col)]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [term for term, score in scored[:n] if score > 0]
    except Exception:  # noqa: BLE001
        return []


def classify_complaint(text: str) -> dict:
    """Returns a dict with predicted category, confidence, model metadata,
    and top contributing terms. If the model isn't available, returns a
    clearly-marked fallback result instead of a fake prediction."""
    if not text or not text.strip():
        return {
            "available": False,
            "category": None,
            "confidence": None,
            "reason": "empty_text",
        }

    if not is_available():
        return {
            "available": False,
            "category": None,
            "confidence": None,
            "reason": _load_error,
        }

    model = _bundle["model"]
    labels = _bundle["labels"]

    prediction = model.predict([text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba([text])[0]
            class_index = list(model.classes_).index(prediction)
            confidence = float(proba[class_index])
        except Exception:  # noqa: BLE001
            confidence = None

    return {
        "available": True,
        "category": prediction,
        "confidence": confidence,
        "model_name": _bundle.get("model_name"),
        "trained_on": _bundle.get("trained_on"),
        "top_terms": top_contributing_terms(text),
        "all_labels": labels,
    }
