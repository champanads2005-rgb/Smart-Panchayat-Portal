"""
Inference wrapper for the priority predictor. Loaded once at import time.

Explainability approach: since the model mixes TF-IDF text, one-hot
category, and a numeric recurrence_count through a ColumnTransformer,
per-feature SHAP-style attribution is heavier than this phase needs.
Instead we use a real, computed ablation: we re-run the model with each
input group (text / category / recurrence) replaced by a neutral baseline
and measure how much the predicted class's probability drops. That drop
*is* the reported "importance" for that factor on this specific
prediction - not an invented number.
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "priority_predictor.joblib"

_bundle = None
_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    else:
        _load_error = f"Model file not found at {MODEL_PATH}. Run ml/training/train_priority.py first."
except Exception as exc:  # noqa: BLE001
    _load_error = f"Failed to load priority model: {exc}"


def is_available() -> bool:
    return _bundle is not None


def _row(text, category, recurrence_count):
    return pd.DataFrame([{
        "text": text,
        "category": category,
        "recurrence_count": recurrence_count,
    }])


def _class_probability(model, row, predicted_class):
    proba = model.predict_proba(row)[0]
    idx = list(model.classes_).index(predicted_class)
    return float(proba[idx])


def _ablation_factors(model, text, category, recurrence_count, predicted_class):
    """Neutral baselines: empty-ish text, most-common category placeholder
    that still exists in training (we just reuse 'Other'), and
    recurrence_count = 0. Each ablation swaps ONE factor to its baseline
    and measures the resulting drop in predicted-class probability."""
    baseline_row = _row(text, category, recurrence_count)
    full_confidence = _class_probability(model, baseline_row, predicted_class)

    factors = {}

    no_text_row = _row(" ", category, recurrence_count)
    factors["complaint wording"] = full_confidence - _class_probability(model, no_text_row, predicted_class)

    no_category_row = _row(text, "Other", recurrence_count)
    factors["category"] = full_confidence - _class_probability(model, no_category_row, predicted_class)

    no_recurrence_row = _row(text, category, 0)
    factors["recent similar complaints"] = full_confidence - _class_probability(model, no_recurrence_row, predicted_class)

    # Sort by magnitude of contribution, keep only positive contributors
    ranked = sorted(factors.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, contribution in ranked if contribution > 0.01]


def predict_priority(text: str, category: str, recurrence_count: int = 0) -> dict:
    if not text or not text.strip() or not category:
        return {"available": False, "priority": None, "confidence": None, "reason": "missing_input"}

    if not is_available():
        return {"available": False, "priority": None, "confidence": None, "reason": _load_error}

    model = _bundle["model"]
    row = _row(text, category, recurrence_count)

    prediction = model.predict(row)[0]
    confidence = _class_probability(model, row, prediction)
    factors = _ablation_factors(model, text, category, recurrence_count, prediction)

    return {
        "available": True,
        "priority": prediction,
        "confidence": confidence,
        "model_name": _bundle.get("model_name"),
        "trained_on": _bundle.get("trained_on"),
        "top_factors": factors,
        "recurrence_count_used": recurrence_count,
    }
