"""
Inference wrapper for resolution-time prediction. Loaded once at import.

Returns an ESTIMATE, never a guaranteed deadline - the API response shape
makes that explicit via `is_estimate: True` and `low_confidence`, and the
frontend copy (see static/js/ai_resolution.js) says "estimated" rather
than "will be resolved by".
"""

from pathlib import Path
from typing import Optional

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "resolution_time_predictor.joblib"

_bundle = None
_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    else:
        _load_error = f"Model file not found at {MODEL_PATH}. Run ml/training/train_resolution.py first."
except Exception as exc:  # noqa: BLE001
    _load_error = f"Failed to load resolution-time model: {exc}"

# A model trained on the documented synthetic bootstrap (vs. real
# historical data) is flagged as low-confidence in every response, so
# nothing downstream can present it as a measured, real-world estimate.
_LOW_CONFIDENCE_SOURCES = {"synthetic"}


def is_available() -> bool:
    return _bundle is not None


def predict_resolution_time(category: str, priority: str, recurrence_count: int = 0) -> dict:
    if not category or not priority:
        return {
            "available": False,
            "predicted_hours": None,
            "reason": "missing_input",
        }

    if not is_available():
        return {
            "available": False,
            "predicted_hours": None,
            "reason": _load_error,
        }

    model = _bundle["model"]
    trained_on = _bundle.get("trained_on")

    try:
        import pandas as pd
        row = pd.DataFrame([{
            "category": category,
            "priority": priority,
            "recurrence_count": recurrence_count,
        }])
        predicted_hours = float(model.predict(row)[0])
        predicted_hours = max(1.0, predicted_hours)  # guard against nonsensical negative/zero output
    except Exception as exc:  # noqa: BLE001 - never let a bad input crash the request
        return {
            "available": False,
            "predicted_hours": None,
            "reason": f"prediction_failed: {exc}",
        }

    return {
        "available": True,
        "is_estimate": True,
        "predicted_hours": round(predicted_hours, 1),
        "predicted_days": round(predicted_hours / 24.0, 1),
        "model_name": _bundle.get("model_name"),
        "trained_on": trained_on,
        "low_confidence": trained_on in _LOW_CONFIDENCE_SOURCES,
        "recurrence_count_used": recurrence_count,
    }
