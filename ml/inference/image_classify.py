"""
Inference wrapper around the trained pothole image classifier.

Loads the saved model ONCE at import time (not per-request), exactly like
ml/inference/classify.py, and exposes analyze_image_bytes() used by the
Flask API. Degrades gracefully (available=False, never a crash) if the
model file is missing.

IMPORTANT - what this model can and can't do (see ml/cv/data/README.md
for the full rationale): it is trained and evaluated on exactly ONE
class of civic issue - road potholes, as a binary "Pothole" vs "Plain
road" whole-image classifier. SUPPORTED_CLASSES below is the single
source of truth for what this endpoint is allowed to claim to detect;
both the API and the UI read it from here so they can never drift out
of sync with what the model actually supports.
"""

from pathlib import Path
from typing import Optional

import joblib

from config import Config
from ml.cv.preprocessing import (
    InvalidImageError,
    extract_features,
    image_bytes_to_features,
    strip_metadata_and_reencode,
)

MODEL_PATH = Path(__file__).resolve().parent.parent / "cv" / "models" / "pothole_classifier.joblib"

# The ONLY civic-issue class this model actually detects. Deliberately a
# short list - never add a label here that the model wasn't trained and
# evaluated on (see ml/cv/data/README.md "What was tried and why the
# other four classes aren't supported").
SUPPORTED_CLASSES = ["Pothole"]

# Maps a confident model prediction to the existing complaint category
# ENUM (complaint.sql). Only classes the model can actually predict
# appear here - "Plain" intentionally has no entry, because "no pothole
# visible" is not a complaint category to suggest.
LABEL_TO_COMPLAINT_CATEGORY = {
    "Pothole": "Road",
}

_bundle = None
_load_error: Optional[str] = None

try:
    if MODEL_PATH.exists():
        _bundle = joblib.load(MODEL_PATH)
    else:
        _load_error = (
            f"Image model file not found at {MODEL_PATH}. Run "
            "ml/cv/dataset_prep.py and ml/cv/train_image_classifier.py first."
        )
except Exception as exc:  # noqa: BLE001 - any load failure degrades gracefully
    _load_error = f"Failed to load image classifier: {exc}"


def is_available() -> bool:
    return _bundle is not None


def get_load_error() -> Optional[str]:
    return _load_error


def _confidence_threshold() -> float:
    return Config.CV_CONFIDENCE_MIN


def analyze_image_bytes(raw_bytes: bytes) -> dict:
    """Validates and classifies an uploaded image. Returns a dict that is
    always JSON-serializable and never raises for bad input - invalid
    images and a missing model both come back as a clearly-marked,
    non-fake result (available=False or is_low_confidence=True), per the
    master prompt's requirement to handle these cases gracefully rather
    than crash or fabricate a result.
    """
    try:
        features, image = image_bytes_to_features(raw_bytes)
    except InvalidImageError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "detected_label": None,
            "confidence": None,
            "suggested_category": None,
            "is_low_confidence": None,
            "supported_classes": SUPPORTED_CLASSES,
        }

    if not is_available():
        return {
            "available": False,
            "reason": _load_error,
            "detected_label": None,
            "confidence": None,
            "suggested_category": None,
            "is_low_confidence": None,
            "supported_classes": SUPPORTED_CLASSES,
        }

    model = _bundle["model"]
    scaler = _bundle["scaler"]
    classes = _bundle["classes"]
    positive_label = _bundle.get("positive_label", "Pothole")

    features_scaled = scaler.transform([features])
    proba = model.predict_proba(features_scaled)[0]
    pred_idx = int(proba.argmax())
    predicted_label = classes[pred_idx]
    confidence = float(proba[pred_idx])

    threshold = _confidence_threshold()
    is_low_confidence = confidence < threshold

    suggested_category = None
    detected_issue = None
    if predicted_label == positive_label and not is_low_confidence:
        suggested_category = LABEL_TO_COMPLAINT_CATEGORY.get(predicted_label)
        detected_issue = predicted_label
    elif predicted_label == positive_label and is_low_confidence:
        # Model leans "Pothole" but isn't confident enough to suggest a
        # category outright - surfaced to the UI as an explicit
        # uncertain/fallback state, never silently upgraded to a
        # suggestion.
        detected_issue = predicted_label
    else:
        detected_issue = "No pothole detected"

    return {
        "available": True,
        "detected_label": predicted_label,
        "detected_issue": detected_issue,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "is_low_confidence": is_low_confidence,
        "suggested_category": suggested_category,
        "supported_classes": SUPPORTED_CLASSES,
        "model_name": _bundle.get("model_name"),
        "trained_on": _bundle.get("trained_on"),
        "reencoded_image_bytes": strip_metadata_and_reencode(image),
        "image_format": "JPEG",
    }
