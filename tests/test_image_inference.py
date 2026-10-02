"""
Tests for ml/inference/image_classify.py - real shipped model smoke
tests, invalid-input handling, and graceful degradation when the model
file is missing.

Run: pytest tests/test_image_inference.py -v
"""

import importlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

import ml.inference.image_classify as image_classify


def _make_jpeg_bytes(size=(200, 150), color=(120, 120, 120)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_model_is_available_after_training():
    """Assumes ml/cv/train_image_classifier.py has already been run (it
    has, as part of this phase) - smoke test that the shipped model
    actually loads."""
    assert image_classify.is_available() is True


def test_supported_classes_is_exactly_pothole():
    """This is the single source of truth read by both the API and the
    UI - it must never silently grow to claim a class the model wasn't
    trained on."""
    assert image_classify.SUPPORTED_CLASSES == ["Pothole"]


def test_analyze_valid_image_returns_well_formed_result():
    result = image_classify.analyze_image_bytes(_make_jpeg_bytes())
    assert result["available"] is True
    assert result["detected_label"] in ("Plain", "Pothole")
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["is_low_confidence"], bool)
    assert result["supported_classes"] == ["Pothole"]
    assert "reencoded_image_bytes" in result
    assert isinstance(result["reencoded_image_bytes"], bytes)


def test_analyze_never_suggests_a_category_for_the_plain_class():
    """Even if the model is very confident it's looking at a plain road,
    that's "no issue detected", not a complaint category suggestion."""
    result = image_classify.analyze_image_bytes(_make_jpeg_bytes(color=(180, 180, 180)))
    if result["detected_label"] == "Plain":
        assert result["suggested_category"] is None


def test_analyze_only_suggests_category_above_confidence_threshold():
    result = image_classify.analyze_image_bytes(_make_jpeg_bytes())
    if result["is_low_confidence"]:
        assert result["suggested_category"] is None
    if result["suggested_category"] is not None:
        assert result["is_low_confidence"] is False
        assert result["detected_label"] == "Pothole"
        assert result["suggested_category"] == "Road"


def test_analyze_invalid_image_returns_unavailable_not_a_crash():
    result = image_classify.analyze_image_bytes(b"not an image")
    assert result["available"] is False
    assert result["reason"] == "not_a_valid_image"
    assert result["detected_label"] is None
    assert result["supported_classes"] == ["Pothole"]


def test_analyze_empty_bytes_returns_unavailable():
    result = image_classify.analyze_image_bytes(b"")
    assert result["available"] is False
    assert result["reason"] == "empty_file"


def test_missing_model_degrades_gracefully(monkeypatch):
    """Simulates the model file never having been trained/shipped -
    must never raise, must clearly report why."""
    monkeypatch.setattr(image_classify, "_bundle", None)
    monkeypatch.setattr(image_classify, "_load_error", "simulated: model file not found")

    result = image_classify.analyze_image_bytes(_make_jpeg_bytes())
    assert result["available"] is False
    assert result["reason"] == "simulated: model file not found"
    assert result["supported_classes"] == ["Pothole"]


def test_get_load_error_is_none_when_model_present():
    assert image_classify.get_load_error() is None
