"""
Tests for ml/inference/resolution_time.py — prediction correctness and,
importantly, graceful handling of missing input and a missing/broken
model file (this must never crash a request).

Run: pytest tests/test_resolution_inference.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib

import ml.inference.resolution_time as rt


def test_model_is_available_after_training():
    """Assumes ml/training/train_resolution.py has already been run (it
    has, as part of this phase) - this is a smoke test that the shipped
    model file actually loads."""
    assert rt.is_available() is True


def test_predict_returns_estimate_shape_for_valid_input():
    result = rt.predict_resolution_time("Water", "High", recurrence_count=2)
    assert result["available"] is True
    assert result["is_estimate"] is True
    assert isinstance(result["predicted_hours"], float)
    assert result["predicted_hours"] > 0
    assert isinstance(result["predicted_days"], float)
    assert "low_confidence" in result


def test_predict_flags_low_confidence_for_synthetic_model():
    """The model shipped in this phase was trained on the documented
    synthetic bootstrap (no real resolved-complaint history exists yet),
    so every prediction MUST come back flagged low_confidence=True - this
    guards against ever silently presenting a bootstrap estimate as a
    measured, real-world one."""
    result = rt.predict_resolution_time("Road", "Medium", recurrence_count=0)
    assert result["trained_on"] == "synthetic"
    assert result["low_confidence"] is True


def test_predict_handles_missing_category_gracefully():
    result = rt.predict_resolution_time("", "High")
    assert result["available"] is False
    assert result["reason"] == "missing_input"
    assert result["predicted_hours"] is None


def test_predict_handles_missing_priority_gracefully():
    result = rt.predict_resolution_time("Water", "")
    assert result["available"] is False
    assert result["reason"] == "missing_input"


def test_predict_handles_unknown_category_without_crashing():
    """An unseen category value must not raise - the OneHotEncoder in the
    trained pipeline was built with handle_unknown='ignore' for exactly
    this reason."""
    result = rt.predict_resolution_time("NotARealCategory", "High")
    assert result["available"] is True  # degrades to a prediction, doesn't crash
    assert result["predicted_hours"] > 0


def test_predict_degrades_gracefully_when_model_missing(monkeypatch):
    """Simulates a missing/corrupt model file (e.g. training was never
    run) and checks the module reports unavailable instead of raising."""
    monkeypatch.setattr(rt, "_bundle", None)
    monkeypatch.setattr(rt, "_load_error", "Model file not found (simulated for test).")

    result = rt.predict_resolution_time("Water", "High")
    assert result["available"] is False
    assert result["predicted_hours"] is None
    assert "not found" in result["reason"]
