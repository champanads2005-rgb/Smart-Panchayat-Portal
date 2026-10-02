"""
Integration tests for POST /api/ai/predict-resolution-time via Flask's
test client. The real MySQL DB is mocked out (this sandbox/CI has no
MySQL server) - only app.get_db_connection() and
app.get_recent_similar_count() are patched; the route, session handling,
and the real ml.inference.resolution_time module are exercised for real.

Run: pytest tests/test_api_resolution_time.py -v
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "root")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "test_db")

import pytest

import app as appmodule


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        return self._row

    def close(self):
        pass


class _FakeDB:
    def __init__(self, row):
        self._row = row

    def cursor(self, dictionary=False):
        return _FakeCursor(self._row)

    def close(self):
        pass


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def test_predict_resolution_time_requires_login(client):
    resp = client.post("/api/ai/predict-resolution-time", json={"category": "Water", "priority": "High"})
    assert resp.status_code == 401
    assert resp.get_json()["available"] is False


def test_predict_resolution_time_returns_estimate_when_authenticated(client, monkeypatch):
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB({"village": "Kadri"}))
    monkeypatch.setattr(appmodule, "get_recent_similar_count", lambda category, village, days=30: 2)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = "citizen"

    resp = client.post(
        "/api/ai/predict-resolution-time",
        json={"category": "Water", "priority": "High"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["is_estimate"] is True
    assert data["predicted_hours"] > 0
    assert data["recurrence_count_used"] == 2


def test_predict_resolution_time_handles_recurrence_lookup_failure(client, monkeypatch):
    """If the recurrence-count DB query blows up for any reason, the
    endpoint must still return a usable prediction (with recurrence
    falling back to 0) rather than a 500."""
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB({"village": "Kadri"}))

    def _boom(category, village, days=30):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(appmodule, "get_recent_similar_count", _boom)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = "citizen"

    resp = client.post(
        "/api/ai/predict-resolution-time",
        json={"category": "Road", "priority": "Low"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["recurrence_count_used"] == 0


def test_predict_resolution_time_missing_fields_returns_unavailable(client, monkeypatch):
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB({"village": "Kadri"}))
    monkeypatch.setattr(appmodule, "get_recent_similar_count", lambda category, village, days=30: 0)

    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = "citizen"

    resp = client.post("/api/ai/predict-resolution-time", json={"category": "", "priority": ""})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False
