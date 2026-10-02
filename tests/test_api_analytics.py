"""
Integration tests for the AI analytics routes:
  GET /admin/analytics
  GET /api/analytics/hotspots
  GET /api/analytics/anomalies

DB access is mocked via _fetch_complaint_records_for_analytics -
everything else (auth checks, route wiring, the real analytics module)
is exercised for real.

Run: pytest tests/test_api_analytics.py -v
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "root")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "test_db")

import pytest

import app as appmodule


def _sample_records():
    start = datetime(2026, 1, 1)
    records = []
    for week in range(12):
        for _ in range(3):
            records.append({
                "category": "Water",
                "village": "Kadri",
                "created_at": start + timedelta(days=week * 7 + 1),
            })
    return records


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def _login_admin(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = "admin"
        sess["user_name"] = "Test Admin"


def test_analytics_page_requires_login(client):
    resp = client.get("/admin/analytics")
    assert resp.status_code in (302, 401)  # redirected to login


def test_analytics_page_requires_admin_role(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = "citizen"
    resp = client.get("/admin/analytics")
    assert resp.status_code == 302  # redirected to dashboard, not admin content


def test_analytics_page_renders_for_admin(client, monkeypatch):
    monkeypatch.setattr(appmodule, "_fetch_complaint_records_for_analytics", _sample_records)
    _login_admin(client)
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    assert b"Complaint Hotspots" in resp.data


def test_hotspots_api_requires_admin(client):
    resp = client.get("/api/analytics/hotspots")
    assert resp.status_code == 401


def test_hotspots_api_returns_real_computed_data(client, monkeypatch):
    monkeypatch.setattr(appmodule, "_fetch_complaint_records_for_analytics", _sample_records)
    _login_admin(client)
    resp = client.get("/api/analytics/hotspots")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert any(h["village"] == "Kadri" and h["category"] == "Water" for h in data["hotspots"])


def test_anomalies_api_returns_both_methods(client, monkeypatch):
    monkeypatch.setattr(appmodule, "_fetch_complaint_records_for_analytics", _sample_records)
    _login_admin(client)
    resp = client.get("/api/analytics/anomalies")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert "statistical" in data
    assert "isolation_forest" in data


def test_analytics_apis_handle_empty_database(client, monkeypatch):
    monkeypatch.setattr(appmodule, "_fetch_complaint_records_for_analytics", lambda: [])
    _login_admin(client)

    resp = client.get("/api/analytics/hotspots")
    assert resp.status_code == 200
    assert resp.get_json()["hotspots"] == []

    resp = client.get("/api/analytics/anomalies")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["statistical"] == []
    assert data["isolation_forest"] == []
