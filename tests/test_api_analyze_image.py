"""
Integration tests for POST /api/ai/analyze-image via Flask's test
client. Mirrors the existing convention in tests/test_api_resolution_time.py
- no real MySQL server is needed for this endpoint (it doesn't touch the
DB at all), so no DB mocking is required here; DB mocking IS used in
tests/test_complaint_image_flow.py for the routes that do hit the DB.

Run: pytest tests/test_api_analyze_image.py -v
"""

import io
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
from PIL import Image

import app as appmodule


def _jpeg_file(size=(200, 150), color=(120, 120, 120), name="photo.jpg"):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return (buf, name)


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def _login(client, role="citizen"):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = role


def test_analyze_image_requires_login(client):
    data = {"image": _jpeg_file()}
    resp = client.post("/api/ai/analyze-image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 401
    assert resp.get_json()["available"] is False


def test_analyze_image_rejects_non_citizen_roles(client):
    _login(client, role="officer")
    data = {"image": _jpeg_file()}
    resp = client.post("/api/ai/analyze-image", data=data, content_type="multipart/form-data")
    assert resp.status_code == 401


def test_analyze_image_requires_a_file(client):
    _login(client)
    resp = client.post("/api/ai/analyze-image", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["reason"] == "no_file_provided"


def test_analyze_image_rejects_non_image_file(client):
    _login(client)
    fake_file = (io.BytesIO(b"not an image, just some bytes"), "notes.txt")
    resp = client.post(
        "/api/ai/analyze-image",
        data={"image": fake_file},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False
    assert data["reason"] == "not_a_valid_image"


def test_analyze_image_rejects_oversized_file(client, monkeypatch):
    _login(client)
    # Shrink the limit for this test rather than actually uploading 5MB+.
    monkeypatch.setattr(appmodule, "MAX_IMAGE_UPLOAD_BYTES", 100)
    big_file = _jpeg_file(size=(500, 500))
    resp = client.post(
        "/api/ai/analyze-image",
        data={"image": big_file},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert resp.get_json()["reason"] == "file_too_large"


def test_analyze_image_returns_result_for_valid_authenticated_upload(client):
    _login(client)
    resp = client.post(
        "/api/ai/analyze-image",
        data={"image": _jpeg_file()},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["detected_label"] in ("Plain", "Pothole")
    assert data["supported_classes"] == ["Pothole"]
    # The preview endpoint must never leak raw re-encoded image bytes.
    assert "reencoded_image_bytes" not in data
