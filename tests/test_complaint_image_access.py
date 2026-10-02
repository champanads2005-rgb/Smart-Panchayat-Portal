"""
Integration tests for GET /complaint-image/<id> - the ONLY route that
ever serves an uploaded complaint photo (never static/, see
config.py UPLOAD_FOLDER and ml/cv/data/README.md). The real MySQL DB is
mocked (this sandbox/CI has no MySQL server), matching the existing
convention in tests/test_api_resolution_time.py; the route's own
authorization logic is exercised for real.

Run: pytest tests/test_complaint_image_access.py -v
"""

import os
import sys
import tempfile
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


@pytest.fixture
def stored_image(tmp_path, monkeypatch):
    """Points UPLOAD_FOLDER at a temp dir and drops one real file in it,
    so the route's os.path.isfile()/send_file() calls succeed for real
    instead of needing to be mocked too."""
    monkeypatch.setattr(appmodule.Config, "UPLOAD_FOLDER", str(tmp_path))
    filename = "abc123.jpg"
    (tmp_path / filename).write_bytes(b"\xff\xd8\xff\xe0fakejpegbytes")
    return filename


def _login(client, user_id=1, role="citizen"):
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_role"] = role


def test_complaint_image_requires_login(client, stored_image):
    resp = client.get("/complaint-image/1")
    assert resp.status_code in (302, 401)  # redirected to login


def test_owner_can_view_their_own_image(client, monkeypatch, stored_image):
    _login(client, user_id=42, role="citizen")
    monkeypatch.setattr(
        appmodule,
        "get_db_connection",
        lambda: _FakeDB({
            "id": 1,
            "stored_filename": stored_image,
            "content_type": "image/jpeg",
            "owner_id": 42,
        }),
    )
    resp = client.get("/complaint-image/1")
    assert resp.status_code == 200


def test_other_citizen_cannot_view_someone_elses_image(client, monkeypatch, stored_image):
    """A citizen who is NOT the complaint's owner gets a plain 404 - the
    same response a genuinely nonexistent image would give, so the
    response itself can't be used to probe which image IDs exist."""
    _login(client, user_id=999, role="citizen")
    monkeypatch.setattr(
        appmodule,
        "get_db_connection",
        lambda: _FakeDB({
            "id": 1,
            "stored_filename": stored_image,
            "content_type": "image/jpeg",
            "owner_id": 42,  # belongs to a different citizen
        }),
    )
    resp = client.get("/complaint-image/1")
    assert resp.status_code == 404


def test_officer_can_view_any_complaint_image(client, monkeypatch, stored_image):
    _login(client, user_id=5, role="officer")
    monkeypatch.setattr(
        appmodule,
        "get_db_connection",
        lambda: _FakeDB({
            "id": 1,
            "stored_filename": stored_image,
            "content_type": "image/jpeg",
            "owner_id": 42,
        }),
    )
    resp = client.get("/complaint-image/1")
    assert resp.status_code == 200


def test_admin_can_view_any_complaint_image(client, monkeypatch, stored_image):
    _login(client, user_id=6, role="admin")
    monkeypatch.setattr(
        appmodule,
        "get_db_connection",
        lambda: _FakeDB({
            "id": 1,
            "stored_filename": stored_image,
            "content_type": "image/jpeg",
            "owner_id": 42,
        }),
    )
    resp = client.get("/complaint-image/1")
    assert resp.status_code == 200


def test_nonexistent_image_returns_404(client, monkeypatch):
    _login(client, user_id=1, role="citizen")
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB(None))
    resp = client.get("/complaint-image/999999")
    assert resp.status_code == 404


def test_missing_file_on_disk_returns_404_not_a_crash(client, monkeypatch, tmp_path):
    """DB row exists but the file was somehow deleted from disk -
    must not 500."""
    monkeypatch.setattr(appmodule.Config, "UPLOAD_FOLDER", str(tmp_path))
    _login(client, user_id=42, role="citizen")
    monkeypatch.setattr(
        appmodule,
        "get_db_connection",
        lambda: _FakeDB({
            "id": 1,
            "stored_filename": "does-not-exist.jpg",
            "content_type": "image/jpeg",
            "owner_id": 42,
        }),
    )
    resp = client.get("/complaint-image/1")
    assert resp.status_code == 404
