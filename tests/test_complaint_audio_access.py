"""
Integration tests for GET /complaint-audio/<id> - mirrors
tests/test_complaint_image_access.py exactly (same authorization logic
in app.py, same anti-enumeration 404 pattern). The real MySQL DB is
mocked; the route's own authorization logic runs for real.

Run: pytest tests/test_complaint_audio_access.py -v
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


@pytest.fixture
def stored_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(appmodule.Config, "AUDIO_UPLOAD_FOLDER", str(tmp_path))
    filename = "abc123.wav"
    (tmp_path / filename).write_bytes(b"RIFFfakewavdata")
    return filename


def _login(client, user_id=1, role="citizen"):
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_role"] = role


def test_complaint_audio_requires_login(client, stored_audio):
    resp = client.get("/complaint-audio/1")
    assert resp.status_code in (302, 401)


def test_owner_can_access_their_own_audio(client, monkeypatch, stored_audio):
    _login(client, user_id=42, role="citizen")
    monkeypatch.setattr(
        appmodule, "get_db_connection",
        lambda: _FakeDB({"id": 1, "stored_filename": stored_audio, "content_type": "audio/wav", "owner_id": 42}),
    )
    resp = client.get("/complaint-audio/1")
    assert resp.status_code == 200


def test_other_citizen_cannot_access_someone_elses_audio(client, monkeypatch, stored_audio):
    _login(client, user_id=999, role="citizen")
    monkeypatch.setattr(
        appmodule, "get_db_connection",
        lambda: _FakeDB({"id": 1, "stored_filename": stored_audio, "content_type": "audio/wav", "owner_id": 42}),
    )
    resp = client.get("/complaint-audio/1")
    assert resp.status_code == 404


def test_officer_can_access_any_complaint_audio(client, monkeypatch, stored_audio):
    _login(client, user_id=5, role="officer")
    monkeypatch.setattr(
        appmodule, "get_db_connection",
        lambda: _FakeDB({"id": 1, "stored_filename": stored_audio, "content_type": "audio/wav", "owner_id": 42}),
    )
    resp = client.get("/complaint-audio/1")
    assert resp.status_code == 200


def test_admin_can_access_any_complaint_audio(client, monkeypatch, stored_audio):
    _login(client, user_id=6, role="admin")
    monkeypatch.setattr(
        appmodule, "get_db_connection",
        lambda: _FakeDB({"id": 1, "stored_filename": stored_audio, "content_type": "audio/wav", "owner_id": 42}),
    )
    resp = client.get("/complaint-audio/1")
    assert resp.status_code == 200


def test_nonexistent_audio_returns_404(client, monkeypatch):
    _login(client, user_id=1, role="citizen")
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB(None))
    resp = client.get("/complaint-audio/999999")
    assert resp.status_code == 404


def test_missing_file_on_disk_returns_404_not_a_crash(client, monkeypatch, tmp_path):
    monkeypatch.setattr(appmodule.Config, "AUDIO_UPLOAD_FOLDER", str(tmp_path))
    _login(client, user_id=42, role="citizen")
    monkeypatch.setattr(
        appmodule, "get_db_connection",
        lambda: _FakeDB({"id": 1, "stored_filename": "gone.wav", "content_type": "audio/wav", "owner_id": 42}),
    )
    resp = client.get("/complaint-audio/1")
    assert resp.status_code == 404
