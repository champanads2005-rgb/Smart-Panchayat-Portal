"""
Integration tests for POST /complaint with an optional attached photo -
the full Phase 7 flow through the real Flask route: image re-validated
server-side (never trusting the earlier /api/ai/analyze-image preview
alone), complaint + complaint_updates + complaint_images rows inserted,
and the file actually written to UPLOAD_FOLDER. The real MySQL DB is
mocked (this sandbox/CI has no MySQL server) - only the DB layer is
faked; CSRF, session/role checks, and the real image pipeline all run
for real.

Run: pytest tests/test_complaint_submission_with_image.py -v
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


class _RecordingCursor:
    """Records every INSERT so the test can assert exactly what was
    written, and hands out an incrementing lastrowid like a real
    auto-increment primary key would."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.lastrowid = None
        self._next_id = 100

    def execute(self, query, params=None):
        self.recorder.append((" ".join(query.split()), params))
        if query.strip().upper().startswith("INSERT INTO COMPLAINTS"):
            self.lastrowid = self._next_id
            self._next_id += 1

    def close(self):
        pass


class _RecordingDB:
    def __init__(self, recorder):
        self.recorder = recorder
        self.committed = False

    def cursor(self, dictionary=False):
        return _RecordingCursor(self.recorder)

    def commit(self):
        self.committed = True

    def close(self):
        pass


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def _login_with_csrf(client):
    with client.session_transaction() as sess:
        sess["user_id"] = 7
        sess["user_role"] = "citizen"
        sess["csrf_token"] = "test-csrf-token"
    return "test-csrf-token"


def _base_form(csrf_token):
    return {
        "csrf_token": csrf_token,
        "title": "Big pothole on Main Road",
        "category": "Road",
        "description": "There is a dangerous pothole near the bus stop.",
        "priority": "High",
    }


def test_submit_complaint_with_image_stores_image_row_and_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(appmodule.Config, "UPLOAD_FOLDER", str(tmp_path))
    csrf_token = _login_with_csrf(client)

    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form(csrf_token)
    form["image"] = _jpeg_file()

    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 302  # redirect to my-complaints on success

    queries = [q for q, _ in recorder]
    assert any("INSERT INTO COMPLAINTS" in q.upper() for q in queries)
    assert any("INSERT INTO COMPLAINT_UPDATES" in q.upper() for q in queries)
    assert any("INSERT INTO COMPLAINT_IMAGES" in q.upper() for q in queries)

    # Exactly one file should have been written to the (temp) upload folder.
    written_files = list(tmp_path.iterdir())
    assert len(written_files) == 1
    assert written_files[0].suffix == ".jpg"


def test_submit_complaint_without_image_does_not_touch_complaint_images(client, monkeypatch):
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    resp = client.post("/complaint", data=_base_form(csrf_token), content_type="multipart/form-data")
    assert resp.status_code == 302

    queries = [q for q, _ in recorder]
    assert not any("INSERT INTO COMPLAINT_IMAGES" in q.upper() for q in queries)


def test_submit_complaint_with_invalid_image_still_succeeds(client, monkeypatch, tmp_path):
    """An unreadable 'photo' must never block submitting the actual
    complaint - the photo is supplementary evidence, not a required
    field (see the design-choice comment in app.py's complaint() view)."""
    monkeypatch.setattr(appmodule.Config, "UPLOAD_FOLDER", str(tmp_path))
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form(csrf_token)
    form["image"] = (io.BytesIO(b"not an image"), "broken.jpg")

    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 302

    queries = [q for q, _ in recorder]
    assert any("INSERT INTO COMPLAINTS" in q.upper() for q in queries)
    assert not any("INSERT INTO COMPLAINT_IMAGES" in q.upper() for q in queries)
    assert list(tmp_path.iterdir()) == []


def test_submit_complaint_requires_login(client):
    """No session csrf_token exists yet for an unauthenticated visitor,
    so the existing global CSRF check (app.py's enforce_csrf_on_forms,
    unchanged by this phase) rejects the request before the route's own
    login check ever runs - a 400, not a redirect. This matches the
    app's existing, pre-Phase-7 behavior for every other CSRF-protected
    form POST."""
    resp = client.post("/complaint", data=_base_form("whatever"), content_type="multipart/form-data")
    assert resp.status_code == 400


def test_submit_complaint_rejects_bad_csrf_token(client, monkeypatch):
    with client.session_transaction() as sess:
        sess["user_id"] = 7
        sess["user_role"] = "citizen"
        sess["csrf_token"] = "real-token"
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form("wrong-token")
    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert recorder == []  # never even reached the DB
