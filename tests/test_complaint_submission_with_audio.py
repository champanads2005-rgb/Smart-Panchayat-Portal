"""
Integration tests for POST /complaint with an optional attached voice
recording - mirrors tests/test_complaint_submission_with_image.py's
structure and DB-mocking approach exactly.

Run: pytest tests/test_complaint_submission_with_audio.py -v
"""

import io
import os
import shutil
import subprocess
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

ESPEAK_AVAILABLE = shutil.which("espeak-ng") is not None


def _speech_file(text="there is no water supply in our village", name="speech.wav"):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["espeak-ng", "-w", tmp.name, text], capture_output=True, timeout=15, check=True)
        data = Path(tmp.name).read_bytes()
    return (io.BytesIO(data), name)


class _RecordingCursor:
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
        "title": "No water supply",
        "category": "Water",
        "description": "There has been no water supply for three days.",
        "priority": "High",
    }


@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_submit_complaint_with_audio_stores_audio_row_and_file(client, monkeypatch, tmp_path):
    monkeypatch.setattr(appmodule.Config, "AUDIO_UPLOAD_FOLDER", str(tmp_path))
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form(csrf_token)
    form["audio"] = _speech_file()
    form["audio_language"] = "en"

    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 302

    queries = [q for q, _ in recorder]
    assert any("INSERT INTO COMPLAINTS" in q.upper() for q in queries)
    assert any("INSERT INTO COMPLAINT_AUDIO" in q.upper() for q in queries)

    written_files = list(tmp_path.iterdir())
    assert len(written_files) == 1
    assert written_files[0].suffix == ".wav"


def test_submit_complaint_without_audio_does_not_touch_complaint_audio(client, monkeypatch):
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    resp = client.post("/complaint", data=_base_form(csrf_token), content_type="multipart/form-data")
    assert resp.status_code == 302

    queries = [q for q, _ in recorder]
    assert not any("INSERT INTO COMPLAINT_AUDIO" in q.upper() for q in queries)


def test_submit_complaint_with_invalid_audio_still_succeeds(client, monkeypatch, tmp_path):
    """A bad recording never blocks submitting the complaint text itself
    - same design choice as the Phase 7 image path."""
    monkeypatch.setattr(appmodule.Config, "AUDIO_UPLOAD_FOLDER", str(tmp_path))
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form(csrf_token)
    form["audio"] = (io.BytesIO(b"not real audio data"), "broken.wav")
    form["audio_language"] = "en"

    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 302

    queries = [q for q, _ in recorder]
    assert any("INSERT INTO COMPLAINTS" in q.upper() for q in queries)
    assert not any("INSERT INTO COMPLAINT_AUDIO" in q.upper() for q in queries)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_submit_complaint_kannada_audio_stores_row_but_marks_transcription_unavailable(client, monkeypatch, tmp_path):
    """Kannada audio is still stored as an attached recording (the
    citizen may want it kept even though no transcript could be
    produced) - it's just marked as having no transcription available."""
    monkeypatch.setattr(appmodule.Config, "AUDIO_UPLOAD_FOLDER", str(tmp_path))
    csrf_token = _login_with_csrf(client)
    recorder = []
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _RecordingDB(recorder))

    form = _base_form(csrf_token)
    form["audio"] = _speech_file()  # engine/language mismatch is fine - just needs decodable audio
    form["audio_language"] = "kn"

    resp = client.post("/complaint", data=form, content_type="multipart/form-data")
    assert resp.status_code == 302

    insert_audio_calls = [(q, p) for q, p in recorder if "INSERT INTO COMPLAINT_AUDIO" in q.upper()]
    assert len(insert_audio_calls) == 1
    params = insert_audio_calls[0][1]
    # column order: complaint_id, stored_filename, content_type, file_size,
    # duration_seconds, language_selected, transcription_engine,
    # raw_transcript, confirmed_transcript, asr_confidence,
    # is_low_confidence, transcription_available, uploaded_by
    assert params[5] == "kn"
    assert params[11] is False  # transcription_available


def test_submit_complaint_requires_login(client):
    resp = client.post("/complaint", data=_base_form("whatever"), content_type="multipart/form-data")
    assert resp.status_code == 400  # CSRF check runs before login check - see test_complaint_submission_with_image.py
