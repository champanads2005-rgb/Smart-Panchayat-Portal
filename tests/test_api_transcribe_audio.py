"""
Integration tests for POST /api/ai/transcribe-audio via Flask's test
client - auth/role checks, missing/invalid file, unsupported language,
oversized file, and a real end-to-end transcription of synthesized
speech.

Run: pytest tests/test_api_transcribe_audio.py -v
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


def _speech_file(text="there is a water leak near my house", name="speech.wav"):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run(["espeak-ng", "-w", tmp.name, text], capture_output=True, timeout=15, check=True)
        data = Path(tmp.name).read_bytes()
    return (io.BytesIO(data), name)


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def _login(client, role="citizen"):
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_role"] = role


def test_transcribe_audio_requires_login(client):
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en", "audio": (io.BytesIO(b"x"), "a.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_transcribe_audio_rejects_non_citizen_roles(client):
    _login(client, role="officer")
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en", "audio": (io.BytesIO(b"x"), "a.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 401


def test_transcribe_audio_rejects_unsupported_language(client):
    _login(client)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "ta", "audio": (io.BytesIO(b"x"), "a.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json()["reason"] == "unsupported_language"


def test_transcribe_audio_requires_a_file(client):
    _login(client)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert resp.get_json()["reason"] == "no_file_provided"


def test_transcribe_audio_rejects_non_audio_file(client):
    _login(client)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en", "audio": (io.BytesIO(b"not audio data"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False
    assert data["reason"] == "not_a_valid_audio_file"


def test_transcribe_audio_rejects_oversized_file(client, monkeypatch):
    _login(client)
    monkeypatch.setattr(appmodule, "MAX_AUDIO_UPLOAD_BYTES", 100)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en", "audio": (io.BytesIO(b"x" * 200), "a.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert resp.get_json()["reason"] == "file_too_large"


@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_transcribe_audio_english_returns_real_transcript(client):
    _login(client)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "en", "audio": _speech_file()},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert isinstance(data["transcript"], str)
    assert "wav_bytes" not in data  # preview call must never leak raw audio bytes back


def test_transcribe_audio_kannada_reports_unavailable_not_a_crash(client):
    _login(client)
    resp = client.post(
        "/api/ai/transcribe-audio",
        data={"language": "kn", "audio": (io.BytesIO(b"not real audio"), "a.wav")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is False
