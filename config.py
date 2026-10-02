"""
Centralized configuration loaded from environment variables (.env).

Nothing in this file is a real secret - actual values live in a local
.env file (see .env.example) that is NEVER committed to source control.
"""

import os
import secrets

from dotenv import load_dotenv

load_dotenv()


def _require_or_generate_secret_key():
    """In production this MUST be set via env var. For local/dev
    convenience only, we generate a random one and warn - we never
    hardcode a fixed secret in source."""
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    generated = secrets.token_hex(32)
    print(
        "[config] WARNING: SECRET_KEY not set in environment. "
        "Generated a temporary one for this process only - sessions will "
        "invalidate on restart. Set SECRET_KEY in your .env for real use."
    )
    return generated


class Config:
    SECRET_KEY = _require_or_generate_secret_key()

    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_USER = os.environ.get("DB_USER", "root")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
    DB_NAME = os.environ.get("DB_NAME", "smart_panchayat")

    # ML / AI config
    CLASSIFIER_CONFIDENCE_MIN = float(os.environ.get("CLASSIFIER_CONFIDENCE_MIN", "0.35"))

    # Reserved for future Ollama/RAG integration - configurable, never hardcoded
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

    # Phase 7: image-based complaint detection
    # Below this confidence, a "Pothole" prediction is shown as uncertain
    # rather than turned into a category suggestion (see
    # ml/inference/image_classify.py).
    CV_CONFIDENCE_MIN = float(os.environ.get("CV_CONFIDENCE_MIN", "0.65"))

    # Stored OUTSIDE static/ on purpose - static/ is served by Flask with
    # no auth check, which would defeat requirement 11 (prevent
    # unauthorized access to uploaded complaint images). Images are only
    # ever served through the authenticated /complaint-image/<id> route.
    UPLOAD_FOLDER = os.environ.get(
        "UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "uploads", "complaint_images")
    )
    MAX_IMAGE_UPLOAD_BYTES = int(os.environ.get("MAX_IMAGE_UPLOAD_BYTES", str(5 * 1024 * 1024)))  # 5 MB

    # Phase 8: voice complaint submission
    AUDIO_UPLOAD_FOLDER = os.environ.get(
        "AUDIO_UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "uploads", "complaint_audio")
    )
    MAX_AUDIO_UPLOAD_BYTES = int(os.environ.get("MAX_AUDIO_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB
    MAX_AUDIO_DURATION_SECONDS = int(os.environ.get("MAX_AUDIO_DURATION_SECONDS", "120"))
    # How long an original recording is kept before scripts/purge_expired_audio.py
    # deletes it (the complaint's transcript already lives permanently in
    # complaints.description independent of this - only the ORIGINAL
    # AUDIO FILE is subject to retention). 0 disables auto-purging.
    AUDIO_RETENTION_DAYS = int(os.environ.get("AUDIO_RETENTION_DAYS", "90"))
    # Where a real deployment's pre-downloaded Whisper weights live, if
    # any (see ml/asr/data/README.md) - None means "try to download from
    # Hugging Face Hub", which this sandbox cannot reach.
    WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR") or None
    WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "tiny")
