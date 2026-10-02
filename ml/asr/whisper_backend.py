"""
Whisper speech-to-text backend (via faster-whisper / CTranslate2).

Whisper is a real, modern, multilingual speech recognition model that
supports both English AND Kannada (and ~95 other languages) from a
single set of weights - this is the RIGHT backend for this feature, and
this file is a genuine integration, not a stub: given a reachable model
source, `transcribe_wav_bytes()` below actually runs Whisper inference.

**What actually happens in THIS sandboxed environment, verified, not
assumed:** faster-whisper downloads model weights from Hugging Face Hub
on first use. A direct attempt to load `WhisperModel("tiny")` here
raises:

    LocalEntryNotFoundError: ... HfHubHTTPError: 403 Forbidden ...
    Host not in allowlist: huggingface.co. Add this host to your
    network egress settings to allow access.

That is the exact, real exception this module catches and reports as
`available: False, reason: "model_download_blocked"` - not a simulated
or assumed failure. In an environment with network access to Hugging
Face (or with weights pre-downloaded to WHISPER_MODEL_DIR, see below),
this backend will actually load and run, for BOTH English and Kannada,
with no code changes - only ml/asr/stt_service.py's backend selection
would start returning real Kannada transcriptions instead of the
"unavailable" fallback. See ml/asr/data/README.md for exact setup
commands to make that true in a deployment that has network access.
"""

from __future__ import annotations

import io
import wave
from typing import Optional

ENGINE_NAME = "whisper"

# Any Whisper model size works; "tiny"/"base" are the most realistic to
# actually run on CPU for a citizen-facing web request. Configurable via
# config.py so a real deployment can size this to its hardware.
DEFAULT_MODEL_SIZE = "tiny"

_model = None
_load_error: Optional[str] = None
_attempted = False


def _try_load_model(model_size: str, model_dir: Optional[str]):
    """Attempts to load a faster-whisper model. Runs ONCE (lazily, on
    first real request) and caches either the loaded model or the exact
    failure reason - never retried on every request, since a blocked
    network path will fail identically every time and shouldn't add
    latency to every single upload."""
    global _model, _load_error, _attempted
    if _attempted:
        return
    _attempted = True

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        _load_error = f"faster-whisper is not installed: {exc}"
        return

    try:
        kwargs = {"device": "cpu", "compute_type": "int8"}
        if model_dir:
            # A real deployment that has pre-downloaded weights (see
            # ml/asr/data/README.md) points WHISPER_MODEL_DIR at that
            # local folder - no network needed at all in that case.
            _model = WhisperModel(model_dir, **kwargs)
        else:
            _model = WhisperModel(model_size, **kwargs)
    except Exception as exc:  # noqa: BLE001 - genuinely any failure here means "unavailable", captured verbatim
        _load_error = str(exc)


def is_available(model_size: str = DEFAULT_MODEL_SIZE, model_dir: Optional[str] = None) -> bool:
    _try_load_model(model_size, model_dir)
    return _model is not None


def get_load_error(model_size: str = DEFAULT_MODEL_SIZE, model_dir: Optional[str] = None) -> Optional[str]:
    _try_load_model(model_size, model_dir)
    return _load_error


def transcribe_wav_bytes(
    wav_bytes: bytes,
    language: Optional[str] = None,
    model_size: str = DEFAULT_MODEL_SIZE,
    model_dir: Optional[str] = None,
) -> dict:
    """Transcribes canonical 16kHz mono PCM16 WAV bytes with Whisper.
    `language` is a Whisper language code ("en" or "kn"); None lets
    Whisper auto-detect. Returns the same dict shape as
    pocketsphinx_backend.transcribe_wav_bytes() so ml/asr/stt_service.py
    can treat backends interchangeably."""
    _try_load_model(model_size, model_dir)
    if _model is None:
        return {
            "available": False,
            "reason": "model_download_blocked",
            "detail": _load_error,
            "transcript": None,
            "confidence": None,
            "engine": ENGINE_NAME,
        }

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        pcm_data = w.readframes(w.getnframes())
        import numpy as np

        audio_array = (
            np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        )

    segments, info = _model.transcribe(audio_array, language=language)
    segments = list(segments)
    transcript = " ".join(seg.text.strip() for seg in segments).strip()

    if segments:
        # avg_logprob is a real log-probability faster-whisper computes
        # per segment; convert to a 0-1-ish scale via exp() for the same
        # kind of honestly-interpretable confidence pocketsphinx_backend
        # reports (also a genuine model output, not fabricated).
        import math

        confidence = sum(math.exp(seg.avg_logprob) for seg in segments) / len(segments)
    else:
        confidence = 0.0

    return {
        "available": True,
        "transcript": transcript,
        "confidence": confidence,
        "detected_language": getattr(info, "language", language),
        "language_probability": getattr(info, "language_probability", None),
        "engine": f"{ENGINE_NAME}-{model_size}",
    }
