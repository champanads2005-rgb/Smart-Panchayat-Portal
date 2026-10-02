"""
Orchestrates speech-to-text across backends and languages for the Phase
8 voice complaint feature. This is the ONLY module app.py talks to for
transcription - it decides which backend(s) to try for a given
language and normalizes their results into one response shape, so
app.py never needs to know PocketSphinx or Whisper exist.

Design (see ml/asr/data/README.md for the full investigation this is
based on):
    - English ("en"): try Whisper first (best accuracy if its weights
      happen to be reachable), fall back to PocketSphinx (real, bundled,
      always available offline, but noticeably less accurate) if
      Whisper's weights can't be loaded.
    - Kannada ("kn"): Whisper is the ONLY backend that has ever
      supported Kannada here - PocketSphinx ships English only. In this
      sandbox Whisper's weights are not reachable (verified - see
      ml/asr/whisper_backend.py), so a Kannada request honestly reports
      unavailable with a clear reason and setup path, rather than
      running an English-only model on Kannada speech and returning
      garbage dressed up as a transcript.
"""

from __future__ import annotations

from typing import Optional

from ml.asr.audio_preprocessing import InvalidAudioError, validate_and_transcode_audio
from ml.asr import pocketsphinx_backend, whisper_backend

SUPPORTED_LANGUAGES = ["en", "kn"]
LANGUAGE_LABELS = {"en": "English", "kn": "Kannada"}

# Below this, a transcript is shown to the citizen as explicitly
# uncertain (still editable and usable - this is NOT a hard block, just
# an honest flag, matching the same "surface uncertainty, never hide it"
# pattern Phase 7 used for the image confidence threshold).
LOW_CONFIDENCE_THRESHOLD = 0.35


def transcribe_audio(
    raw_bytes: bytes,
    language: str,
    max_duration_seconds: int = 120,
    whisper_model_size: str = "tiny",
    whisper_model_dir: Optional[str] = None,
) -> dict:
    """Validates raw uploaded audio bytes and transcribes them in the
    requested language. Never raises for bad/unsupported input - always
    returns a dict with `available` set, so app.py can turn any outcome
    into a clean JSON response.
    """
    if language not in SUPPORTED_LANGUAGES:
        return {
            "available": False,
            "reason": "unsupported_language",
            "supported_languages": SUPPORTED_LANGUAGES,
            "transcript": None,
            "confidence": None,
            "wav_bytes": None,
        }

    try:
        validated = validate_and_transcode_audio(raw_bytes, max_duration_seconds=max_duration_seconds)
    except InvalidAudioError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "transcript": None,
            "confidence": None,
            "wav_bytes": None,
        }

    if validated.is_likely_silent:
        return {
            "available": False,
            "reason": "audio_appears_silent",
            "duration_seconds": validated.duration_seconds,
            "transcript": None,
            "confidence": None,
            # Still hand back the validated audio - a silent recording
            # is still a real recording the citizen might want attached,
            # even though there's nothing to transcribe.
            "wav_bytes": validated.wav_bytes,
        }

    whisper_lang_code = language  # Whisper and our own codes both use ISO 639-1 "en"/"kn"
    result = whisper_backend.transcribe_wav_bytes(
        validated.wav_bytes,
        language=whisper_lang_code,
        model_size=whisper_model_size,
        model_dir=whisper_model_dir,
    )

    if not result["available"] and language == "en":
        # English has a real, always-available fallback. Kannada does not.
        result = pocketsphinx_backend.transcribe_wav_bytes(validated.wav_bytes)

    if not result["available"]:
        return {
            "available": False,
            "reason": result.get("reason", "transcription_failed"),
            "detail": result.get("detail"),
            "duration_seconds": validated.duration_seconds,
            "transcript": None,
            "confidence": None,
            "wav_bytes": validated.wav_bytes,
        }

    confidence = result["confidence"]
    return {
        "available": True,
        "transcript": result["transcript"],
        "confidence": confidence,
        "is_low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
        "engine": result["engine"],
        "language": language,
        "duration_seconds": validated.duration_seconds,
        "wav_bytes": validated.wav_bytes,
    }


def translate_text(text: str, source_language: str, target_language: str) -> dict:
    """Optional Kannada<->English translation (master prompt requirement
    12). NOT implemented, on purpose, rather than faked: this project
    checked Argos Translate's own package index
    (raw.githubusercontent.com/argosopentech/argospm-index/main/index.json,
    reachable from this sandbox) and Kannada is not among ANY of its
    ~50 supported language codes; Google/Azure/AWS translation APIs
    would need an internet-reachable paid API and a key this project
    does not have. There is no suitable, available Kannada<->English
    translation system to integrate, so this always reports
    unavailable rather than returning a fabricated or English-passthrough
    'translation'. See ml/asr/data/README.md for the full investigation.
    """
    return {
        "available": False,
        "reason": "no_suitable_kannada_translation_system_available",
        "translated_text": None,
    }
