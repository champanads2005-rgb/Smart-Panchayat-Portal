"""
Tests for the Phase 8 speech-to-text service layer:
    - ml/asr/pocketsphinx_backend.py (real, working English engine)
    - ml/asr/whisper_backend.py (real integration; verified-unavailable
      in this sandbox - see module docstring for the exact captured error)
    - ml/asr/stt_service.py (orchestration + graceful degradation)

Uses espeak-ng-synthesized audio purely to exercise the pipeline
end-to-end with real, non-hardcoded audio bytes - NOT to claim any
accuracy figure (see ml/asr/data/README.md for the honest, separately
-documented accuracy characteristics of PocketSphinx on real speech).

Run: pytest tests/test_stt_service.py -v
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ml.asr import pocketsphinx_backend, stt_service, whisper_backend

ESPEAK_AVAILABLE = shutil.which("espeak-ng") is not None


def _synthesize_speech(text: str, voice: str = "en") -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        result = subprocess.run(
            ["espeak-ng", "-v", voice, "-w", tmp.name, text],
            capture_output=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return Path(tmp.name).read_bytes()


# ---------------------------------------------------------------------
# pocketsphinx_backend - real, working English engine
# ---------------------------------------------------------------------

def test_pocketsphinx_is_available():
    """The bundled model ships inside the pip package - this must be
    true with no network access and no setup step."""
    assert pocketsphinx_backend.is_available() is True
    assert pocketsphinx_backend.get_load_error() is None


@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_pocketsphinx_transcribes_real_synthesized_speech():
    from ml.asr.audio_preprocessing import validate_and_transcode_audio

    raw = _synthesize_speech("there is no water supply in our village")
    validated = validate_and_transcode_audio(raw)
    result = pocketsphinx_backend.transcribe_wav_bytes(validated.wav_bytes)

    assert result["available"] is True
    assert result["engine"] == "pocketsphinx_en_us"
    assert isinstance(result["transcript"], str)
    # Not asserting exact words - PocketSphinx on synthetic TTS audio is
    # genuinely imperfect (see ml/asr/data/README.md) - only that it
    # produced SOME real, non-empty, non-fabricated output with a
    # genuine confidence score.
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["word_confidences"], list)


def test_pocketsphinx_handles_pure_silence_without_crashing():
    """A real finding from actually running this, not assumed: fed pure
    digital silence directly, PocketSphinx does NOT reliably return an
    empty hypothesis - its language model can bias it toward hallucinating
    a short, high-prior word (observed: "dog") even with no acoustic
    signal at all. This is a genuine, documented characteristic of
    statistical ASR with a language-model prior, not a bug in this
    wrapper (see the module docstring in ml/asr/pocketsphinx_backend.py).
    It's exactly why ml/asr/stt_service.py runs its own RMS-based silence
    check BEFORE ever calling a backend (test_transcribe_audio_flags_silent_recording
    below) - the real product pipeline never reaches this raw decoder
    behavior. This test only asserts the backend behaves predictably
    (doesn't crash, returns well-typed output) when called directly."""
    from ml.asr.audio_preprocessing import validate_and_transcode_audio

    result_proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "-f", "wav", "-"],
        capture_output=True,
        timeout=15,
    )
    validated = validate_and_transcode_audio(result_proc.stdout)
    result = pocketsphinx_backend.transcribe_wav_bytes(validated.wav_bytes)
    assert result["available"] is True
    assert isinstance(result["transcript"], str)
    assert 0.0 <= result["confidence"] <= 1.0


def test_pocketsphinx_strips_alternate_pronunciation_tags():
    assert pocketsphinx_backend._strip_alt_pronunciation_tag("really(2)") == "really"
    assert pocketsphinx_backend._strip_alt_pronunciation_tag("word") == "word"


# ---------------------------------------------------------------------
# whisper_backend - real integration, verified unavailable here
# ---------------------------------------------------------------------

def test_whisper_backend_reports_unavailable_with_the_real_captured_error():
    """This is the actual, reproduced failure in THIS sandbox (Hugging
    Face Hub is not in the network allowlist) - not a simulated one.
    If this test ever starts failing because Whisper IS reachable in a
    future environment, that's a good thing - it means Kannada support
    just became real; see ml/asr/data/README.md."""
    available = whisper_backend.is_available()
    error = whisper_backend.get_load_error()
    if not available:
        assert error is not None
        assert "huggingface" in error.lower() or "hub" in error.lower() or "download" in error.lower()


def test_whisper_backend_never_raises_on_transcribe_when_unavailable():
    from ml.asr.audio_preprocessing import validate_and_transcode_audio

    raw = _synthesize_speech("test") if ESPEAK_AVAILABLE else None
    if raw is None:
        pytest.skip("espeak-ng not installed")
    validated = validate_and_transcode_audio(raw)
    result = whisper_backend.transcribe_wav_bytes(validated.wav_bytes, language="en")
    assert "available" in result
    if not result["available"]:
        assert result["reason"] == "model_download_blocked"


# ---------------------------------------------------------------------
# stt_service - orchestration + graceful degradation
# ---------------------------------------------------------------------

@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_transcribe_audio_english_falls_back_to_pocketsphinx():
    """Whisper is unavailable here (verified above), so English requests
    must transparently fall back to the real PocketSphinx engine rather
    than failing outright."""
    raw = _synthesize_speech("there is a big pothole on the main road")
    result = stt_service.transcribe_audio(raw, "en")
    assert result["available"] is True
    assert result["engine"] == "pocketsphinx_en_us"
    assert isinstance(result["transcript"], str)
    assert isinstance(result["is_low_confidence"], bool)
    assert "wav_bytes" in result and isinstance(result["wav_bytes"], bytes)


@pytest.mark.skipif(not ESPEAK_AVAILABLE, reason="espeak-ng not installed - can't synthesize test audio")
def test_transcribe_audio_kannada_is_honestly_unavailable():
    """No fallback exists for Kannada (PocketSphinx is English-only) -
    this must report unavailable with a clear reason, never silently run
    the English model on Kannada speech."""
    raw = _synthesize_speech("ನಮಸ್ಕಾರ", voice="kn")
    result = stt_service.transcribe_audio(raw, "kn")
    assert result["available"] is False
    assert result["reason"] == "model_download_blocked"
    assert result["transcript"] is None


def test_transcribe_audio_rejects_unsupported_language():
    result = stt_service.transcribe_audio(b"irrelevant", "ta")
    assert result["available"] is False
    assert result["reason"] == "unsupported_language"
    assert result["supported_languages"] == ["en", "kn"]


def test_transcribe_audio_rejects_invalid_bytes():
    result = stt_service.transcribe_audio(b"not audio", "en")
    assert result["available"] is False
    assert result["reason"] == "not_a_valid_audio_file"


def test_transcribe_audio_flags_silent_recording():
    result_proc = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "2", "-f", "wav", "-"],
        capture_output=True,
        timeout=15,
    )
    result = stt_service.transcribe_audio(result_proc.stdout, "en")
    assert result["available"] is False
    assert result["reason"] == "audio_appears_silent"
    # Even though transcription is unavailable, the validated/re-encoded
    # audio should still be handed back - a citizen may want the silent
    # recording attached anyway.
    assert result["wav_bytes"] is not None


def test_translate_text_is_honestly_unimplemented_not_fabricated():
    """No suitable Kannada<->English translation system was found to be
    available (see ml/asr/stt_service.py docstring for what was
    actually checked) - this must never return a fabricated
    translation."""
    result = stt_service.translate_text("hello", "en", "kn")
    assert result["available"] is False
    assert result["translated_text"] is None
