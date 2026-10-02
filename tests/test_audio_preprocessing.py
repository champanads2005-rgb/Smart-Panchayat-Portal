"""
Tests for ml/asr/audio_preprocessing.py - real validation via ffprobe/
ffmpeg (already installed system binaries), using small synthetic audio
(silence and tone generated via ffmpeg's lavfi) purely to exercise
validation mechanics. This does NOT claim anything about real-world
speech - see ml/asr/data/README.md for the actual, honestly-measured
speech-recognition behavior.

Run: pytest tests/test_audio_preprocessing.py -v
"""

import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ml.asr.audio_preprocessing import (
    InvalidAudioError,
    TARGET_CHANNELS,
    TARGET_SAMPLE_RATE,
    validate_and_transcode_audio,
)


def _make_tone_wav_bytes(duration_seconds=2.0, freq=440, sample_rate=22050) -> bytes:
    """A real, audible (non-silent) synthetic WAV via ffmpeg's lavfi
    sine generator - genuinely has energy, unlike pure silence."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency={freq}:duration={duration_seconds}:sample_rate={sample_rate}",
            "-f", "wav", "-",
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _make_silence_wav_bytes(duration_seconds=2.0, sample_rate=16000) -> bytes:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", str(duration_seconds),
            "-f", "wav", "-",
        ],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_valid_tone_audio_is_accepted_and_transcoded():
    raw = _make_tone_wav_bytes()
    result = validate_and_transcode_audio(raw)
    assert result.duration_seconds == pytest.approx(2.0, abs=0.1)
    assert result.is_likely_silent is False
    assert result.rms > 0

    with wave.open(__import__("io").BytesIO(result.wav_bytes), "rb") as w:
        assert w.getframerate() == TARGET_SAMPLE_RATE
        assert w.getnchannels() == TARGET_CHANNELS
        assert w.getsampwidth() == 2  # 16-bit PCM


def test_transcodes_from_a_different_source_sample_rate():
    """Source is 22050Hz (a common non-16kHz rate) - output must always
    be the canonical 16kHz regardless of input rate."""
    raw = _make_tone_wav_bytes(sample_rate=22050)
    result = validate_and_transcode_audio(raw)
    with wave.open(__import__("io").BytesIO(result.wav_bytes), "rb") as w:
        assert w.getframerate() == 16000


def test_silent_audio_is_flagged_but_not_rejected():
    """Silence is still a valid, decodable recording - just flagged so
    the caller (ml/asr/stt_service.py) can skip wasting time on STT and
    report 'no speech detected' instead of a nonsense transcript."""
    raw = _make_silence_wav_bytes()
    result = validate_and_transcode_audio(raw)
    assert result.is_likely_silent is True
    assert result.rms < 80


def test_empty_bytes_rejected():
    with pytest.raises(InvalidAudioError):
        validate_and_transcode_audio(b"")


def test_garbage_bytes_rejected():
    with pytest.raises(InvalidAudioError) as exc_info:
        validate_and_transcode_audio(b"this is not audio data at all, just text" * 20)
    assert "not_a_valid_audio_file" in str(exc_info.value)


def test_too_short_audio_rejected():
    raw = _make_tone_wav_bytes(duration_seconds=0.1)
    with pytest.raises(InvalidAudioError) as exc_info:
        validate_and_transcode_audio(raw)
    assert "audio_too_short" in str(exc_info.value)


def test_too_long_audio_rejected():
    raw = _make_tone_wav_bytes(duration_seconds=3.0)
    with pytest.raises(InvalidAudioError) as exc_info:
        validate_and_transcode_audio(raw, max_duration_seconds=2)
    assert "audio_too_long" in str(exc_info.value)


def test_max_duration_is_configurable():
    """The same 3-second clip that's rejected with a 2-second cap must
    be accepted when the cap is raised - proves the limit is actually
    being applied, not hardcoded."""
    raw = _make_tone_wav_bytes(duration_seconds=3.0)
    result = validate_and_transcode_audio(raw, max_duration_seconds=10)
    assert result.duration_seconds == pytest.approx(3.0, abs=0.1)
