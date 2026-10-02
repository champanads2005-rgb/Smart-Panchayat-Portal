"""
Shared audio validation + canonical re-encoding for the Phase 8 voice
complaint pipeline.

Design mirrors ml/cv/preprocessing.py from Phase 7 on purpose: never
trust a client-supplied filename or Content-Type header - decode the
actual bytes with a real tool (here, ffprobe/ffmpeg, which are already
installed on this system) and validate what they actually contain.

Pipeline:
    1. Write the uploaded bytes to a temp file (ffmpeg/ffprobe need a
       real file path, not an in-memory buffer).
    2. `ffprobe` inspects the REAL container/codec/duration - rejects
       anything that isn't decodable audio, regardless of what
       extension or Content-Type the browser sent.
    3. Reject audio that is too long (MAX_AUDIO_DURATION_SECONDS) or
       implausibly short/silent (likely an accidental empty recording).
    4. `ffmpeg` re-encodes to a canonical 16kHz mono 16-bit PCM WAV -
       both what ml/asr/pocketsphinx_backend.py requires AND what gets
       written to permanent storage (so we never store the client's
       original, unvalidated container/codec verbatim - the same
       "decode then re-encode, never pass through raw bytes" principle
       Phase 7 used for images).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Optional

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1

MIN_DURATION_SECONDS = 0.6  # shorter than this is almost certainly an accidental empty recording
DEFAULT_MAX_DURATION_SECONDS = 120

# A recording that decodes fine but is near-silent throughout - most
# likely the microphone wasn't actually capturing anything. RMS is on a
# 0-32767 (16-bit PCM) scale.
SILENCE_RMS_THRESHOLD = 80


class InvalidAudioError(ValueError):
    """Raised for any audio that fails validation - undecodable data, a
    duration outside allowed bounds, or audio that is silent throughout.
    Callers turn this into a clean 4xx JSON error instead of a 500."""


@dataclass
class ValidatedAudio:
    wav_bytes: bytes          # canonical 16kHz mono PCM16 WAV, ready for STT and storage
    duration_seconds: float
    rms: float
    is_likely_silent: bool


def _run(cmd: list) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=30)


def _probe_duration_seconds(path: str) -> Optional[float]:
    result = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path,
    ])
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
        return float(data["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def _compute_rms(wav_path: str) -> float:
    """Root-mean-square amplitude of a 16-bit PCM WAV, used only to flag
    likely-silent recordings. Implemented with numpy rather than the
    stdlib `audioop` module, which is deprecated and slated for removal
    in Python 3.13."""
    import numpy as np

    with wave.open(wav_path, "rb") as w:
        frames = w.readframes(w.getnframes())
        if not frames:
            return 0.0
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
        return float(np.sqrt(np.mean(np.square(samples))))


def validate_and_transcode_audio(
    raw_bytes: bytes, max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS
) -> ValidatedAudio:
    """Validates raw uploaded audio bytes and returns a canonical 16kHz
    mono PCM16 WAV version, or raises InvalidAudioError. This is the
    ONLY function that should ever be trusted to turn an upload into
    something safe to store or feed to a speech recognizer."""
    if not raw_bytes:
        raise InvalidAudioError("empty_file")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "upload.bin")
        with open(src_path, "wb") as f:
            f.write(raw_bytes)

        duration = _probe_duration_seconds(src_path)
        if duration is None:
            raise InvalidAudioError("not_a_valid_audio_file")

        if duration < MIN_DURATION_SECONDS:
            raise InvalidAudioError("audio_too_short")
        if duration > max_duration_seconds:
            raise InvalidAudioError(f"audio_too_long:max_{max_duration_seconds}s")

        wav_path = os.path.join(tmpdir, "converted.wav")
        result = _run([
            "ffmpeg", "-y", "-i", src_path,
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            "-sample_fmt", "s16",
            "-f", "wav",
            wav_path,
        ])
        if result.returncode != 0 or not os.path.isfile(wav_path):
            raise InvalidAudioError("decode_failed")

        rms = _compute_rms(wav_path)
        is_likely_silent = rms < SILENCE_RMS_THRESHOLD

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()

        return ValidatedAudio(
            wav_bytes=wav_bytes,
            duration_seconds=duration,
            rms=rms,
            is_likely_silent=is_likely_silent,
        )
