"""
English speech-to-text backend using PocketSphinx (CMU Sphinx).

This is genuinely real and genuinely offline: the `pocketsphinx` pip
package ships a complete US English acoustic model, language model, and
pronunciation dictionary INSIDE the wheel itself
(pocketsphinx/model/en-us/) - nothing is downloaded at runtime, and
nothing here is a hardcoded/simulated transcription. See
ml/asr/data/README.md for exactly why this (rather than Whisper) is the
backend that actually runs in this environment, and for the honest,
measured accuracy characteristics of this specific engine.

IMPORTANT, stated plainly: PocketSphinx is a ~20-year-old HMM/GMM
statistical ASR system, not a modern neural model. It is real and it
runs, but it is meaningfully less accurate than Whisper on natural
speech, especially with background noise, accents, or non-native
English. It also only ships an ENGLISH model - there is no Kannada
model bundled or reachable in this sandbox (see whisper_backend.py and
ml/asr/data/README.md for what was actually tried for Kannada).

A real finding from actually testing this (see
tests/test_stt_service.py): fed pure digital silence directly, this
decoder does NOT reliably return an empty hypothesis - its language
model prior can bias it toward hallucinating a short, plausible word
(observed: "dog") even with zero acoustic signal. This is a genuine
characteristic of statistical ASR with a language-model prior, not a
bug in this wrapper. It is exactly why
ml/asr/stt_service.transcribe_audio() runs its own RMS-based silence
check on the audio BEFORE ever calling a backend - the real product
pipeline never reaches this raw decoder behavior; only a direct,
bypassing call to this module would.
"""

from __future__ import annotations

import io
import re
import wave
from pathlib import Path
from typing import Optional

ENGINE_NAME = "pocketsphinx_en_us"

# Tokens PocketSphinx emits that are not actual spoken words - excluded
# both from the transcript text and from the confidence calculation
# below.
_NON_WORD_TOKENS = {"<s>", "</s>", "<sil>", "[SPEECH]", "[NOISE]"}

_decoder_factory = None
_load_error: Optional[str] = None

_ALT_PRONUNCIATION_RE = re.compile(r"\(\d+\)$")


def _strip_alt_pronunciation_tag(word: str) -> str:
    """PocketSphinx's dictionary marks alternate pronunciations of the
    same word as e.g. "really(2)" - strip that for display; it is not
    part of the actual transcribed word."""
    return _ALT_PRONUNCIATION_RE.sub("", word)


def _get_decoder_factory():
    """Lazily builds (once) a callable that returns a fresh Decoder
    configured with the bundled en-us model. Lazy so importing this
    module never has a side effect / never slows down app startup, and
    a missing/broken pocketsphinx installation degrades this backend to
    unavailable rather than crashing the whole app at import time."""
    global _decoder_factory, _load_error
    if _decoder_factory is not None or _load_error is not None:
        return _decoder_factory

    try:
        from pocketsphinx import Decoder, get_model_path

        model_dir = Path(get_model_path()) / "en-us"
        config_kwargs = {
            "hmm": str(model_dir / "en-us"),
            "lm": str(model_dir / "en-us.lm.bin"),
            "dict": str(model_dir / "cmudict-en-us.dict"),
        }
        for key, path in config_kwargs.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"Expected bundled model file missing: {path}")

        def factory():
            return Decoder(**config_kwargs)

        # Build one decoder now to fail fast (and loudly, once) if the
        # bundled model is somehow broken/missing, rather than failing
        # silently on every request.
        factory()
        _decoder_factory = factory
    except Exception as exc:  # noqa: BLE001 - any failure here degrades this backend gracefully
        _load_error = f"PocketSphinx English model unavailable: {exc}"

    return _decoder_factory


def is_available() -> bool:
    return _get_decoder_factory() is not None


def get_load_error() -> Optional[str]:
    _get_decoder_factory()  # ensure the lazy check has run at least once
    return _load_error


def transcribe_wav_bytes(wav_bytes: bytes) -> dict:
    """Transcribes a canonical 16kHz mono PCM16 WAV (as produced by
    ml/asr/audio_preprocessing.py) using the bundled English model.

    Returns a dict with keys: available, transcript, confidence,
    word_confidences, engine. Confidence is the arithmetic mean of
    PocketSphinx's own per-word posterior probabilities (decoder.seg(),
    each already a real 0-1 value the decoder itself computes) over the
    actual spoken words in the 1-best hypothesis - NOT the raw utterance
    joint-probability (`hyp.prob`), which is a product of many small
    probabilities and reads as near-zero even for a good transcription;
    the per-word posterior average is the honestly interpretable number.
    An empty hypothesis (nothing recognized) returns confidence 0.0 and
    an empty transcript, not an error - that is a valid outcome for
    silent/unintelligible audio.
    """
    factory = _get_decoder_factory()
    if factory is None:
        return {
            "available": False,
            "reason": _load_error,
            "transcript": None,
            "confidence": None,
            "engine": ENGINE_NAME,
        }

    decoder = factory()

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            return {
                "available": False,
                "reason": "unexpected_wav_format_expected_16k_mono_pcm16",
                "transcript": None,
                "confidence": None,
                "engine": ENGINE_NAME,
            }
        pcm_data = w.readframes(w.getnframes())

    decoder.start_utt()
    decoder.process_raw(pcm_data, False, True)
    decoder.end_utt()

    hyp = decoder.hyp()
    if hyp is None or not hyp.hypstr:
        return {
            "available": True,
            "transcript": "",
            "confidence": 0.0,
            "word_confidences": [],
            "engine": ENGINE_NAME,
        }

    word_confidences = [
        {"word": _strip_alt_pronunciation_tag(seg.word), "confidence": float(seg.prob)}
        for seg in decoder.seg()
        if seg.word not in _NON_WORD_TOKENS
    ]
    if word_confidences:
        confidence = sum(wc["confidence"] for wc in word_confidences) / len(word_confidences)
    else:
        confidence = 0.0

    return {
        "available": True,
        "transcript": hyp.hypstr,
        "confidence": confidence,
        "word_confidences": word_confidences,
        "engine": ENGINE_NAME,
    }
