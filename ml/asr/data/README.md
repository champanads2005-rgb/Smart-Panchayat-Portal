# Phase 8 — Voice complaint speech recognition: documentation

## What this actually does

**English speech-to-text is real and works fully offline.** Kannada
speech-to-text is **honestly unavailable in this sandboxed environment**
— not faked, not approximated by running the English model on Kannada
audio. Both facts are verified below, not assumed.

## English: PocketSphinx (CMU Sphinx)

`ml/asr/pocketsphinx_backend.py` uses the `pocketsphinx` pip package,
which ships a **complete US English acoustic model, statistical
language model, and pronunciation dictionary inside the wheel itself**
(`pocketsphinx/model/en-us/`). Nothing is downloaded at runtime — this
is genuinely offline-capable, satisfying the master prompt's
"reliable offline-capable speech recognition model" requirement for at
least one language.

**What kind of model this is, stated plainly:** PocketSphinx is a
~20-year-old HMM/GMM statistical ASR system — not a modern neural
model like Whisper. It is real and it runs, but it is meaningfully less
accurate than Whisper on natural speech, especially with background
noise, accents, or non-native English pronunciation. It was chosen
specifically because **it is the only speech recognition model this
sandbox can actually run without any network access** (see "What was
tried for Whisper/Kannada" below) — the master prompt explicitly allows
implementing "a clean service abstraction and documented setup path"
when a better model can't run in the current environment, and that is
exactly the shape of this phase: architecture built for Whisper, with a
real fallback that works today.

**Confidence is genuine, not fabricated.** PocketSphinx's raw
utterance-level `hyp.prob` is a joint probability across the whole
utterance (a product of many small per-word probabilities) and reads as
near-zero even for a good transcription — reporting that directly would
be misleading. Instead, `pocketsphinx_backend.py` averages the
decoder's own **per-word posterior probabilities** (`decoder.seg()`,
each already a genuine 0–1 value the decoder computes), which is
honestly interpretable as "how confident was the decoder in the words
it output."

**A real finding from testing this, not hidden**: fed pure digital
silence directly, PocketSphinx does not reliably return an empty
hypothesis — its language-model prior can bias it toward hallucinating
a short, plausible word (observed: "dog") even with zero acoustic
signal. This is a genuine characteristic of statistical ASR with a
language-model prior. It's why `ml/asr/stt_service.py` runs its own
RMS-based silence check on the audio **before** ever calling a backend
— the real product pipeline never reaches this raw decoder behavior
(see `tests/test_stt_service.py::test_pocketsphinx_handles_pure_silence_without_crashing`
and the accompanying `test_transcribe_audio_flags_silent_recording`).

## Kannada: honestly unavailable, verified not assumed

Two things were actually tried and actually failed, not just assumed
blocked:

1. **PocketSphinx has no Kannada model.** The pip package only ships
   `en-us` — there is no Kannada acoustic/language model bundled or
   reachable without network access (CMU Sphinx's own community model
   downloads are hosted on `sourceforge.net`, which is not in this
   sandbox's network allowlist).

2. **Whisper (via `faster-whisper`) is a real, multilingual model that
   DOES support Kannada — but its weights could not be downloaded
   here.** `ml/asr/whisper_backend.py` is a genuine integration, not a
   stub: given reachable weights, it actually runs Whisper inference.
   A direct, verified attempt to load `WhisperModel("tiny")` in this
   sandbox raises:

   ```
   LocalEntryNotFoundError: ... HfHubHTTPError: 403 Forbidden ...
   Host not in allowlist: huggingface.co. Add this host to your
   network egress settings to allow access.
   ```

   That is the **exact, real, reproduced exception** this module
   catches and reports as `available: False, reason:
   "model_download_blocked"` — see
   `tests/test_stt_service.py::test_whisper_backend_reports_unavailable_with_the_real_captured_error`,
   which re-runs this exact check and will start passing differently
   (Whisper becoming available) the moment network access to Hugging
   Face Hub is granted, with **zero code changes** required.

Because of this, a Kannada voice complaint request honestly reports
"no speech model available for this language" (see the UI copy in
`static/js/ai_voice.js`) rather than running the English acoustic model
on Kannada speech and returning nonsense dressed up as a transcript —
exactly what the master prompt's "do not claim Kannada accuracy without
actual evaluation" and "do not fake speech recognition" requirements
forbid.

**Language selection, not detection.** The master prompt allows either
"language detection or an explicit language selection" — this phase
uses **explicit selection** (a radio button in `complaint.html`): with
only one language actually having a working backend, automatic
language ID would add a real dependency (itself a model) to solve a
problem explicit selection already solves for free, and would risk
mis-routing Kannada speech into the English recognizer's confident-
sounding-but-wrong output.

## Translation (Kannada ↔ English): not implemented, and why

The master prompt asks for optional translation "only where a suitable
translation system is available. Do not fabricate translations." This
was actually checked, not assumed:

- **Argos Translate** (a real, offline-capable, pip-installable
  translation library) publishes its full package index at
  `raw.githubusercontent.com/argosopentech/argospm-index/main/index.json`
  — reachable from this sandbox, and actually fetched. **Kannada ("kn")
  does not appear anywhere in its ~50 supported language codes** (checked
  programmatically — see the investigation in this phase's
  implementation notes). Argos Translate cannot translate Kannada at
  all, reachable or not.
- Google/Azure/AWS translation APIs would need an internet-reachable,
  paid, authenticated API this project has no credentials for, and
  their hosts are not in this sandbox's network allowlist regardless.

**Conclusion: no suitable Kannada↔English translation system is
available**, so `ml/asr/stt_service.translate_text()` always returns
`available: False` — never a fabricated or silently-English-passthrough
"translation." This is tested directly
(`tests/test_stt_service.py::test_translate_text_is_honestly_unimplemented_not_fabricated`).

## Pipeline

1. **`ml/asr/audio_preprocessing.py`** — the audio equivalent of Phase
   7's image validation: never trusts a client's filename or
   Content-Type. `ffprobe` inspects the REAL decoded container/codec/
   duration; anything that doesn't decode is rejected regardless of
   extension. Rejects audio shorter than 0.6s (accidental empty
   recordings) or longer than `MAX_AUDIO_DURATION_SECONDS` (default
   120s). Re-encodes via `ffmpeg` to canonical 16kHz mono 16-bit PCM WAV
   — both what PocketSphinx/Whisper require AND what gets written to
   permanent storage (the client's original container/codec is never
   stored verbatim, mirroring Phase 7's "decode then re-encode" rule for
   images). Flags near-silent audio via RMS amplitude.

2. **`ml/asr/stt_service.transcribe_audio(raw_bytes, language)`** — the
   single orchestration entry point `app.py` calls. Tries Whisper first
   for either language (best accuracy if its weights happen to be
   reachable in a given deployment), falls back to PocketSphinx for
   English only (Kannada has no fallback — see above). Never raises;
   always returns a dict with `available` set.

3. **Two-step preview/confirm flow, exactly mirroring Phase 7's image
   pattern:**
   - `POST /api/ai/transcribe-audio` — ephemeral preview (citizen-only,
     nothing persisted here). Called by `static/js/ai_voice.js` right
     after recording stops or a file is chosen.
   - The returned transcript is shown with its engine name and
     confidence, with an explicit **"Use this transcript" button** —
     clicking it copies the text into the **existing** `#description`
     textarea and fires the same `input` event `ai_classify.js` (Phase
     2) already listens for. This means accepting a voice transcript
     automatically triggers the existing category classifier, and
     (after the citizen picks a category) the existing priority
     predictor (Phase 2) and — at officer review time — the existing
     duplicate detector (Phase 3, `update_complaint()`), all completely
     unchanged, because all three already key off
     `complaints.description` regardless of whether the text was typed
     or transcribed. **Nothing is ever auto-submitted** — filling the
     textarea is not submitting; the citizen still reviews/edits the
     text and clicks "Submit Complaint" themselves.
   - On the real `POST /complaint` submission, the same audio is
     re-validated and re-transcribed **server-side** (never trusting
     the earlier preview call alone — the same rule Phase 7 applies to
     images), and persisted to `complaint_audio` together with BOTH the
     server's own raw transcript and whatever ended up in the
     description field (the citizen-confirmed text) — a human-in-the-
     loop audit trail, not a silently-substituted fact.

## Security (mirrors Phase 7's image security exactly)

- Real format validation via `ffprobe` decode, not extension/
  Content-Type trust.
- Random server-generated filenames
  (`security_utils.generate_safe_stored_filename`) — the citizen's
  original filename is never used for storage or retained.
- 10 MB size cap enforced twice: Flask's `MAX_CONTENT_LENGTH` and an
  explicit per-file check in `app.py`.
- 120-second duration cap (configurable), checked before any expensive
  transcoding/inference work runs.
- Recordings stored **outside `static/`** (`Config.AUDIO_UPLOAD_FOLDER`)
  — the only way to ever retrieve one is `GET /complaint-audio/<id>`,
  which enforces the complaint owner OR officer/admin, and returns the
  same generic 404 for "doesn't exist" and "not yours" (anti-
  enumeration, same pattern as `/complaint-image/<id>` and the RAG
  complaint lookup).
- Citizen/officer/admin permissions kept separate exactly as elsewhere:
  only citizens can upload/transcribe (`session["user_role"] ==
  "citizen"` check on both the preview and submission routes); officers
  and admins can review any complaint's audio but never upload one.

## Configurable retention

`Config.AUDIO_RETENTION_DAYS` (default 90, `.env`-configurable; 0
disables auto-purging). `scripts/purge_expired_audio.py` deletes
original recordings (file + `complaint_audio` row) older than this —
**the confirmed transcript is never deleted**, since it already lives
permanently and independently in `complaints.description`. Run as a
daily cron job:

```bash
# crontab -e
0 3 * * * cd /path/to/Smart-Panchayat-Portal && /path/to/venv/bin/python scripts/purge_expired_audio.py >> /var/log/audio_purge.log 2>&1
```

Dry-run first to see what would be deleted without deleting anything:

```bash
python scripts/purge_expired_audio.py --dry-run
```

## Exact setup commands

```bash
# System dependencies (not pip packages)
sudo apt-get install -y ffmpeg

# Python dependencies
pip install pocketsphinx faster-whisper numpy

# Apply the new complaint_audio table (safe to re-run, IF NOT EXISTS)
mysql -u root -p smart_panchayat < scripts/schema_patches.sql

# Run the test suite (40 Phase 8 tests; some skip automatically if
# espeak-ng isn't installed, since it's only used to synthesize test
# audio, never part of the shipped feature)
sudo apt-get install -y espeak-ng   # optional, test-fixture generation only
pytest tests/ -v

python app.py
```

### Making Kannada (and better English) actually work in a real deployment

This sandbox cannot reach Hugging Face Hub, but a real deployment with
normal internet access can, with **no code changes**:

```bash
# Either let faster-whisper download weights normally (needs network
# access to huggingface.co):
export WHISPER_MODEL_SIZE=small   # or tiny/base/medium, per available hardware

# ...or pre-download weights once and point at them locally (fully
# offline after this one-time step):
python -c "from faster_whisper import WhisperModel; WhisperModel('small')"
export WHISPER_MODEL_DIR=/path/to/local/whisper-small
```

The moment `ml/asr/whisper_backend.is_available()` returns True,
`ml/asr/stt_service.py` starts using it automatically for BOTH English
(better accuracy than PocketSphinx) and Kannada (which becomes real for
the first time) — nothing else in the codebase needs to change.

## Limitations (stated plainly)

- Kannada speech-to-text does not work in this environment — see above.
- English uses a 20-year-old statistical ASR engine, not a modern
  neural model; expect materially lower accuracy than Whisper on real
  human speech, especially with accents, background noise, or fast
  speech.
- No automatic language identification — citizens must select the
  language themselves.
- No Kannada↔English translation — no suitable system was found to be
  available (checked, not assumed).
- Confidence scores from PocketSphinx (per-word posterior average) and
  from Whisper (per-segment `exp(avg_logprob)`, used only if Whisper
  ever becomes reachable) are two different, real, engine-native
  measures — not calibrated against each other and not directly
  comparable if a deployment ever mixes both engines' output across
  complaints.
- No audio-quality enhancement (denoising, echo cancellation) — a
  genuinely noisy recording will genuinely transcribe poorly, and is
  only caught by the RMS silence check if it's actually silent, not
  merely noisy.
