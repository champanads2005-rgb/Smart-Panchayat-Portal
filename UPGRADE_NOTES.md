# Smart Panchayat Portal — Phase 1 Upgrade Notes

Scope of this pass: **security/architecture fixes + one genuine, trained ML
feature** (complaint category classification). This was a deliberate choice
to do a smaller set of things correctly rather than attempt the full
50-section master prompt in one pass and risk faking parts of it.

## A. Audit findings (before changes)

- Passwords stored and compared in **plaintext**.
- MySQL root password and Flask `secret_key` **hardcoded in source**
  (`app.py`, `database.py`).
- No CSRF protection on any POST form.
- No AI/ML anywhere — category and priority were manually picked dropdowns.
- Parameterized queries were already used correctly (no obvious SQL
  injection), which was preserved.
- No `.env`, no `requirements.txt`, no tests, no API layer.

## B. What changed

1. **Secrets moved to environment variables.** `config.py` loads
   `SECRET_KEY`, `DB_HOST/USER/PASSWORD/NAME` from `.env` (see
   `.env.example`). Nothing is hardcoded anymore. If `SECRET_KEY` is unset,
   a random one is generated per-process with a visible warning (dev
   convenience only — set a real one for anything beyond local testing).

2. **Password hashing.** `security_utils.py` uses Werkzeug's
   `generate_password_hash` / `check_password_hash` (salted PBKDF2-SHA256).
   - New registrations are hashed.
   - Login checks the hash if the stored value looks like one, and
     **transparently re-hashes** a legacy plaintext password the moment
     that user next logs in successfully, so old accounts migrate without
     forcing a password reset.
   - `scripts/migrate_passwords.py` does a bulk one-time migration for any
     accounts that never log in again on their own.

3. **CSRF protection.** A session-bound token is injected into every
   template via `{{ csrf_token }}` and validated on every non-JSON POST in
   a `before_request` hook (`security_utils.validate_csrf`). Added the
   hidden field to `login.html`, `register.html`, `complaint.html`,
   `update_complaint.html`.

4. **Real, trained complaint classifier** (`ml/`):
   - `ml/data/generate_synthetic_data.py` — generates a labeled,
     **clearly-marked-synthetic** dataset (840 rows, 7 categories matching
     the existing `complaints.category` ENUM) because no real historical
     complaint corpus was provided. Templated sentences with randomized
     ward/duration/intensifier variation, seeded for reproducibility.
   - `ml/training/train_classifier.py` — trains and compares **two real
     models** (TF-IDF + Logistic Regression, TF-IDF + Linear SVM with
     probability calibration), evaluates both with accuracy, macro F1,
     per-class precision/recall/F1, and a confusion matrix on a genuine
     80/20 held-out split, then saves the better one.
   - `ml/inference/classify.py` — loads the saved model **once** (not
     per-request), exposes `classify_complaint(text)` returning category +
     real confidence (from `predict_proba`) + top contributing terms
     (linear-model coefficient × TF-IDF weight, a simple explainability
     signal) + graceful fallback if the model file is missing.
   - `POST /api/ai/classify` — new authenticated JSON API endpoint.
   - `complaint.html` + `static/js/ai_classify.js` — as the citizen types
     the description, the form calls the API and shows *"AI suggests
     category: X (confidence Y%)"* with a **"Use this" button**. The
     citizen must click it — nothing is auto-applied, and the manual
     dropdown remains the source of truth. This is the human-in-the-loop
     pattern the master prompt calls for, kept intentionally simple.

## C. Actual model results (not invented — see `ml/evaluation/`)

Both `tfidf_logreg` and `tfidf_linear_svm` scored **100% accuracy / macro
F1 on the held-out synthetic test set** (168 samples, 24 per class).
Logistic Regression was selected (tie-break: simpler, cheaper to serve,
directly exposes calibrated probabilities without a calibration wrapper).

**This 100% number should not be quoted as real-world accuracy in a viva
or report.** It reflects the classifier learning the synthetic template
vocabulary near-perfectly, not genuine generalization to unpredictable
citizen phrasing. I spot-checked it on hand-written sentences that don't
match any training template and it held up reasonably (see below), but
that's informal, not a benchmark.

```
"We have not received drinking water for three days in our area."
  -> Water (0.83)
"There is a huge pothole on the main road near the school."
  -> Road (0.90)
"Garbage has not been collected for a week and it smells terrible."
  -> Sanitation (0.66)
"The street light near my house has been broken for a month."
  -> Street Light (0.85)
```

**For a viva, the honest framing is:** *"This is a real, trained,
evaluated NLP pipeline with genuine train/test methodology. Because no
real complaint history existed, it was bootstrapped on documented
synthetic data. The near-perfect test score reflects that synthetic
templates are easy to separate, not real-world performance — the model
generalizes reasonably on informal spot-checks but would need real
citizen-submitted complaints for a trustworthy accuracy claim."* That is a
more defensible answer than presenting the 100% as if it were measured on
real data.

## D. Phase 2 — Priority prediction

Same discipline as Phase 1: real trained models, real evaluation,
documented synthetic data, human-in-the-loop UI.

1. **`ml/data/generate_priority_data.py`** — synthetic dataset (900 rows)
   built by combining three signals into a label: text severity (urgency
   phrases), a category baseline risk, and a `recurrence_count` feature,
   plus random noise, then bucketed into Low/Medium/High **by quantile**
   (not fixed thresholds) so all three classes have enough samples to
   evaluate meaningfully. Kept to the existing 3-level ENUM
   (`High`/`Medium`/`Low`) rather than adding a 4th "Critical" tier that
   would need a schema migration.

2. **`ml/training/train_priority.py`** — combines TF-IDF text +
   one-hot category + the numeric `recurrence_count` via a
   `ColumnTransformer`, compares Logistic Regression vs Random Forest,
   evaluates on a genuine 80/20 split, selects by macro F1.

3. **Real results** (`ml/evaluation/priority_metrics.txt`):
   Random Forest selected — **58.3% accuracy, 0.53 macro F1** against a
   33% random-guess baseline for 3 balanced classes. This is a much more
   defensible number than the classifier's 100%, because priority is
   genuinely harder to separate — it's the honest result of a harder,
   noisier task, not a tuning failure. High-priority recall is the
   weakest class (0.28) — worth flagging in a viva as a direction for
   more data/features rather than papering over it.

4. **`ml/inference/priority.py`** — loads the model once; explainability
   here is a real **ablation**: it re-runs the model with each input
   group (text / category / recurrence) swapped for a neutral baseline
   and reports the actual probability drop as that factor's contribution
   — not an invented "confidence" number.

5. **Real feature, not synthetic, at inference time**: `app.py` adds
   `get_recent_similar_count()`, which queries the actual `complaints`
   and `users` tables for how many same-category complaints came from the
   citizen's own village in the last 30 days. Only the *training label*
   is synthetic — the recurrence feature used at prediction time is
   computed from the real database.

6. **`POST /api/ai/predict-priority`** — authenticated; looks up the
   citizen's village server-side (never trusts a client-supplied
   location), computes recurrence, and returns priority + confidence +
   factors. `complaint.html` + `static/js/ai_priority.js` show *"AI
   suggests priority: X (confidence Y%) — driven by: ..."* with a **"Use
   this"** button once both description and category are filled in.

**Still deferred:** duplicate detection, resolution-time prediction,
escalation risk, and everything else listed below.

## E. Phase 3 — Duplicate / related complaint detection

**Honest limitation stated up front:** the master prompt calls for
transformer sentence embeddings (Sentence-BERT style). This sandbox has
no network route to a model hub (huggingface.co isn't reachable here), so
a pretrained embedding model can't actually be downloaded and run. Rather
than fake that with a model that was never loaded, this phase implements
a real, weaker, fully-offline alternative and documents the gap plainly.

1. **`ml/inference/duplicate.py`** — similarity is computed as
   `0.5 × cosine(TF-IDF word 1-2gram) + 0.5 × cosine(TF-IDF char 3-5gram)`
   between the target complaint and candidates, fit fresh per comparison
   set (fine at this scale). Word n-grams catch shared vocabulary; char
   n-grams catch shared phrasing even through small wording changes.

2. **Threshold (0.12) is empirically calibrated, not guessed** — I hand-
   wrote duplicate pairs (e.g. "No water supply in Ward 4 for three
   days." vs "Water hasn't been coming to houses in Ward 4.") and
   unrelated pairs, scored them, and picked a threshold that separated
   them cleanly in that small test (duplicates scored 0.13–0.44,
   unrelated scored 0.00–0.05). This is real calibration against real
   examples, but it's a handful of examples, not a validated benchmark —
   recalibrate once real complaint volume exists.

3. **Where it's used**: officer/admin complaint review
   (`/update-complaint/<id>`) now queries the real `complaints` table for
   other complaints in the same category, computes similarity against
   the real candidate descriptions, and shows a "Possibly Related
   Complaints" panel with each match's id, status, and similarity %.
   Nothing is auto-merged or auto-linked — per the master prompt's own
   instruction not to automatically delete/merge complaints, this phase
   is read-only/informational; merge/link tooling is future work.

4. **Upgrade path, documented not built**: if this runs somewhere with
   real internet access, swap the TF-IDF similarity in
   `_cosine_to_all`/`find_similar_complaints` for
   `sentence-transformers` embeddings + cosine similarity — the function
   signature (`find_similar_complaints(text, candidates, threshold)`)
   is designed to stay the same so nothing in `app.py` would need to
   change.



## G. How to run

```bash
cd Smart-Panchayat-Portal
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env        # then edit .env with your real DB password + a random SECRET_KEY

# apply the missing complaint_updates table (needed for the app to work at all,
# and for resolution-time features) - safe to run on an existing DB
mysql -u root -p smart_panchayat < scripts/schema_patches.sql

# one-time: train the classifier (already done once, but reproducible)
python ml/training/train_classifier.py

# one-time: train the priority predictor (already done once, but reproducible)
python ml/training/train_priority.py

# one-time: train the resolution-time predictor (checks for real data first,
# falls back to synthetic bootstrap automatically - see UPGRADE_NOTES.md section H)
python ml/data/extract_resolution_history.py   # writes resolution_real.csv IF enough real data exists
python ml/training/train_resolution.py

# build the RAG knowledge-base index (Phase 6)
python rag/ingestion/ingest.py
python rag/evaluation/evaluate_rag.py   # retrieval metrics now; generation metrics too if Ollama is running

# run the test suite
pytest tests/ -v   # 103 tests as of Phase 6

# if you have an existing DB with plaintext passwords from before this upgrade:
python scripts/migrate_passwords.py

python app.py
```

## F. (superseded — see section J for the current deferred list, updated through Phase 6)


## G. Phase 4 audit finding (before implementing)

`app.py` has always read from and written to a `complaint_updates` table
(complaint creation, every status change, and the citizen tracking page
all depend on it), but **`complaint.sql` never defined this table**. On a
database built strictly from the provided SQL file, complaint submission
and status updates would fail outright with a missing-table error. This
also matters specifically for Phase 4: `complaints.updated_at` bumps on
*any* row change (a priority edit, a remark, anything), not just the
transition to "Resolved" — it cannot be used to measure resolution time.
Only `complaint_updates` (status history with timestamps) can. Fixed via
`scripts/schema_patches.sql` (`CREATE TABLE IF NOT EXISTS`, safe to run on
an existing DB).

## H. Phase 4 — Resolution-time prediction (regression)

1. **Real-data path built first, synthetic is the fallback, not the
   default.** `ml/data/extract_resolution_history.py` queries the live
   DB for complaints with a real `Resolved` timestamp in
   `complaint_updates`, computes `resolution_hours` via
   `ml/data/temporal_features.py`, and requires at least 50 valid rows
   before it will write a training file — below that, it says so and
   defers to synthetic data rather than training on too little to be
   meaningful. **On this project's fresh install, that real path
   correctly finds 0 rows** (no complaints have ever been resolved), so
   the model shipped in this phase was trained on the synthetic
   fallback — the code will automatically switch to real data the
   moment 50+ resolved complaints with valid timestamps exist, no
   changes needed.

2. **Leakage prevention, actually enforced, actually tested.**
   `ml/data/temporal_features.py` is a small set of pure functions
   (`compute_recurrence_count`, `resolution_hours`) with no DB
   dependency, so the leakage rule — a complaint's recurrence count may
   only include *other* complaints created strictly *before* it — is
   directly unit-tested (`tests/test_temporal_features.py`), including a
   test that a record created *after* the target complaint is correctly
   excluded. `priority` is read from its current DB value, which is a
   documented, un-fixed leakage risk if an officer edits priority after
   the fact (no priority-history table exists to prevent it — flagged,
   not silently ignored).

3. **Synthetic bootstrap dataset**
   (`ml/data/generate_resolution_time_data.py`, 1,000 rows): every
   assumption is written into the file as a comment, not hidden in code —
   category baseline hours, "higher priority resolves faster" and
   "higher recurrence resolves marginally faster" are stated
   assumptions, not measured facts, and resolution time is sampled
   log-normally (positive, right-skewed) rather than with symmetric
   noise, which is the statistically appropriate choice for a duration
   target.

4. **Three real models compared** on a genuine 80/20 split, each
   wrapped to train on `log(resolution_hours)` and exponentiate
   predictions back (confirmed empirically during development to reduce
   RMSE vs. training directly on raw hours — the right transform for a
   skewed positive target, not just a default choice):

   | Model | MAE (hrs) | RMSE (hrs) | R² |
   |---|---|---|---|
   | Linear Regression (selected) | 21.13 | 32.43 | 0.445 |
   | Random Forest | 22.36 | 34.47 | 0.373 |
   | Gradient Boosting | 22.07 | 33.76 | 0.399 |

   XGBoost specifically was **not** added as a fourth candidate — sklearn's
   built-in `GradientBoostingRegressor` was used instead to avoid pulling
   in a new heavyweight dependency for one model in this phase. That's a
   documented substitution, not a hidden shortcut; it answers the same
   "linear vs. bagging vs. boosting" comparison.

   R² of 0.445 is modest and I'm not dressing it up — three engineered
   inputs (category, priority, recurrence) can only explain so much of a
   duration that has real noise baked into how it was generated.

5. **`ml/inference/resolution_time.py`** loads the model once; every
   response carries `is_estimate: true` and `low_confidence: true`
   (because `trained_on == "synthetic"`) — this flag is what stops a
   bootstrap number from ever being presented as a measured, real-world
   figure anywhere downstream. Unknown category/priority values and a
   missing/corrupt model file are both handled without raising (tested).

6. **`POST /api/ai/predict-resolution-time`** — authenticated, looks up
   the citizen's village server-side, computes a real recurrence count,
   returns the estimate. Wired into:
   - the citizen complaint form (`static/js/ai_resolution.js`) — a
     rough estimate shown once category+priority are set, explicitly
     worded "estimate, not a guaranteed deadline", no accept button
     (nothing to apply to the form);
   - the citizen tracking page (only for complaints still open — a
     resolved/rejected complaint doesn't get a stale prediction);
   - the officer/admin review page (same open-only rule), alongside the
     duplicate-detection panel from Phase 3.

7. **Tests — genuinely run, 23/23 passing**
   (`tests/test_temporal_features.py`, `tests/test_resolution_data_pipeline.py`,
   `tests/test_resolution_inference.py`, `tests/test_api_resolution_time.py`):
   leakage-prevention logic, synthetic data schema/ranges, inference
   correctness and graceful degradation (missing input, unknown category,
   missing model file), and the Flask API route (auth required, DB
   failure inside the route doesn't 500, malformed input handled) via a
   mocked DB — run with `pytest tests/ -v`.

**Still deferred:** escalation risk, sentiment, hotspot/anomaly/forecasting
analytics, Ollama, RAG, computer vision, voice, Kannada, GIS, Docker, and
the 18-chapter academic report.

## I. Phase 5 — Hotspot & anomaly detection (admin analytics dashboard)

**Scope cut made up front, stated plainly:** the master prompt's GIS/
heatmap vision (section 21) needs lat/lng coordinates or ward-boundary
polygons. This schema has neither — `users.village` is free text, and
complaints carry no coordinates at all. Geocoding villages would need an
external API and key; drawing ward boundaries would mean inventing
geography that isn't this project's data. Rather than fake a map with
placeholder coordinates, this phase delivers the real analytics —
hotspot ranking and anomaly detection — as tables on an admin dashboard,
grouped by the real `village` field. A true map is a documented follow-on
once real coordinates exist, not something built with fabricated ones.

1. **`ml/analytics/hotspot_anomaly.py`** — three real, unsupervised
   functions operating directly on live complaint records (category +
   village + created_at). No training/labels needed for this phase,
   unlike Phases 1-4:
   - `get_hotspots()` — ranks village+category pairs by volume in the
     last 30 days, with a real first-half-vs-second-half trend
     comparison (increasing/decreasing/stable), not a guess.
   - `detect_statistical_anomalies()` — flags a group's most recent week
     as an outlier against **its own** history, requiring at least 4
     prior weeks before it will flag anything at all (a group with 1-2
     weeks of data has no real baseline — skipped, not guessed at).
   - `detect_isolationforest_anomalies()` — fits `sklearn.IsolationForest`
     across every (village, category, week) count at once; doesn't need
     per-group history the way the statistical method does.

2. **A real bug was found and fixed by actually testing this, not just
   writing it and moving on.** I built a synthetic validation scenario
   with a known 85-complaint spike injected into one village/category,
   to check the detectors actually catch what they're supposed to. The
   injected spike happened to straddle two calendar week boundaries (56
   complaints landed in one pandas week, 30 in the next). The initial
   mean/std z-score implementation **missed the second week** — the
   first week's 56 dragged the baseline mean and standard deviation up
   enough that 30 looked unremarkable by comparison. Switched to
   median/MAD (the modified z-score, Iglewicz & Hoaglin threshold 3.5),
   which is robust to exactly this kind of contamination, and re-ran the
   same scenario — both weeks now correctly flagged (z-scores of 18+).
   This finding and the fix are encoded as a permanent regression test:
   `tests/test_hotspot_anomaly.py::test_statistical_anomaly_catches_spike_that_spans_two_calendar_weeks`.

3. **Where it's used**: a new admin-only page (`/admin/analytics`,
   linked from the admin dashboard) shows three tables — hotspots,
   statistical anomalies, isolation-forest anomalies — computed from
   whatever complaints currently exist in the real database (this
   project's fresh install will show mostly-empty tables until real
   complaint volume accumulates, which is expected and honestly labeled,
   not padded with fake rows). Two read-only JSON APIs
   (`/api/analytics/hotspots`, `/api/analytics/anomalies`), admin-only,
   back the same data for future dashboard/chart use.

4. **Tests — 8 new, 38/38 total passing**
   (`tests/test_hotspot_anomaly.py`, `tests/test_api_analytics.py`):
   hotspot ranking, the minimum-history guard, the flat-history
   divide-by-zero edge case, the spike-spanning-two-weeks regression test
   described above, IsolationForest's own too-little-data guard, and
   Flask route/API auth + empty-database handling.

**Still deferred:** escalation risk, sentiment, forecasting, Ollama, RAG,
computer vision, voice, Kannada, GIS with real coordinates, Docker, and
the 18-chapter academic report.

## J. Phase 6 — Ollama + RAG Assistant

**Two honest limitations stated up front, not discovered halfway through:**

1. This sandbox has **no reachable Ollama instance** — there's no route
   to Ollama's model registry to pull a model, and no Ollama binary
   installed. Every "Ollama unreachable" code path described below is
   verified against a REAL connection failure (confirmed via
   `requests.get(...)` actually refusing the connection), not simulated.
   Generation itself (an LLM actually producing an answer) has not run
   even once during this phase's development — it needs to be verified
   on a machine with Ollama installed (exact commands in section K).
2. No official Panchayat scheme/policy documents were provided. Rather
   than invent plausible-sounding government rules — which risks a
   citizen being told a fabricated policy as fact — the knowledge base
   (`rag/knowledge_base/`, 7 documents) describes **this portal's own
   actual, verified behavior**: complaint categories, status workflow,
   priority levels, submission steps, resolution-time estimates,
   duplicate-detection policy, and escalation guidance. All of it is
   checkable against the real application code. See
   `rag/knowledge_base/README.md` for the full reasoning.

### What was built

1. **`rag/generation/llm_service.py`** — Ollama HTTP client, configurable
   via `OLLAMA_BASE_URL`/`OLLAMA_MODEL` (already existed as placeholders
   in `config.py` since Phase 1). Handles connection refused, timeout,
   missing model (404), non-200 status, malformed JSON, and empty
   response — all as a single `OllamaUnavailableError` the pipeline
   catches, never a raw crash.

2. **Ingestion pipeline** (`rag/ingestion/chunker.py`,
   `rag/embeddings/embedder.py`, `rag/ingestion/ingest.py`): markdown
   documents are split by heading then chunked (60 words, 15 overlap —
   tuned down from a 120-word default after finding it diluted a
   six-item status list into one unfocused chunk), embedded, and stored
   in a FAISS `IndexFlatIP` index with a JSON metadata sidecar mapping
   vectors back to source document + heading + text.

3. **A real bug found by testing, not assumed away** — the first
   embedder used TF-IDF + TruncatedSVD (classic LSA). On this 14-chunk
   corpus, completely unrelated queries ("What is the capital of
   France?") scored HIGHER similarity than genuinely relevant ones,
   because SVD has nowhere near enough documents to learn real latent
   structure and ends up fitting noise. Switched to plain
   stopword-filtered, unigram TF-IDF — verified separation: on-topic
   0.21–0.53, off-topic 0.00–0.21 for the initial calibration set.
   Encoded as permanent regression tests in
   `tests/test_rag_embeddings.py` and `tests/test_rag_retrieval.py`.

4. **Retrieval threshold (0.20)** was calibrated the same empirical way
   as Phase 3's duplicate-detection threshold, not guessed.

5. **`rag/generation/rag_pipeline.py`** — orchestrates retrieve → prompt
   → generate with FOUR distinct, tested outcomes: index not
   built/empty KB (`available: false`), no relevant chunks found
   (`available: true`, explicit "couldn't find this" — no LLM call is
   even made), Ollama unreachable (`degraded: true`, returns the raw
   retrieved source text instead of a synthesized answer — retrieval
   still worked even though generation didn't), and success
   (`degraded: false`, generated answer + cited sources).

6. **Security design for personal data, stated plainly**: a citizen's
   own complaint status is **never** put into an LLM prompt or generated
   by the LLM (`rag/generation/complaint_lookup.py`). If a question
   mentions a complaint number, the route does a direct, authorization-
   checked database lookup and returns a deterministic, template-
   formatted answer — bypassing the LLM entirely for anything involving
   personal data, because access control belongs in code, not in a
   prompt a model could be tricked into ignoring. A citizen who is not
   the complaint's owner gets the exact same generic "I couldn't find
   that complaint under your account" message whether the complaint
   doesn't exist or just isn't theirs, so the response itself can't leak
   which complaint IDs belong to other citizens. Verified with a test
   that asserts the complaint's real category/status never appear in the
   response to an unauthorized requester
   (`tests/test_api_rag.py::test_rag_query_complaint_lookup_blocks_other_citizen`).

7. **New UI**: `/assistant` page for all three roles (linked from each
   dashboard), `POST /api/rag/query`. The frontend shows the degraded-
   mode notice honestly rather than hiding it.

### RAG evaluation — real numbers, including the unflattering one

`rag/evaluation/eval_dataset.json` (8 in-scope + 4 out-of-scope
questions) → `rag/evaluation/evaluate_rag.py` →
`rag/evaluation/results.json`:

- **Retrieval hit rate @ top-3: 100% (8/8)** — the correct source
  document appeared in the top 3 retrieved chunks for every in-scope
  question (though not always ranked #1 — see the known limitation
  below).
- **Correct-decline rate: 50% (2/4)** — only 2 of 4 off-topic questions
  were correctly rejected. Investigated, not just reported: "What is the
  current GST rate in India?" scored 0.206 similarity (just above the
  0.20 threshold) purely because it shares the single word "current"
  with "current status" phrasing used throughout the knowledge base —
  on a corpus this sparse, one incidental shared word is enough to
  produce a misleadingly nonzero score. This is a genuine limitation of
  TF-IDF similarity at small scale, not a bug to paper over; it would
  improve with a larger, more varied knowledge base or a real embedding
  model, neither available in this environment.
- **Generation/groundedness evaluation: not run** — `evaluate_rag.py`
  checks whether Ollama is reachable before attempting this and skips
  cleanly with a clear message when it isn't (as it correctly did here).
  The script is ready to run this half unmodified the moment Ollama is
  available — see section K for the command.

### Tests — 65 new, 103/103 total passing across all six phases

`tests/test_rag_chunker.py`, `tests/test_rag_embeddings.py` (includes
the SVD-regression test), `tests/test_rag_retrieval.py` (against the
real shipped index), `tests/test_rag_prompt_builder.py`,
`tests/test_llm_service.py` (mocked Ollama responses **plus one real
connection-refused test**), `tests/test_complaint_lookup.py` (the
authorization logic), `tests/test_rag_pipeline.py`, and
`tests/test_api_rag.py` (Flask integration, including the cross-user
authorization test described above).

**Still deferred:** escalation-risk prediction, sentiment analysis,
complaint forecasting, computer vision, voice input, Kannada support,
real-coordinate GIS, Docker, and the 18-chapter academic report.

## K. Ollama setup (to actually verify generation, which this sandbox could not do)

```bash
# 1. Install Ollama (see https://ollama.com for your OS)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model - must match OLLAMA_MODEL in your .env (default: llama3)
ollama pull llama3

# 3. Ollama serves on http://localhost:11434 by default - matches
#    OLLAMA_BASE_URL's default in .env.example. Change both together
#    if you run it elsewhere.

# 4. Build the knowledge base index (run once, or after editing
#    rag/knowledge_base/*.md)
python rag/ingestion/ingest.py

# 5. Run the RAG evaluation (retrieval metrics always run; generation
#    metrics run automatically too, now that Ollama is reachable)
python rag/evaluation/evaluate_rag.py

# 6. Run the full test suite
pytest tests/ -v

# 7. Start the app and open /assistant
python app.py
```

## L. Phase 7 — AI-based image complaint detection (computer vision)

**Honest scope statement up front, not discovered halfway through:** the
master prompt for this phase asked for photo-based detection of
potholes, garbage accumulation, damaged roads, broken street lights, and
drainage problems, using a pretrained model or transfer learning. Two
things in this sandbox made the full ask impossible to deliver
honestly, so this phase scopes down rather than fakes the rest:

1. **No pretrained CNN backbone was obtainable.** `torchvision`/`timm`
   fetch ImageNet weights from `download.pytorch.org` on first use; a
   direct connection test from this sandbox returns
   `x-deny-reason: host_not_allowed`. Same story for Hugging Face Hub
   (`huggingface.co`, also blocked). Transfer learning from a pretrained
   backbone — the master prompt's preferred approach — could not run
   here.
2. **No dataset was obtainable for four of the five requested classes.**
   Kaggle, Google Drive, and Hugging Face Hub (where the realistic
   candidate datasets for garbage/street-light/drainage all live) are
   all unreachable. The one dataset that WAS reachable — a small,
   real, web-scraped pothole-vs-plain-road image set hosted directly as
   files-in-git on GitHub (not LFS, not Drive) — only covers potholes.

Rather than invent detection for classes with no data behind them
(explicitly forbidden by the master prompt), this phase ships a real,
trained, evaluated pipeline for **one class — road potholes** — and
builds the entire surrounding system (upload, security, confirmation
workflow, API, error handling) to be the same regardless of how many
classes the model eventually supports. Adding a class later is a
retraining exercise (get a dataset, rerun `dataset_prep.py` +
`train_image_classifier.py`), not an architecture change, because both
the API and the UI read the supported-class list from one place
(`ml/inference/image_classify.SUPPORTED_CLASSES`) rather than
hardcoding it in multiple spots.

### What was built

1. **`ml/cv/preprocessing.py`** — shared, single source of truth for
   both training and inference: validates real decoded image format
   (not extension/Content-Type — this actually caught something, see
   below), strips EXIF/GPS metadata, resizes to 128×128, and extracts a
   HOG (edge/texture) + HSV color-histogram feature vector. Used
   identically by `train_image_classifier.py` and
   `ml/inference/image_classify.py`, so there is no train/serve skew.

2. **A real dataset, not synthetic, with real, documented problems
   found by actually running the validator against it**: the source
   repo's 714 images (`Somashekarbm/PotholeDetection` on GitHub) dedup
   to 706 unique files by content hash (8 exact duplicates dropped).
   Of those, **23 files have a `.jpg` extension but decode as WEBP** —
   an artifact of how they were scraped. Because the validator checks
   the actual decoded format rather than trusting the extension (which
   is exactly the "file type validation" the master prompt asks for on
   uploads), these 23 are correctly excluded from training rather than
   silently treated as valid JPEGs. Final counts used: 545 train / 68
   val / 70 test. Full provenance, license caveat (no LICENSE file in
   the source repo — the raw images are therefore NOT redistributed in
   this repo, only the code + manifest + metrics), and limitations are
   in `ml/cv/data/README.md`.

3. **`ml/cv/dataset_prep.py`** — builds a manifest (relative path,
   label, split, content hash) from the externally-cloned raw dataset;
   does its own stratified 80/10/10 split with a fixed seed (the source
   repo's own test split is only 18 images — too small to evaluate
   anything meaningfully).

4. **`ml/cv/train_image_classifier.py`** — trains and compares three
   real classifiers (Logistic Regression, calibrated Linear SVM, Random
   Forest) on the `train` split, selects by macro-F1 on `val`, reports
   FINAL numbers on the untouched `test` split exactly once.

   **Real held-out test results** (`ml/cv/evaluation/image_classifier_metrics.txt`):
   Random Forest selected (val macro-F1 0.941).

   | Metric (test set, n=70) | Value |
   |---|---|
   | Accuracy | 95.7% |
   | Macro F1 | 0.957 |
   | ROC-AUC (Pothole positive) | 0.994 |
   | Pothole precision / recall | 0.946 / 0.972 |
   | Plain precision / recall | 0.970 / 0.941 |

   **mAP is not reported and does not apply** — this is a whole-image
   binary classifier with no bounding boxes, not an object detector.
   Stated explicitly in both the metrics file and
   `ml/cv/data/README.md` rather than fabricating a number for a metric
   that doesn't fit the model type.

   These numbers are real but should be read with the same caution the
   rest of this project applies to its other high test scores: 706
   images from one source, web-scraped and "highly inconsistent" by the
   source author's own admission, is not a claim of production-grade
   robustness on arbitrary citizen photos.

5. **`ml/inference/image_classify.py`** — loads the model once;
   `analyze_image_bytes()` never raises for bad input (invalid image,
   empty file, missing model file all come back as a clearly-marked
   `available: false` result, never a crash or a fake prediction).
   Below `CV_CONFIDENCE_MIN` (default 0.65, configurable via `.env`), a
   "Pothole" prediction is surfaced as explicitly uncertain rather than
   turned into a category suggestion — the master prompt's
   "uncertainty/fallback status" requirement.

6. **Security for uploads** (`security_utils.generate_safe_stored_filename`,
   `config.py` `UPLOAD_FOLDER`/`MAX_IMAGE_UPLOAD_BYTES`):
   - Real format validation (decoded bytes, not extension/Content-Type).
   - Random server-generated filenames — the citizen's original filename
     is never used for on-disk storage or even retained.
   - EXIF/GPS metadata stripped before storage (a citizen's phone photo
     can carry their home's exact GPS coordinates in EXIF).
   - 5 MB size cap enforced twice: Flask's server-wide
     `MAX_CONTENT_LENGTH` AND an explicit per-file check in the route
     (`app.py`'s `_read_uploaded_image`), so the limit holds even if one
     of the two were ever changed independently.
   - Images stored **outside `static/`** specifically because Flask
     serves `static/` with no auth check at all — that would have
     defeated the "prevent unauthorized access to uploaded images"
     requirement outright. The only way to ever retrieve an image is
     `GET /complaint-image/<id>`.

7. **`GET /complaint-image/<id>`** — authorization: the complaint's
   owner, or any officer/admin, may view it; anyone else gets a plain
   404 — the exact same response a nonexistent image ID would give, so
   the response itself can't be used to probe which image IDs exist
   (the same anti-enumeration pattern Phase 6's complaint lookup
   already established for the same reason).

8. **`POST /api/ai/analyze-image`** — citizen-only, ephemeral preview
   (nothing is persisted by this call): validates + classifies an
   uploaded photo and returns the result. Exempted from the CSRF check
   the same way the existing JSON AI endpoints are (same-origin fetch,
   no server-state mutation) — see the updated comment on
   `enforce_csrf_on_forms` in `app.py`.

9. **Human-in-the-loop, integrated with the existing category/priority
   services, not bolted on separately**: `complaint.html` gets a photo
   field; `static/js/ai_image.js` calls the preview endpoint on file
   select and shows *"AI detected: Pothole (confidence X%) → suggested
   category: Road"* with a **"Use this category" button** — nothing is
   auto-filled. Accepting it sets the category `<select>` and fires a
   real `change` event, which the EXISTING `ai_priority.js` (Phase 2)
   already listens for — so an accepted image suggestion automatically
   feeds into the real priority-prediction call with zero changes
   needed to `ml/inference/priority.py`. On final form submission, the
   photo is re-validated and re-classified server-side (`app.py`'s
   `complaint()` view never trusts the earlier preview call alone,
   since a client could submit the form without ever calling the
   preview endpoint at all) and persisted to `complaint_images`, which
   records both the AI's suggestion and whether the citizen actually
   used it (`ai_suggestion_confirmed`) — an audit trail, not an
   auto-applied fact.

10. **Officer/citizen review UI**: `update_complaint.html` (officer/admin)
    and `track_complaint.html` (citizen) both display attached photos —
    served exclusively through the authenticated route above — alongside
    the AI's label, confidence, and whether the suggestion was confirmed,
    with an explicit reminder that the label is a suggestion to review,
    not a verified fact.

11. **Graceful degradation, tested, not just asserted in comments**: an
    invalid image, an oversized file, a missing model file, and a
    corrupt/missing file on disk at serve time all return clean 4xx/404
    responses — verified by the test suite below, none of them a 500.
    An invalid or oversized photo never blocks the complaint itself
    from being submitted — the photo is supplementary evidence, not a
    required field, a design choice stated explicitly in `app.py`.

### Database changes

`scripts/schema_patches.sql` (same cumulative, `IF NOT EXISTS`-guarded
file used since Phase 4) adds `complaint_images`: one row per uploaded
photo, storing the random stored filename, content type/size, the
model's real prediction (label/confidence/low-confidence flag/model
name), the category the AI suggested (if any), and whether the citizen
actually confirmed that suggestion. Foreign keys cascade on complaint
and user deletion, matching the existing schema's pattern.

### Tests — 39 new, 142/142 total passing across all seven phases

`tests/test_image_preprocessing.py` (12 — validation of valid/corrupt/
truncated/disguised-format/too-small images, EXIF handling, feature
determinism and shape), `tests/test_image_inference.py` (9 — real
shipped-model smoke test, confidence-threshold logic, invalid-input and
missing-model graceful degradation), `tests/test_api_analyze_image.py`
(6 — auth/role checks, missing file, non-image file, oversized file,
valid upload flow via Flask's test client), `tests/test_complaint_image_access.py`
(7 — owner/other-citizen/officer/admin authorization, nonexistent
image, file-missing-on-disk edge case), and
`tests/test_complaint_submission_with_image.py` (5 — full `/complaint`
POST flow with and without an image, invalid image not blocking
submission, CSRF/login enforcement) — run with `pytest tests/ -v`.

### How to reproduce training (dataset not shipped - see ml/cv/data/README.md)

```bash
git clone --depth 1 https://github.com/Somashekarbm/PotholeDetection.git /tmp/pothole_src
python ml/cv/dataset_prep.py --raw-dir "/tmp/pothole_src/My Dataset"
python ml/cv/train_image_classifier.py --raw-dir "/tmp/pothole_src/My Dataset"
# apply the new complaint_images table (safe to re-run, IF NOT EXISTS)
mysql -u root -p smart_panchayat < scripts/schema_patches.sql
pytest tests/ -v   # 142 tests as of Phase 7
python app.py
```

**Still deferred:** garbage/sanitation, broken-street-light, and
drainage image detection (no accessible dataset in this sandbox — see
`ml/cv/data/README.md`), a pretrained CNN/transfer-learning backbone (no
reachable weights host), voice input, Kannada support, real-coordinate
GIS, Docker, and the 18-chapter academic report.

## M. Phase 8 — Multilingual voice complaint system (Kannada + English)

**Honest scope statement up front:** English speech-to-text is real and
works fully offline. Kannada speech-to-text does **not** work in this
sandbox — verified, not assumed, exactly the same way Phase 6's Ollama
gap and Phase 7's four missing image classes were verified rather than
guessed at. Full investigation and reasoning:
`ml/asr/data/README.md`.

### What was actually tried, and what genuinely works

1. **PocketSphinx for English** — the `pocketsphinx` pip package ships
   a complete US English acoustic/language model + dictionary **inside
   the wheel itself**; nothing downloads at runtime. This is genuinely
   offline-capable and was tested end-to-end on real synthesized
   speech, producing real (imperfect — it's a ~20-year-old HMM/GMM
   engine, not neural) transcripts with a genuine confidence score
   derived from the decoder's own per-word posterior probabilities
   (not the raw utterance joint-probability, which reads as near-zero
   even for good transcriptions and would be misleading to surface).

2. **Whisper (via faster-whisper) for Kannada — a real integration,
   verified blocked, not just assumed blocked.** A direct attempt to
   load `WhisperModel("tiny")` in this sandbox produces:
   ```
   LocalEntryNotFoundError: ... HfHubHTTPError: 403 Forbidden ...
   Host not in allowlist: huggingface.co.
   ```
   `ml/asr/whisper_backend.py` catches this EXACT captured error and
   reports `available: False, reason: "model_download_blocked"`. In a
   deployment with network access to Hugging Face (or pre-downloaded
   weights via `WHISPER_MODEL_DIR`), this backend runs for real, for
   BOTH English and Kannada, with zero code changes — only
   `ml/asr/stt_service.py`'s existing fallback logic would start
   preferring it.

3. **Kannada translation — checked and confirmed unavailable, not
   skipped without looking.** Argos Translate's real package index
   (fetched from `raw.githubusercontent.com`, reachable here) lists
   ~50 supported languages; Kannada is not one of them, at all. No
   fabricated or English-passthrough "translation" was built — the
   translation function always honestly reports unavailable.

### What was built

- **`ml/asr/audio_preprocessing.py`** — real validation via `ffprobe`
  (decodes actual bytes, never trusts extension/Content-Type),
  duration bounds (0.6s–120s, configurable), RMS-based silence
  detection, canonical re-encoding to 16kHz mono PCM16 WAV for both
  inference and storage (never stores the client's original
  container/codec verbatim — same principle as Phase 7's image
  metadata-stripping re-encode).
- **A real, documented finding from testing, not hidden**: fed pure
  digital silence directly, PocketSphinx can hallucinate a short word
  (observed: "dog") rather than returning an empty hypothesis — a
  genuine language-model-prior artifact of statistical ASR. This is
  exactly why the RMS silence check runs *before* any backend is ever
  called in the real pipeline.
- **`ml/asr/pocketsphinx_backend.py`** / **`ml/asr/whisper_backend.py`**
  / **`ml/asr/stt_service.py`** — backend implementations plus the
  single orchestration entry point `app.py` calls; tries Whisper first
  for either language, falls back to PocketSphinx for English only
  (Kannada has no fallback, honestly).
- **Explicit language selection, not automatic detection** — a radio
  button in `complaint.html` ("English" / "Kannada (ಕನ್ನಡ)"). With only
  one language having a working backend right now, adding a language-ID
  model would add real complexity to solve a problem explicit selection
  already solves, and risks silently mis-routing Kannada speech into a
  confident-sounding-but-wrong English transcript.
- **Two-step preview/confirm flow, reusing Phase 2/3's existing
  pipelines with zero changes to them:** `POST /api/ai/transcribe-audio`
  (ephemeral citizen-only preview, nothing persisted) returns a
  transcript with an explicit **"Use this transcript" button**
  (`static/js/ai_voice.js`) that copies the text into the **existing**
  `#description` textarea and fires the same `input` event
  `ai_classify.js` (Phase 2) already listens for — so accepting a voice
  transcript automatically triggers the existing classifier, and after
  category selection, the existing priority predictor, and at officer
  review time the existing duplicate detector (Phase 3's
  `update_complaint()`, unchanged), because all three already key off
  `complaints.description` regardless of whether it was typed or
  transcribed. **Nothing auto-submits** — filling the textarea isn't
  submitting; the citizen reviews/edits and clicks "Submit Complaint"
  themselves.
- **Recording UI**: `static/js/ai_voice.js` uses `MediaRecorder` to
  capture audio directly in the browser (with a graceful "please upload
  a file instead" fallback if the browser doesn't support it), or
  accepts an uploaded audio file through the same input.
- **Server-side re-validation on real submission** (`app.py`'s
  `complaint()` view): the audio is re-validated and re-transcribed
  server-side — never trusting the earlier preview call alone (same
  rule as Phase 7's image path) — and persisted to a new
  `complaint_audio` table with BOTH the server's raw transcript and
  whatever ended up in the description field (the citizen-confirmed
  text), a human-in-the-loop audit trail.
- **`GET /complaint-audio/<id>`** — same authorization and
  anti-enumeration pattern as `/complaint-image/<id>`: complaint owner
  or officer/admin only, identical generic 404 for "doesn't exist" and
  "not yours".
- **Security**: real ffprobe-based format validation, random
  server-generated filenames, 10 MB size cap (enforced twice: Flask's
  `MAX_CONTENT_LENGTH` and an explicit per-file check), 120-second
  duration cap, storage outside `static/`.
- **Configurable retention**: `Config.AUDIO_RETENTION_DAYS` (default 90)
  + `scripts/purge_expired_audio.py` (with `--dry-run`), meant to run as
  a daily cron job. Only the original recording is subject to
  retention — the confirmed transcript is permanent complaint history
  in `complaints.description` regardless.
- **Officer/citizen review UI**: `update_complaint.html` (officer/admin)
  shows the recording, language, engine, confidence, raw transcript,
  and whether the citizen's confirmed text differed from the raw
  transcript; `track_complaint.html` (citizen) shows their own
  recording(s) and confirmed transcript.

### Database changes

`scripts/schema_patches.sql` adds `complaint_audio`: stored filename,
content type/size, duration, selected language, which engine actually
produced a transcript (or NULL if none was available), the raw
pre-edit transcript, the citizen-confirmed transcript, ASR confidence,
a low-confidence flag, and an availability flag — plus an index on
`uploaded_at` to make the retention purge query efficient.

### Tests — 40 new, 182/182 total passing across all eight phases

`tests/test_audio_preprocessing.py` (8 — real ffmpeg/ffprobe validation:
valid/silent/garbage/too-short/too-long audio, sample-rate transcoding,
configurable duration limits), `tests/test_stt_service.py` (12 — real
PocketSphinx transcription of synthesized speech, the documented
silence-hallucination finding, the real captured Whisper/Hugging-Face
failure, English fallback behavior, verified Kannada unavailability,
unsupported-language/invalid-audio/silence handling, the honest
translation stub), `tests/test_api_transcribe_audio.py` (8 — auth/role
checks, missing/invalid/oversized file, unsupported language, a real
end-to-end English transcription via Flask's test client),
`tests/test_complaint_audio_access.py` (7 — owner/other-citizen/
officer/admin authorization, nonexistent audio, file-missing-on-disk),
and `tests/test_complaint_submission_with_audio.py` (5 — full
`/complaint` POST flow with and without audio, invalid audio not
blocking submission, Kannada audio stored with transcription correctly
marked unavailable, CSRF/login enforcement). Tests that need
`espeak-ng` to synthesize test speech skip automatically if it isn't
installed (it is a test-fixture tool only, never part of the shipped
feature) — run with `pytest tests/ -v`.

### Exact setup commands

```bash
sudo apt-get install -y ffmpeg              # required at runtime (audio decode/transcode)
sudo apt-get install -y espeak-ng           # optional - test-fixture speech synthesis only
pip install pocketsphinx faster-whisper numpy
mysql -u root -p smart_panchayat < scripts/schema_patches.sql   # adds complaint_audio (safe to re-run)
pytest tests/ -v                             # 182 tests as of Phase 8
python app.py
```

See `ml/asr/data/README.md` for how to make Kannada (and better
English) actually work in a deployment with real network access —
no code changes needed, only `WHISPER_MODEL_DIR`/`WHISPER_MODEL_SIZE`
configuration once weights are reachable.

**Still deferred:** Kannada speech recognition (no reachable model
weights in this sandbox), Kannada↔English translation (no suitable
system exists at all, not just unreachable), automatic language
detection, garbage/street-light/drainage image detection (Phase 7),
a pretrained CNN backbone for images, real-coordinate GIS, Docker, and
the 18-chapter academic report.

