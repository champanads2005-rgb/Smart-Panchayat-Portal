-- Schema patch discovered during the Phase 4 audit.
--
-- app.py has always read from and written to a `complaint_updates` table
-- (INSERT on complaint creation and on every status change, SELECT for
-- the citizen tracking page), but complaint.sql never defined it. On a
-- fresh database created only from complaint.sql, every complaint
-- submission and status update would fail with "table doesn't exist".
--
-- This table is also the ONLY reliable source of a resolution timestamp:
-- complaints.updated_at bumps on ANY change (priority edits, remarks,
-- etc.), not just the transition to "Resolved", so it cannot be used to
-- measure resolution time. The row in complaint_updates where
-- status = 'Resolved' can.
--
-- Safe to run on an existing database - IF NOT EXISTS guards it.

USE smart_panchayat;

CREATE TABLE IF NOT EXISTS complaint_updates (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    status ENUM(
        'Submitted',
        'Under Review',
        'Assigned',
        'In Progress',
        'Resolved',
        'Rejected'
    ) NOT NULL,

    message TEXT,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE
);

-- Helpful for the resolution-time feature extraction query in
-- ml/data/extract_resolution_history.py (filtering by complaint_id and
-- status, ordering by updated_at).
CREATE INDEX IF NOT EXISTS idx_complaint_updates_lookup
    ON complaint_updates (complaint_id, status, updated_at);

-- ---------------------------------------------------------------------
-- Phase 7 - AI-based image complaint detection
-- ---------------------------------------------------------------------
-- One row per uploaded complaint image. `stored_filename` is a random,
-- server-generated name (see security_utils.generate_safe_stored_filename)
-- - the citizen's original filename is never used for on-disk storage
-- and is not even retained, to avoid leaking anything through it.
-- Images live outside static/ (see config.py UPLOAD_FOLDER) and are only
-- ever served through the authenticated /complaint-image/<id> route in
-- app.py, never as a static file.
CREATE TABLE IF NOT EXISTS complaint_images (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    stored_filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(50) NOT NULL,

    file_size INT NOT NULL,

    -- What the trained image model actually predicted for this photo
    -- (NULL if the model was unavailable at upload time - see
    -- ml/inference/image_classify.py). detected_label is one of the
    -- model's real training classes ("Pothole"/"Plain"), never a
    -- fabricated category.
    detected_label VARCHAR(50),

    confidence FLOAT,

    is_low_confidence BOOLEAN DEFAULT TRUE,

    model_name VARCHAR(100),

    -- The complaint category the AI suggested FROM this image (NULL if
    -- no confident suggestion was made). This is only ever a suggestion
    -- - the actual complaints.category value is whatever the citizen
    -- picked, confirmed or not.
    ai_suggested_category ENUM(
        'Water',
        'Electricity',
        'Road',
        'Sanitation',
        'Street Light',
        'Drainage',
        'Other'
    ),

    -- Whether the citizen actually clicked "Use this" to apply the AI
    -- suggestion to the complaint's category, vs. picking manually
    -- despite seeing (or not seeing) a suggestion. Human-in-the-loop
    -- audit trail, matching the existing text-classifier/priority UI
    -- pattern already used elsewhere in this project.
    ai_suggestion_confirmed BOOLEAN DEFAULT FALSE,

    uploaded_by INT NOT NULL,

    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE,

    FOREIGN KEY (uploaded_by)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_complaint_images_complaint
    ON complaint_images (complaint_id);

-- ---------------------------------------------------------------------
-- Phase 8 - multilingual voice complaint system
-- ---------------------------------------------------------------------
-- One row per uploaded voice complaint recording. Mirrors
-- complaint_images in spirit: random server-generated stored_filename,
-- served only through the authenticated /complaint-audio/<id> route in
-- app.py, never as a static file. The ORIGINAL recording is subject to
-- config.py's AUDIO_RETENTION_DAYS (see scripts/purge_expired_audio.py)
-- - the transcript text is NOT deleted on purge, since it already lives
-- permanently and independently in complaints.description.
CREATE TABLE IF NOT EXISTS complaint_audio (
    id INT AUTO_INCREMENT PRIMARY KEY,

    complaint_id INT NOT NULL,

    stored_filename VARCHAR(255) NOT NULL,

    content_type VARCHAR(50) NOT NULL,

    file_size INT NOT NULL,

    duration_seconds FLOAT,

    -- Language the citizen explicitly selected before recording -
    -- Phase 8 uses explicit selection, not automatic language ID (see
    -- ml/asr/data/README.md for why).
    language_selected ENUM('en', 'kn') NOT NULL,

    -- Which speech engine actually produced the transcript
    -- ('pocketsphinx_en_us', 'whisper-tiny', etc.), or NULL if none was
    -- available (see transcription_available below) - never fabricated.
    transcription_engine VARCHAR(50),

    -- The engine's own raw output, BEFORE the citizen edited anything.
    raw_transcript TEXT,

    -- What the citizen actually confirmed/edited before submitting -
    -- this is what was copied into complaints.description and fed to
    -- the classifier/priority/duplicate-detection pipelines. Kept
    -- alongside raw_transcript as a human-in-the-loop audit trail
    -- (requirement: do not submit automatically without confirmation).
    confirmed_transcript TEXT,

    asr_confidence FLOAT,

    is_low_confidence BOOLEAN DEFAULT TRUE,

    -- FALSE when the language selected had no available speech engine
    -- at all (Kannada, in this environment - see
    -- ml/asr/whisper_backend.py) or the audio failed validation. The
    -- audio can still be attached/stored even when this is FALSE - a
    -- citizen may want the original recording kept even if no
    -- transcript could be produced automatically.
    transcription_available BOOLEAN DEFAULT FALSE,

    uploaded_by INT NOT NULL,

    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (complaint_id)
        REFERENCES complaints(id)
        ON DELETE CASCADE,

    FOREIGN KEY (uploaded_by)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_complaint_audio_complaint
    ON complaint_audio (complaint_id);

CREATE INDEX IF NOT EXISTS idx_complaint_audio_retention
    ON complaint_audio (uploaded_at);
