import os

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort, send_file
import mysql.connector

from config import Config
from security_utils import (
    generate_safe_stored_filename,
    get_csrf_token,
    hash_password,
    looks_like_hash,
    validate_csrf,
    verify_password,
)
from ml.inference.classify import classify_complaint
from ml.inference.priority import predict_priority
from ml.inference.duplicate import find_similar_complaints
from ml.inference.resolution_time import predict_resolution_time
from ml.inference.image_classify import (
    analyze_image_bytes,
    SUPPORTED_CLASSES as IMAGE_SUPPORTED_CLASSES,
)
from ml.cv.preprocessing import InvalidImageError
from ml.asr.stt_service import (
    transcribe_audio,
    SUPPORTED_LANGUAGES as AUDIO_SUPPORTED_LANGUAGES,
    LANGUAGE_LABELS as AUDIO_LANGUAGE_LABELS,
)
from ml.analytics.hotspot_anomaly import (
    get_hotspots,
    detect_statistical_anomalies,
    detect_isolationforest_anomalies,
)
from rag.generation.rag_pipeline import answer_question as rag_answer_question
from rag.generation.complaint_lookup import (
    extract_complaint_id,
    is_authorized as rag_is_authorized,
    format_complaint_answer,
    not_found_message as rag_not_found_message,
)

app = Flask(__name__)

app.secret_key = Config.SECRET_KEY

# Session cookie hardening
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)

# Server-level cap on ANY request body (Flask rejects oversized uploads
# before they're even fully read into memory), in addition to the
# explicit per-file checks in the image/audio upload routes below. Sized
# to the larger of the two upload types (Phase 8's audio cap) plus
# headroom for other form fields.
app.config["MAX_CONTENT_LENGTH"] = (
    max(Config.MAX_IMAGE_UPLOAD_BYTES, Config.MAX_AUDIO_UPLOAD_BYTES) + (256 * 1024)
)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.AUDIO_UPLOAD_FOLDER, exist_ok=True)


@app.context_processor
def inject_csrf_token():
    """Makes {{ csrf_token }} available in every template automatically."""
    return {"csrf_token": get_csrf_token()}


@app.before_request
def enforce_csrf_on_forms():
    """Validates the CSRF token on every state-changing POST request that
    submits a normal HTML form. JSON API requests (used by the AI widget)
    are same-origin fetches and are left to the browser's own SOP; they do
    not mutate server state. /api/ai/analyze-image and
    /api/ai/transcribe-audio are multipart (not JSON) same-origin
    fetches for the same reason: both only run inference ephemerally and
    write nothing to the database, so they are exempted the same way the
    JSON AI endpoints are."""
    if request.path in ("/api/ai/analyze-image", "/api/ai/transcribe-audio"):
        return
    if request.method == "POST" and request.content_type and "application/json" not in request.content_type:
        submitted_token = request.form.get("csrf_token")
        validate_csrf(submitted_token)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_db_connection():
    return mysql.connector.connect(
        host=Config.DB_HOST,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=Config.DB_NAME
    )


def get_user_panchayat_id(user_id):
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT panchayat_id FROM users WHERE id = %s", (user_id,))
    row = cursor.fetchone()
    cursor.close()
    db.close()
    return row["panchayat_id"] if row else None


def get_all_panchayats():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, name, district, state
        FROM panchayats
        ORDER BY name ASC
    """)
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return rows


def complaint_belongs_to_user_panchayat(complaint_id, panchayat_id):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute(
        "SELECT 1 FROM complaints WHERE id = %s AND panchayat_id = %s",
        (complaint_id, panchayat_id),
    )
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result is not None


# =========================================================
# AI: COMPLAINT CLASSIFICATION API
# =========================================================

@app.route("/api/ai/classify", methods=["POST"])
def api_classify_complaint():
    if "user_id" not in session:
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    result = classify_complaint(text)
    return jsonify(result)


# =========================================================
# AI: PRIORITY PREDICTION API
# =========================================================

def get_recent_similar_count(category, village, days=30):
    """REAL feature computation (not synthetic): counts how many complaints
    in the same category and village were submitted in the last `days`
    days. Used as the recurrence_count input to the priority model."""
    if not category or not village:
        return 0

    db = get_db_connection()
    cursor = db.cursor()
    query = """
        SELECT COUNT(*) FROM complaints
        JOIN users ON complaints.user_id = users.id
        WHERE complaints.category = %s
          AND users.village = %s
          AND complaints.created_at >= NOW() - INTERVAL %s DAY
    """
    cursor.execute(query, (category, village, days))
    count = cursor.fetchone()[0]
    cursor.close()
    db.close()
    return count


@app.route("/api/ai/predict-priority", methods=["POST"])
def api_predict_priority():
    if "user_id" not in session:
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")
    category = payload.get("category", "")

    # Look up the logged-in citizen's village server-side (never trust a
    # client-supplied location for this) to compute a real recurrence count.
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT village FROM users WHERE id = %s", (session["user_id"],))
    user_row = cursor.fetchone()
    cursor.close()
    db.close()
    village = user_row["village"] if user_row else None

    try:
        recurrence_count = get_recent_similar_count(category, village)
    except Exception:  # noqa: BLE001 - degrade gracefully, don't break the form
        recurrence_count = 0

    result = predict_priority(text, category, recurrence_count)
    return jsonify(result)


# =========================================================
# AI: RESOLUTION-TIME PREDICTION API
# =========================================================

@app.route("/api/ai/predict-resolution-time", methods=["POST"])
def api_predict_resolution_time():
    if "user_id" not in session:
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    category = payload.get("category", "")
    priority = payload.get("priority", "")

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT village FROM users WHERE id = %s", (session["user_id"],))
    user_row = cursor.fetchone()
    cursor.close()
    db.close()
    village = user_row["village"] if user_row else None

    try:
        recurrence_count = get_recent_similar_count(category, village)
    except Exception:  # noqa: BLE001
        recurrence_count = 0

    result = predict_resolution_time(category, priority, recurrence_count)
    return jsonify(result)


# =========================================================
# AI: IMAGE-BASED COMPLAINT DETECTION (Phase 7)
# =========================================================
#
# Two endpoints:
#   POST /api/ai/analyze-image   - ephemeral preview. Citizen uploads a
#       photo WHILE filling the form; it is validated and classified but
#       NOT persisted here. Mirrors the existing as-you-type suggestion
#       pattern (classify/priority above) - shows a suggestion, requires
#       an explicit "Use this" click, never auto-applies anything.
#   POST /complaint (see below) - the SAME file, re-submitted by the
#       browser as part of the real form post, is validated again
#       (never trust the earlier preview call alone) and this time
#       actually stored, tied to the newly created complaint.
#
# GET /complaint-image/<id> - the only way an image is ever served back;
# static/ is never used for these, so there is no unauthenticated path
# to a citizen's uploaded photo (requirement: prevent unauthorized
# access to uploaded complaint images).

MAX_IMAGE_UPLOAD_BYTES = Config.MAX_IMAGE_UPLOAD_BYTES


def _read_uploaded_image(file_storage):
    """Reads an uploaded file from a Flask request with an explicit size
    cap enforced in application code (in addition to Flask's server-wide
    MAX_CONTENT_LENGTH), so a per-endpoint limit is guaranteed even if
    that global setting is ever changed independently. Returns raw bytes
    or raises ValueError("file_too_large")."""
    raw_bytes = file_storage.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError("file_too_large")
    return raw_bytes


@app.route("/api/ai/analyze-image", methods=["POST"])
def api_analyze_image():
    if "user_id" not in session:
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    if session.get("user_role") != "citizen":
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"available": False, "reason": "no_file_provided"}), 400

    try:
        raw_bytes = _read_uploaded_image(request.files["image"])
    except ValueError:
        return jsonify({
            "available": False,
            "reason": "file_too_large",
            "max_bytes": MAX_IMAGE_UPLOAD_BYTES,
        }), 413

    result = analyze_image_bytes(raw_bytes)
    # The re-encoded image bytes are only needed when actually persisting
    # (in /complaint below) - never send raw image bytes back in a JSON
    # preview response.
    result.pop("reencoded_image_bytes", None)
    return jsonify(result)


@app.route("/complaint-image/<int:image_id>")
def complaint_image(image_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT complaint_images.id, complaint_images.stored_filename,
               complaint_images.content_type, complaints.user_id AS owner_id
        FROM complaint_images
        JOIN complaints ON complaint_images.complaint_id = complaints.id
        WHERE complaint_images.id = %s
        """,
        (image_id,),
    )
    image_row = cursor.fetchone()
    cursor.close()
    db.close()

    # Same generic 404 whether the image doesn't exist or the requester
    # just isn't allowed to see it - this is the same
    # doesn't-leak-existence pattern already used by the RAG complaint
    # lookup (rag/generation/complaint_lookup.py) for the same reason:
    # the response itself must not reveal which complaint IDs belong to
    # other citizens.
    is_owner = image_row and image_row["owner_id"] == session["user_id"]
    is_staff = session.get("user_role") in ("officer", "admin")
    if not image_row or not (is_owner or is_staff):
        abort(404)

    file_path = os.path.join(Config.UPLOAD_FOLDER, image_row["stored_filename"])
    if not os.path.isfile(file_path):
        abort(404)

    return send_file(file_path, mimetype=image_row["content_type"])


# =========================================================
# AI: MULTILINGUAL VOICE COMPLAINT SYSTEM (Phase 8)
# =========================================================
#
# Same two-step pattern as Phase 7's image flow:
#   POST /api/ai/transcribe-audio - ephemeral preview. Citizen records
#       or uploads audio WHILE filling the form; validated and
#       transcribed but NOT persisted here. The transcript is shown for
#       correction and, once accepted, is copied into the EXISTING
#       #description textarea - which means the existing
#       classify/priority AI panels (ai_classify.js / ai_priority.js,
#       Phases 2) pick it up completely unchanged, and so does officer-
#       side duplicate detection (update_complaint(), Phase 3), since
#       all three already key off complaints.description regardless of
#       whether it was typed or transcribed.
#   POST /complaint (see complaint() above) - re-validates and
#       re-transcribes the SAME audio server-side (never trusts the
#       earlier preview call alone) and persists it, tied to the new
#       complaint, together with BOTH the raw and citizen-confirmed
#       transcript text for audit.
#
# GET /complaint-audio/<id> - the only way a recording is ever served
# back; mirrors /complaint-image/<id>'s authorization exactly.

MAX_AUDIO_UPLOAD_BYTES = Config.MAX_AUDIO_UPLOAD_BYTES


def _read_uploaded_audio(file_storage):
    raw_bytes = file_storage.read(MAX_AUDIO_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError("file_too_large")
    return raw_bytes


@app.route("/api/ai/transcribe-audio", methods=["POST"])
def api_transcribe_audio():
    if "user_id" not in session or session.get("user_role") != "citizen":
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    language = request.form.get("language", "")
    if language not in AUDIO_SUPPORTED_LANGUAGES:
        return jsonify({
            "available": False,
            "reason": "unsupported_language",
            "supported_languages": AUDIO_SUPPORTED_LANGUAGES,
        }), 400

    if "audio" not in request.files or request.files["audio"].filename == "":
        return jsonify({"available": False, "reason": "no_file_provided"}), 400

    try:
        raw_bytes = _read_uploaded_audio(request.files["audio"])
    except ValueError:
        return jsonify({
            "available": False,
            "reason": "file_too_large",
            "max_bytes": MAX_AUDIO_UPLOAD_BYTES,
        }), 413

    result = transcribe_audio(
        raw_bytes,
        language,
        max_duration_seconds=Config.MAX_AUDIO_DURATION_SECONDS,
        whisper_model_size=Config.WHISPER_MODEL_SIZE,
        whisper_model_dir=Config.WHISPER_MODEL_DIR,
    )
    # Never send raw audio bytes back in a JSON preview response - only
    # needed when actually persisting (in /complaint).
    result.pop("wav_bytes", None)
    return jsonify(result)


@app.route("/complaint-audio/<int:audio_id>")
def complaint_audio(audio_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT complaint_audio.id, complaint_audio.stored_filename,
               complaint_audio.content_type, complaints.user_id AS owner_id
        FROM complaint_audio
        JOIN complaints ON complaint_audio.complaint_id = complaints.id
        WHERE complaint_audio.id = %s
        """,
        (audio_id,),
    )
    audio_row = cursor.fetchone()
    cursor.close()
    db.close()

    # Same generic-404, anti-enumeration pattern as /complaint-image/<id>
    # and the RAG complaint lookup - see those for the rationale.
    is_owner = audio_row and audio_row["owner_id"] == session["user_id"]
    is_staff = session.get("user_role") in ("officer", "admin")
    if not audio_row or not (is_owner or is_staff):
        abort(404)

    file_path = os.path.join(Config.AUDIO_UPLOAD_FOLDER, audio_row["stored_filename"])
    if not os.path.isfile(file_path):
        abort(404)

    return send_file(file_path, mimetype=audio_row["content_type"])


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# CITIZEN DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") == "officer":
        return redirect(url_for("officer_dashboard"))

    if session.get("user_role") == "admin":
        return redirect(url_for("admin_dashboard"))

    user_id = session["user_id"]

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM complaints
        WHERE user_id = %s
    """, (user_id,))

    total_complaints = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT COUNT(*) AS submitted
        FROM complaints
        WHERE user_id = %s
        AND status = 'Submitted'
    """, (user_id,))

    submitted_complaints = cursor.fetchone()["submitted"]

    cursor.execute("""
        SELECT COUNT(*) AS in_progress
        FROM complaints
        WHERE user_id = %s
        AND status = 'In Progress'
    """, (user_id,))

    in_progress_complaints = cursor.fetchone()["in_progress"]

    cursor.execute("""
        SELECT COUNT(*) AS resolved
        FROM complaints
        WHERE user_id = %s
        AND status = 'Resolved'
    """, (user_id,))

    resolved_complaints = cursor.fetchone()["resolved"]

    cursor.close()
    db.close()

    return render_template(
        "dashboard.html",
        user_name=session["user_name"],
        total_complaints=total_complaints,
        submitted_complaints=submitted_complaints,
        in_progress_complaints=in_progress_complaints,
        resolved_complaints=resolved_complaints
    )


# =========================================================
# REGISTER
# =========================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    panchayats = get_all_panchayats()

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        village = request.form.get("village", "").strip()
        panchayat_id = request.form.get("panchayat_id", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not phone or not village or not panchayat_id:
            return "Please fill all required fields."

        if password != confirm_password:
            return "Passwords do not match!"

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT id FROM panchayats WHERE id = %s", (panchayat_id,))
        if not cursor.fetchone():
            cursor.close(); db.close()
            return "Invalid Panchayat selected!"

        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close(); db.close()
            return "An account with this email already exists!"

        cursor.execute("""
            INSERT INTO users (name, email, phone, village, panchayat_id, password, role)
            VALUES (%s, %s, %s, %s, %s, %s, 'citizen')
        """, (name, email, phone, village, panchayat_id, hash_password(password)))
        db.commit()
        cursor.close(); db.close()
        return redirect(url_for("login"))

    return render_template("register.html", panchayats=panchayats)


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT id, name, email, password, role, panchayat_id
            FROM users
            WHERE email = %s
        """, (email,))
        user = cursor.fetchone()
        cursor.close(); db.close()

        password_ok = False
        if user:
            if looks_like_hash(user["password"]):
                password_ok = verify_password(password, user["password"])
            else:
                password_ok = user["password"] == password
                if password_ok:
                    upgrade_db = get_db_connection()
                    upgrade_cursor = upgrade_db.cursor()
                    upgrade_cursor.execute(
                        "UPDATE users SET password = %s WHERE id = %s",
                        (hash_password(password), user["id"]),
                    )
                    upgrade_db.commit()
                    upgrade_cursor.close(); upgrade_db.close()

        if not password_ok:
            return "Invalid email or password!"

        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        session["user_role"] = user["role"]
        session["panchayat_id"] = user["panchayat_id"]

        if user["role"] == "officer":
            return redirect(url_for("officer_dashboard"))
        if user["role"] == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("dashboard"))

    return render_template("login.html")


# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("login"))


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

@app.route("/complaint", methods=["GET", "POST"])
def complaint():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "citizen":
        return redirect(url_for("dashboard"))

    user_id = session["user_id"]

    if request.method == "POST":

        title = request.form["title"]
        category = request.form["category"]
        description = request.form["description"]
        priority = request.form["priority"]

        # Phase 7: optional image upload. Validated and (if a model is
        # available) classified again here - the earlier
        # /api/ai/analyze-image preview call is never trusted as the
        # sole check, since a client could submit the form without ever
        # calling that endpoint at all.
        image_file = request.files.get("image")
        image_analysis = None
        image_bytes_to_store = None
        image_error = None
        if image_file and image_file.filename:
            try:
                raw_bytes = _read_uploaded_image(image_file)
                image_analysis = analyze_image_bytes(raw_bytes)
                if not image_analysis["available"]:
                    image_error = image_analysis.get("reason")
                else:
                    image_bytes_to_store = image_analysis.pop("reencoded_image_bytes")
            except ValueError:
                image_error = "file_too_large"
        # Design choice, stated plainly: a bad/invalid photo (image_error
        # set, image_bytes_to_store left None) never blocks submission of
        # the complaint itself - the photo is supplementary evidence, not
        # a required field. The citizen already saw the same validation
        # result from the /api/ai/analyze-image preview call before
        # reaching this point (static/js/ai_image.js), so this is a
        # silent no-op here rather than a duplicate error message.

        # Phase 8: optional voice recording. Re-validated and
        # re-transcribed here too - same "never trust the earlier
        # preview call alone" rule as the image path above. The citizen
        # may have edited the transcript in the #description textarea
        # after the preview ran (that IS the point - requirement: show
        # the transcript for correction before submission), so
        # `description` above is the CONFIRMED text, while
        # `raw_transcript` below is this server-side re-transcription,
        # kept side by side purely as an audit trail.
        audio_file = request.files.get("audio")
        audio_language = request.form.get("audio_language", "")
        audio_result = None
        audio_bytes_to_store = None
        if audio_file and audio_file.filename and audio_language in AUDIO_SUPPORTED_LANGUAGES:
            try:
                raw_audio_bytes = _read_uploaded_audio(audio_file)
                audio_result = transcribe_audio(
                    raw_audio_bytes,
                    audio_language,
                    max_duration_seconds=Config.MAX_AUDIO_DURATION_SECONDS,
                    whisper_model_size=Config.WHISPER_MODEL_SIZE,
                    whisper_model_dir=Config.WHISPER_MODEL_DIR,
                )
                # Even if transcription itself failed/was unavailable,
                # audio_preprocessing may still have handed back a valid
                # re-encoded WAV (e.g. the "silent audio" case) - a
                # citizen may still want the original recording attached
                # even with no transcript.
                audio_bytes_to_store = audio_result.pop("wav_bytes", None)
            except ValueError:
                audio_result = {"available": False, "reason": "file_too_large"}
        # Same design choice as the photo path: a bad/invalid/untranscribable
        # recording never blocks submitting the complaint text itself.

        # Human-in-the-loop confirmation (requirement: require citizen or
        # officer confirmation before finalizing an AI-generated category
        # or priority): the citizen's own choice in the category/priority
        # dropdowns is what gets stored either way. We only additionally
        # record WHETHER that choice matches what the AI suggested from
        # the image, for audit purposes - never silently override it.
        ai_suggestion_confirmed = bool(
            image_analysis
            and image_analysis.get("suggested_category")
            and image_analysis["suggested_category"] == category
        )

        db = get_db_connection()
        cursor = db.cursor()

        # Insert complaint
        query = """
        INSERT INTO complaints
        (
            user_id,
            panchayat_id,
            category,
            title,
            description,
            priority,
            status
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            query,
            (
                user_id,
                session.get("panchayat_id"),
                category,
                title,
                description,
                priority,
                "Submitted"
            )
        )

        # Get newly created complaint ID
        complaint_id = cursor.lastrowid

        # Create first tracking update
        update_query = """
        INSERT INTO complaint_updates
        (
            complaint_id,
            status,
            message
        )
        VALUES (%s, %s, %s)
        """

        cursor.execute(
            update_query,
            (
                complaint_id,
                "Submitted",
                "Complaint submitted successfully."
            )
        )

        if image_bytes_to_store is not None:
            stored_filename = generate_safe_stored_filename("jpg")
            file_path = os.path.join(Config.UPLOAD_FOLDER, stored_filename)
            with open(file_path, "wb") as f:
                f.write(image_bytes_to_store)

            cursor.execute(
                """
                INSERT INTO complaint_images
                (complaint_id, stored_filename, content_type, file_size,
                 detected_label, confidence, is_low_confidence, model_name,
                 ai_suggested_category, ai_suggestion_confirmed, uploaded_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    complaint_id,
                    stored_filename,
                    "image/jpeg",
                    len(image_bytes_to_store),
                    image_analysis.get("detected_label"),
                    image_analysis.get("confidence"),
                    image_analysis.get("is_low_confidence"),
                    image_analysis.get("model_name"),
                    image_analysis.get("suggested_category"),
                    ai_suggestion_confirmed,
                    user_id,
                ),
            )

        if audio_bytes_to_store is not None:
            audio_stored_filename = generate_safe_stored_filename("wav")
            audio_file_path = os.path.join(Config.AUDIO_UPLOAD_FOLDER, audio_stored_filename)
            with open(audio_file_path, "wb") as f:
                f.write(audio_bytes_to_store)

            cursor.execute(
                """
                INSERT INTO complaint_audio
                (complaint_id, stored_filename, content_type, file_size,
                 duration_seconds, language_selected, transcription_engine,
                 raw_transcript, confirmed_transcript, asr_confidence,
                 is_low_confidence, transcription_available, uploaded_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    complaint_id,
                    audio_stored_filename,
                    "audio/wav",
                    len(audio_bytes_to_store),
                    audio_result.get("duration_seconds") if audio_result else None,
                    audio_language,
                    audio_result.get("engine") if audio_result else None,
                    audio_result.get("transcript") if audio_result else None,
                    # The confirmed transcript is whatever ended up in the
                    # description field - only worth recording as a voice
                    # confirmation if it actually came from a real
                    # transcript (i.e. transcription succeeded).
                    description if (audio_result and audio_result.get("available")) else None,
                    audio_result.get("confidence") if audio_result else None,
                    audio_result.get("is_low_confidence", True) if audio_result else True,
                    bool(audio_result and audio_result.get("available")),
                    user_id,
                ),
            )

        db.commit()

        cursor.close()
        db.close()

        return redirect(url_for("my_complaints"))

    return render_template(
        "complaint.html",
        image_supported_classes=IMAGE_SUPPORTED_CLASSES,
        audio_supported_languages=AUDIO_SUPPORTED_LANGUAGES,
        audio_language_labels=AUDIO_LANGUAGE_LABELS,
    )


# =========================================================
# MY COMPLAINTS
# =========================================================

@app.route("/my-complaints")
def my_complaints():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "citizen":
        return redirect(url_for("dashboard"))

    user_id = session["user_id"]

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT
        id,
        category,
        title,
        description,
        priority,
        status,
        created_at
    FROM complaints
    WHERE user_id = %s
    ORDER BY created_at DESC
    """

    cursor.execute(query, (user_id,))

    complaints = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "my_complaints.html",
        complaints=complaints
    )


# =========================================================
# TRACK COMPLAINT
# =========================================================

@app.route("/track-complaint/<int:complaint_id>")
def track_complaint(complaint_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "citizen":
        return redirect(url_for("dashboard"))

    user_id = session["user_id"]

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    # Get complaint details
    cursor.execute("""
        SELECT
            complaints.id,
            complaints.category,
            complaints.title,
            complaints.description,
            complaints.priority,
            complaints.status,
            complaints.created_at,
            users.village
        FROM complaints
        JOIN users ON complaints.user_id = users.id
        WHERE complaints.id = %s
        AND complaints.user_id = %s
    """, (complaint_id, user_id))

    complaint_data = cursor.fetchone()

    if complaint_data is None:

        cursor.close()
        db.close()

        return "Complaint not found!"

    # Get tracking history
    cursor.execute("""
        SELECT
            status,
            message,
            updated_at
        FROM complaint_updates
        WHERE complaint_id = %s
        ORDER BY updated_at ASC
    """, (complaint_id,))

    updates = cursor.fetchall()

    cursor.close()
    db.close()

    db = get_db_connection()
    img_cursor = db.cursor(dictionary=True)
    img_cursor.execute(
        """
        SELECT id, detected_label, confidence, is_low_confidence,
               ai_suggested_category
        FROM complaint_images
        WHERE complaint_id = %s
        ORDER BY uploaded_at DESC
        """,
        (complaint_id,),
    )
    complaint_images = img_cursor.fetchall()
    img_cursor.close()
    db.close()

    db = get_db_connection()
    audio_cursor = db.cursor(dictionary=True)
    audio_cursor.execute(
        """
        SELECT id, duration_seconds, language_selected, transcription_engine,
               confirmed_transcript, asr_confidence, is_low_confidence,
               transcription_available
        FROM complaint_audio
        WHERE complaint_id = %s
        ORDER BY uploaded_at DESC
        """,
        (complaint_id,),
    )
    complaint_audio_rows = audio_cursor.fetchall()
    audio_cursor.close()
    db.close()

    # AI: estimated resolution time - only meaningful for complaints still
    # open (predicting a resolution time for an already-Resolved/Rejected
    # complaint would be a stale, misleading number).
    resolution_estimate = None
    if complaint_data["status"] not in ("Resolved", "Rejected"):
        try:
            recurrence_count = get_recent_similar_count(
                complaint_data["category"], complaint_data["village"]
            )
        except Exception:  # noqa: BLE001
            recurrence_count = 0
        resolution_estimate = predict_resolution_time(
            complaint_data["category"], complaint_data["priority"], recurrence_count
        )

    return render_template(
        "track_complaint.html",
        complaint=complaint_data,
        updates=updates,
        resolution_estimate=resolution_estimate,
        complaint_images=complaint_images,
        complaint_audio_rows=complaint_audio_rows,
    )


# =========================================================
# OFFICER DASHBOARD
# =========================================================

@app.route("/officer-dashboard")
def officer_dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") != "officer":
        return redirect(url_for("dashboard"))

    # Refresh the assignment from DB so an admin changing an officer's
    # Panchayat takes effect without relying on a stale session value.
    panchayat_id = get_user_panchayat_id(session["user_id"])
    session["panchayat_id"] = panchayat_id
    if panchayat_id is None:
        return "Officer account is not assigned to a Panchayat. Please contact the administrator."

    db = get_db_connection(); cursor = db.cursor(dictionary=True)
    counts = {}
    for key, condition in [("total", "1=1"), ("submitted", "status = 'Submitted'"), ("in_progress", "status = 'In Progress'"), ("resolved", "status = 'Resolved'")]:
        cursor.execute(f"SELECT COUNT(*) AS total FROM complaints WHERE panchayat_id = %s AND {condition}", (panchayat_id,))
        counts[key] = cursor.fetchone()["total"]

    cursor.execute("SELECT name FROM panchayats WHERE id = %s", (panchayat_id,))
    row = cursor.fetchone()
    panchayat_name = row["name"] if row else "Assigned Panchayat"
    cursor.close(); db.close()

    return render_template(
        "officer_dashboard.html",
        user_name=session["user_name"],
        panchayat_name=panchayat_name,
        total_complaints=counts["total"],
        submitted_complaints=counts["submitted"],
        in_progress_complaints=counts["in_progress"],
        resolved_complaints=counts["resolved"],
    )


# =========================================================
# ALL COMPLAINTS
# SEARCH + FILTER
# =========================================================

@app.route("/all-complaints")
def all_complaints():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") not in ["officer", "admin"]:
        return redirect(url_for("dashboard"))

    search = request.args.get("search", "").strip()
    category = request.args.get("category", "").strip()
    status = request.args.get("status", "").strip()
    priority = request.args.get("priority", "").strip()

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT
        complaints.id,
        complaints.category,
        complaints.title,
        complaints.description,
        complaints.priority,
        complaints.status,
        complaints.created_at,
        complaints.panchayat_id,
        users.name,
        users.email,
        users.phone,
        users.village
    FROM complaints
    JOIN users
    ON complaints.user_id = users.id
    WHERE 1=1
    """

    parameters = []

    if session.get("user_role") == "officer":
        panchayat_id = get_user_panchayat_id(session["user_id"])
        session["panchayat_id"] = panchayat_id
        if panchayat_id is None:
            cursor.close(); db.close()
            return "Officer account is not assigned to a Panchayat. Please contact the administrator."
        query += " AND complaints.panchayat_id = %s "
        parameters.append(panchayat_id)

    if search:

        query += """
        AND (
            complaints.title LIKE %s
            OR users.name LIKE %s
        )
        """

        search_value = "%" + search + "%"

        parameters.append(search_value)
        parameters.append(search_value)

    if category:

        query += """
        AND complaints.category = %s
        """

        parameters.append(category)

    if status:

        query += """
        AND complaints.status = %s
        """

        parameters.append(status)

    if priority:

        query += """
        AND complaints.priority = %s
        """

        parameters.append(priority)

    query += """
    ORDER BY complaints.created_at DESC
    """

    cursor.execute(query, tuple(parameters))

    complaints = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "all_complaints.html",
        complaints=complaints,
        search=search,
        category=category,
        status=status,
        priority=priority
    )


# =========================================================
# UPDATE COMPLAINT
# =========================================================

@app.route("/update-complaint/<int:complaint_id>", methods=["GET", "POST"])
def update_complaint(complaint_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") not in ["officer", "admin"]:
        return redirect(url_for("dashboard"))

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT panchayat_id, user_id FROM complaints WHERE id = %s", (complaint_id,))
    access_row = cursor.fetchone()
    cursor.close()

    if not access_row:
        db.close(); return "Complaint not found!"

    if session.get("user_role") == "officer":
        officer_panchayat = get_user_panchayat_id(session["user_id"])
        session["panchayat_id"] = officer_panchayat
        if officer_panchayat is None or access_row["panchayat_id"] != officer_panchayat:
            db.close(); abort(404)

    if request.method == "POST":
        new_status = request.form.get("status", "").strip()
        allowed_statuses = {"Submitted", "In Progress", "Resolved", "Rejected"}
        if new_status not in allowed_statuses:
            db.close(); return "Invalid status!"

        cursor = db.cursor()
        if session.get("user_role") == "officer":
            cursor.execute("UPDATE complaints SET status = %s WHERE id = %s AND panchayat_id = %s", (new_status, complaint_id, session["panchayat_id"]))
        else:
            cursor.execute("UPDATE complaints SET status = %s WHERE id = %s", (new_status, complaint_id))

        message = {
            "Submitted": "Complaint has been submitted.",
            "In Progress": "Officer is currently working on the complaint.",
            "Resolved": "Complaint has been successfully resolved.",
            "Rejected": "Complaint has been rejected."
        }[new_status]
        cursor.execute("INSERT INTO complaint_updates (complaint_id, status, message) VALUES (%s, %s, %s)", (complaint_id, new_status, message))
        cursor.execute("INSERT INTO notifications (user_id, complaint_id, message, is_read) VALUES (%s, %s, %s, %s)", (access_row["user_id"], complaint_id, message, 0))
        db.commit(); cursor.close(); db.close()
        return redirect(url_for("all_complaints"))

    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT complaints.id, complaints.category, complaints.title, complaints.description,
               complaints.priority, complaints.status, complaints.created_at,
               complaints.panchayat_id, users.name, users.email, users.phone, users.village
        FROM complaints JOIN users ON complaints.user_id = users.id
        WHERE complaints.id = %s
    """, (complaint_id,))
    complaint_data = cursor.fetchone(); cursor.close(); db.close()
    if complaint_data is None:
        return "Complaint not found!"

    db = get_db_connection(); dup_cursor = db.cursor(dictionary=True)
    dup_cursor.execute("""
        SELECT id, title, description, status, priority, created_at
        FROM complaints
        WHERE category = %s AND id != %s AND panchayat_id = %s
        ORDER BY created_at DESC LIMIT 200
    """, (complaint_data["category"], complaint_id, complaint_data["panchayat_id"]))
    candidate_complaints = dup_cursor.fetchall(); dup_cursor.close(); db.close()
    similar_complaints = find_similar_complaints(complaint_data["description"], candidate_complaints)

    db = get_db_connection(); c = db.cursor(dictionary=True)
    c.execute("SELECT id, detected_label, confidence, is_low_confidence, ai_suggested_category, ai_suggestion_confirmed, uploaded_at FROM complaint_images WHERE complaint_id = %s ORDER BY uploaded_at DESC", (complaint_id,))
    complaint_images = c.fetchall(); c.close(); db.close()

    db = get_db_connection(); c = db.cursor(dictionary=True)
    c.execute("SELECT id, duration_seconds, language_selected, transcription_engine, raw_transcript, confirmed_transcript, asr_confidence, is_low_confidence, transcription_available, uploaded_at FROM complaint_audio WHERE complaint_id = %s ORDER BY uploaded_at DESC", (complaint_id,))
    complaint_audio_rows = c.fetchall(); c.close(); db.close()

    resolution_estimate = None
    if complaint_data["status"] not in ("Resolved", "Rejected"):
        try:
            recurrence_count = get_recent_similar_count(complaint_data["category"], complaint_data["village"])
        except Exception:
            recurrence_count = 0
        resolution_estimate = predict_resolution_time(complaint_data["category"], complaint_data["priority"], recurrence_count)

    return render_template("update_complaint.html", complaint=complaint_data, similar_complaints=similar_complaints, resolution_estimate=resolution_estimate, complaint_images=complaint_images, complaint_audio_rows=complaint_audio_rows)


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin-dashboard")
def admin_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role = 'citizen'
    """)

    total_citizens = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role = 'officer'
    """)

    total_officers = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM complaints
    """)

    total_complaints = cursor.fetchone()["total"]

    cursor.close()
    db.close()

    return render_template(
        "admin_dashboard.html",
        user_name=session["user_name"],
        total_citizens=total_citizens,
        total_officers=total_officers,
        total_complaints=total_complaints
    )


# =========================================================
# AI ANALYTICS: HOTSPOTS & ANOMALY DETECTION
# =========================================================

def _fetch_complaint_records_for_analytics():
    """Real DB query - category + created_at from complaints, village
    from the submitting citizen. No coordinates exist in this schema
    (see ml/analytics/hotspot_anomaly.py docstring), so 'location' here
    means the village field, not lat/lng."""
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT complaints.category, complaints.created_at, users.village
        FROM complaints
        JOIN users ON complaints.user_id = users.id
    """)
    records = cursor.fetchall()
    cursor.close()
    db.close()
    return records


@app.route("/admin/analytics")
def ai_analytics():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    records = _fetch_complaint_records_for_analytics()

    hotspots = get_hotspots(records, top_n=10, days=30)
    statistical_anomalies = detect_statistical_anomalies(records)
    isolation_anomalies = detect_isolationforest_anomalies(records)

    return render_template(
        "ai_analytics.html",
        user_name=session.get("user_name"),
        hotspots=hotspots,
        statistical_anomalies=statistical_anomalies,
        isolation_anomalies=isolation_anomalies,
        total_records=len(records),
    )


@app.route("/api/analytics/hotspots")
def api_analytics_hotspots():
    if "user_id" not in session or session.get("user_role") != "admin":
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    records = _fetch_complaint_records_for_analytics()
    return jsonify({"available": True, "hotspots": get_hotspots(records, top_n=10, days=30)})


@app.route("/api/analytics/anomalies")
def api_analytics_anomalies():
    if "user_id" not in session or session.get("user_role") != "admin":
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    records = _fetch_complaint_records_for_analytics()
    return jsonify({
        "available": True,
        "statistical": detect_statistical_anomalies(records),
        "isolation_forest": detect_isolationforest_anomalies(records),
    })


# =========================================================
# MANAGE CITIZENS
# =========================================================

@app.route("/manage-citizens")
def manage_citizens():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    query = """
    SELECT
        id,
        name,
        email,
        phone,
        village
    FROM users
    WHERE role = 'citizen'
    ORDER BY id DESC
    """

    cursor.execute(query)

    citizens = cursor.fetchall()

    cursor.close()
    db.close()

    return render_template(
        "manage_citizens.html",
        citizens=citizens
    )


# =========================================================
# DELETE CITIZEN
# =========================================================

@app.route("/delete-citizen/<int:citizen_id>")
def delete_citizen(citizen_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    db = get_db_connection()
    cursor = db.cursor()

    cursor.execute(
        """
        DELETE FROM complaints
        WHERE user_id = %s
        """,
        (citizen_id,)
    )

    cursor.execute(
        """
        DELETE FROM users
        WHERE id = %s
        AND role = 'citizen'
        """,
        (citizen_id,)
    )

    db.commit()

    cursor.close()
    db.close()

    return redirect(url_for("manage_citizens"))


# =========================================================
# MANAGE OFFICERS
# =========================================================

@app.route("/manage-officers")
def manage_officers():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    db = get_db_connection(); cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT users.id, users.name, users.email, users.phone, users.village,
               users.panchayat_id, panchayats.name AS panchayat_name
        FROM users LEFT JOIN panchayats ON users.panchayat_id = panchayats.id
        WHERE users.role = 'officer' ORDER BY users.id DESC
    """)
    officers = cursor.fetchall(); cursor.close(); db.close()
    return render_template("manage_officers.html", officers=officers)


@app.route("/add-officer", methods=["GET", "POST"])
def add_officer():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    panchayats = get_all_panchayats()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        village = request.form.get("village", "").strip()
        panchayat_id = request.form.get("panchayat_id", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not phone or not panchayat_id or not password:
            return "Please fill all required fields."
        if password != confirm_password:
            return "Passwords do not match!"

        db = get_db_connection(); cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id FROM panchayats WHERE id = %s", (panchayat_id,))
        if not cursor.fetchone():
            cursor.close(); db.close(); return "Invalid Panchayat selected!"
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            cursor.close(); db.close(); return "An account with this email already exists!"

        cursor.execute("""
            INSERT INTO users (name, email, phone, village, panchayat_id, password, role)
            VALUES (%s, %s, %s, %s, %s, %s, 'officer')
        """, (name, email, phone, village, panchayat_id, hash_password(password)))
        db.commit(); cursor.close(); db.close()
        return redirect(url_for("manage_officers"))

    return render_template("add_officer.html", panchayats=panchayats)


@app.route("/edit-officer/<int:officer_id>", methods=["GET", "POST"])
def edit_officer(officer_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))

    panchayats = get_all_panchayats()
    db = get_db_connection(); cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, name, email, phone, village, panchayat_id FROM users WHERE id = %s AND role = 'officer'", (officer_id,))
    officer = cursor.fetchone()
    if officer is None:
        cursor.close(); db.close(); return "Officer not found!"

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        village = request.form.get("village", "").strip()
        panchayat_id = request.form.get("panchayat_id", "").strip()
        password = request.form.get("password", "")
        if not name or not email or not phone or not panchayat_id:
            cursor.close(); db.close(); return "Please fill all required fields."

        cursor.execute("SELECT id FROM panchayats WHERE id = %s", (panchayat_id,))
        if not cursor.fetchone():
            cursor.close(); db.close(); return "Invalid Panchayat selected!"
        cursor.execute("SELECT id FROM users WHERE email = %s AND id != %s", (email, officer_id))
        if cursor.fetchone():
            cursor.close(); db.close(); return "Another account already uses this email!"

        if password:
            cursor.execute("""UPDATE users SET name=%s, email=%s, phone=%s, village=%s, panchayat_id=%s, password=%s WHERE id=%s AND role='officer'""", (name, email, phone, village, panchayat_id, hash_password(password), officer_id))
        else:
            cursor.execute("""UPDATE users SET name=%s, email=%s, phone=%s, village=%s, panchayat_id=%s WHERE id=%s AND role='officer'""", (name, email, phone, village, panchayat_id, officer_id))
        db.commit(); cursor.close(); db.close()
        return redirect(url_for("manage_officers"))

    cursor.close(); db.close()
    return render_template("edit_officer.html", officer=officer, panchayats=panchayats)


# =========================================================
# DELETE OFFICER
# =========================================================

@app.route("/delete-officer/<int:officer_id>")
def delete_officer(officer_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("user_role") != "admin":
        return redirect(url_for("dashboard"))
    db = get_db_connection(); cursor = db.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s AND role = 'officer'", (officer_id,))
    db.commit(); cursor.close(); db.close()
    return redirect(url_for("manage_officers"))


# =========================================================
# NOTIFICATIONS
# =========================================================

@app.route("/notification-count")
def notification_count():
    if "user_id" not in session:
        return jsonify({"count": 0})
    db = get_db_connection(); cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM notifications WHERE user_id = %s AND is_read = 0", (session["user_id"],))
    count = cursor.fetchone()[0]; cursor.close(); db.close()
    return jsonify({"count": count})


@app.route("/notifications")
def notifications():
    if "user_id" not in session:
        return redirect(url_for("login"))
    db = get_db_connection(); cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, complaint_id, message, is_read, created_at
        FROM notifications
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (session["user_id"],))
    rows = cursor.fetchall(); cursor.close(); db.close()
    return render_template("notifications.html", notifications=rows)


@app.route("/mark-notification-read/<int:notification_id>", methods=["POST"])
def mark_notification_read(notification_id):
    if "user_id" not in session:
        return jsonify({"success": False}), 401
    db = get_db_connection(); cursor = db.cursor()
    cursor.execute("UPDATE notifications SET is_read = 1 WHERE id = %s AND user_id = %s", (notification_id, session["user_id"]))
    db.commit(); cursor.close(); db.close()
    return jsonify({"success": True})


# =========================================================
# AI ASSISTANT: RAG (Ollama + knowledge base)
# =========================================================

@app.route("/assistant")
def assistant_page():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("assistant.html", user_role=session.get("user_role"))


@app.route("/api/rag/query", methods=["POST"])
def api_rag_query():
    if "user_id" not in session:
        return jsonify({"available": False, "reason": "unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()

    if not question:
        return jsonify({"available": False, "reason": "empty_question"}), 400

    user_id = session["user_id"]
    user_role = session.get("user_role", "citizen")

    # Personal-data path FIRST, and it never touches the LLM (see
    # rag/generation/complaint_lookup.py for why). Authorization is
    # checked in code before any complaint data is returned.
    complaint_id = extract_complaint_id(question)
    if complaint_id is not None:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, user_id, panchayat_id, category, priority, status, created_at, description FROM complaints WHERE id = %s",
            (complaint_id,),
        )
        complaint = cursor.fetchone()
        cursor.close()

        if complaint is None or not rag_is_authorized(complaint, user_id, user_role):
            db.close()
            return jsonify({
                "available": True,
                "answer": rag_not_found_message(),
                "sources": [],
                "grounded": True,
                "degraded": False,
                "personal_data_lookup": True,
            })

        resolution_estimate = None
        if complaint["status"] not in ("Resolved", "Rejected"):
            village_cursor = db.cursor(dictionary=True)
            village_cursor.execute("SELECT village FROM users WHERE id = %s", (complaint["user_id"],))
            village_row = village_cursor.fetchone()
            village_cursor.close()
            village = village_row["village"] if village_row else None
            try:
                recurrence_count = get_recent_similar_count(complaint["category"], village)
            except Exception:  # noqa: BLE001
                recurrence_count = 0
            resolution_estimate = predict_resolution_time(complaint["category"], complaint["priority"], recurrence_count)

        db.close()

        return jsonify({
            "available": True,
            "answer": format_complaint_answer(complaint, resolution_estimate),
            "sources": [],
            "grounded": True,
            "degraded": False,
            "personal_data_lookup": True,
        })

    # General knowledge-base question -> real RAG path.
    result = rag_answer_question(question)
    result["personal_data_lookup"] = False
    return jsonify(result)




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
