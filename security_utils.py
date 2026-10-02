"""
Security helpers: password hashing and a lightweight session-based CSRF
token, with no extra heavyweight dependency needed beyond Werkzeug (already
a Flask dependency).
"""

import secrets
import uuid

from flask import abort, session
from werkzeug.security import check_password_hash, generate_password_hash


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """PBKDF2-SHA256 via Werkzeug's generate_password_hash (salted)."""
    return generate_password_hash(plain_password)


def verify_password(plain_password: str, stored_hash: str) -> bool:
    return check_password_hash(stored_hash, plain_password)


def looks_like_hash(value: str) -> bool:
    """Werkzeug hashes look like 'method:salt:hash', e.g. 'pbkdf2:sha256:...'."""
    return isinstance(value, str) and value.count("$") + value.count(":") >= 2 and len(value) > 40


# ---------------------------------------------------------------------------
# CSRF protection (session-token based, framework-agnostic)
# ---------------------------------------------------------------------------

def get_csrf_token() -> str:
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]


def validate_csrf(form_token: str) -> None:
    session_token = session.get("csrf_token")
    if not session_token or not form_token or not secrets.compare_digest(session_token, form_token):
        abort(400, description="Invalid or missing CSRF token.")


# ---------------------------------------------------------------------------
# Phase 7: safe storage filenames for uploaded complaint images
# ---------------------------------------------------------------------------

def generate_safe_stored_filename(extension: str) -> str:
    """Generates a random, server-controlled filename for a stored
    upload. NEVER derives any part of the on-disk filename from a
    client-supplied name - that is what prevents path traversal and
    collisions, regardless of what the citizen's browser sent as the
    original filename. `extension` must already be validated by the
    caller (e.g. "jpg") - this function does not sanitize it itself.
    """
    return f"{uuid.uuid4().hex}.{extension}"
