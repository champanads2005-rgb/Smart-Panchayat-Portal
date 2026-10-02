"""
Tests for rag/generation/complaint_lookup.py - especially the
authorization logic, since this is the piece that keeps personal
complaint data from leaking to unauthorized users.

Run: pytest tests/test_complaint_lookup.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generation.complaint_lookup import (
    extract_complaint_id,
    is_authorized,
    format_complaint_answer,
    not_found_message,
)


def test_extract_complaint_id_hash_format():
    assert extract_complaint_id("what's the status of complaint #12") == 12


def test_extract_complaint_id_word_format():
    assert extract_complaint_id("status of complaint 45 please") == 45


def test_extract_complaint_id_id_keyword():
    assert extract_complaint_id("complaint id 7") == 7


def test_extract_complaint_id_returns_none_for_general_question():
    assert extract_complaint_id("What complaint categories exist?") is None


def test_extract_complaint_id_returns_none_for_empty_text():
    assert extract_complaint_id("") is None
    assert extract_complaint_id(None) is None


def test_citizen_is_authorized_for_own_complaint():
    complaint = {"id": 5, "user_id": 1}
    assert is_authorized(complaint, user_id=1, user_role="citizen") is True


def test_citizen_is_not_authorized_for_others_complaint():
    complaint = {"id": 5, "user_id": 2}
    assert is_authorized(complaint, user_id=1, user_role="citizen") is False


def test_officer_is_authorized_for_any_complaint():
    complaint = {"id": 5, "user_id": 999}
    assert is_authorized(complaint, user_id=1, user_role="officer") is True


def test_admin_is_authorized_for_any_complaint():
    complaint = {"id": 5, "user_id": 999}
    assert is_authorized(complaint, user_id=1, user_role="admin") is True


def test_format_complaint_answer_includes_key_facts():
    complaint = {
        "id": 5, "category": "Water", "priority": "High",
        "status": "In Progress", "created_at": "2026-01-01",
    }
    answer = format_complaint_answer(complaint)
    assert "#5" in answer
    assert "Water" in answer
    assert "In Progress" in answer
    assert "High" in answer


def test_format_complaint_answer_includes_resolution_estimate_when_available():
    complaint = {"id": 5, "category": "Water", "priority": "High", "status": "Submitted", "created_at": "2026-01-01"}
    estimate = {"available": True, "predicted_days": 2.3, "low_confidence": True}
    answer = format_complaint_answer(complaint, estimate)
    assert "2.3 days" in answer
    assert "estimate, not a guaranteed deadline" in answer
    assert "bootstrap model" in answer


def test_format_complaint_answer_omits_estimate_when_unavailable():
    complaint = {"id": 5, "category": "Water", "priority": "High", "status": "Resolved", "created_at": "2026-01-01"}
    answer = format_complaint_answer(complaint, {"available": False})
    assert "estimate" not in answer.lower()


def test_not_found_message_is_generic_and_does_not_leak_existence():
    msg1 = not_found_message()
    msg2 = not_found_message()
    assert msg1 == msg2  # same message regardless of WHY it wasn't found
    assert "under your account" in msg1
