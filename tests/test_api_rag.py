"""
Integration tests for GET /assistant and POST /api/rag/query.

DB access is mocked; the real rag_pipeline is exercised for the
general-question path (against the actual shipped index), and the
complaint-lookup path is tested with a fake DB to verify the
authorization logic end-to-end through the Flask route.

Run: pytest tests/test_api_rag.py -v
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "root")
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_NAME", "test_db")

import pytest

import app as appmodule


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self._i = 0

    def execute(self, *args, **kwargs):
        pass

    def fetchone(self):
        row = self._rows[self._i] if self._i < len(self._rows) else None
        self._i += 1
        return row

    def close(self):
        pass


class _FakeDB:
    """Each call to .cursor() returns a cursor over the NEXT list in
    cursor_rows_sequence, so a route that opens multiple cursors in
    sequence (complaint lookup, then village lookup) gets the right
    canned row each time."""
    def __init__(self, cursor_rows_sequence):
        self._sequence = cursor_rows_sequence
        self._call = 0

    def cursor(self, dictionary=False):
        rows = self._sequence[self._call] if self._call < len(self._sequence) else []
        self._call += 1
        return _FakeCursor(rows)

    def close(self):
        pass


@pytest.fixture
def client():
    appmodule.app.config["TESTING"] = True
    return appmodule.app.test_client()


def _login(client, user_id=1, role="citizen"):
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_role"] = role
        sess["user_name"] = "Test User"


def test_assistant_page_requires_login(client):
    resp = client.get("/assistant")
    assert resp.status_code == 302


def test_assistant_page_renders_when_logged_in(client):
    _login(client)
    resp = client.get("/assistant")
    assert resp.status_code == 200


def test_rag_query_requires_login(client):
    resp = client.post("/api/rag/query", json={"question": "hello"})
    assert resp.status_code == 401


def test_rag_query_rejects_empty_question(client):
    _login(client)
    resp = client.post("/api/rag/query", json={"question": "   "})
    assert resp.status_code == 400


def test_rag_query_general_question_uses_real_pipeline(client):
    """No mocking of the RAG pipeline itself - exercises the real index
    and the real (unreachable) Ollama, same as
    test_rag_pipeline.py::test_answer_question_real_end_to_end."""
    _login(client)
    resp = client.post("/api/rag/query", json={"question": "What complaint categories are available?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["available"] is True
    assert data["personal_data_lookup"] is False


def test_rag_query_complaint_lookup_owner_authorized(client, monkeypatch):
    complaint_row = [{
        "id": 5, "user_id": 1, "category": "Water", "priority": "High",
        "status": "In Progress", "created_at": "2026-01-01", "description": "x",
    }]
    village_row = [{"village": "Kadri"}]
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB([complaint_row, village_row]))
    monkeypatch.setattr(appmodule, "get_recent_similar_count", lambda cat, vil, days=30: 1)

    _login(client, user_id=1, role="citizen")
    resp = client.post("/api/rag/query", json={"question": "what's the status of complaint #5"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["personal_data_lookup"] is True
    assert "#5" in data["answer"]
    assert "In Progress" in data["answer"]


def test_rag_query_complaint_lookup_blocks_other_citizen(client, monkeypatch):
    """The security-critical case: a citizen who does NOT own complaint
    #5 must get the generic not-found message, never the complaint's
    details."""
    complaint_row = [{
        "id": 5, "user_id": 999, "category": "Water", "priority": "High",
        "status": "In Progress", "created_at": "2026-01-01", "description": "x",
    }]
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB([complaint_row]))

    _login(client, user_id=1, role="citizen")
    resp = client.post("/api/rag/query", json={"question": "what's the status of complaint #5"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answer"] == "I couldn't find that complaint under your account."
    assert "In Progress" not in data["answer"]
    assert "Water" not in data["answer"]


def test_rag_query_complaint_lookup_allows_officer_for_any_complaint(client, monkeypatch):
    complaint_row = [{
        "id": 5, "user_id": 999, "category": "Water", "priority": "High",
        "status": "Resolved", "created_at": "2026-01-01", "description": "x",
    }]
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB([complaint_row]))

    _login(client, user_id=42, role="officer")
    resp = client.post("/api/rag/query", json={"question": "status of complaint 5"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["personal_data_lookup"] is True
    assert "#5" in data["answer"]


def test_rag_query_complaint_lookup_nonexistent_complaint(client, monkeypatch):
    monkeypatch.setattr(appmodule, "get_db_connection", lambda: _FakeDB([[]]))  # fetchone -> None

    _login(client, user_id=1, role="citizen")
    resp = client.post("/api/rag/query", json={"question": "status of complaint 999999"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["answer"] == "I couldn't find that complaint under your account."
