"""
Tests for rag/generation/llm_service.py.

Most tests mock `requests` to simulate specific Ollama response shapes
(missing model, malformed JSON, etc.) without needing a real server.
One test (test_generate_raises_on_real_connection_refused) makes a REAL
HTTP call to localhost with no server running, to prove the
"unreachable" code path isn't just theoretical.

Run: pytest tests/test_llm_service.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests

from rag.generation import llm_service
from rag.generation.llm_service import OllamaUnavailableError


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


def test_generate_raises_on_real_connection_refused():
    """No mocking here - there really is no Ollama server running in
    this environment, so this proves the failure path is genuine."""
    with pytest.raises(OllamaUnavailableError):
        llm_service.generate("hello", timeout=2)


def test_is_ollama_running_false_on_connection_error(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(requests, "get", fake_get)
    assert llm_service.is_ollama_running() is False


def test_is_ollama_running_true_on_200(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse(status_code=200))
    assert llm_service.is_ollama_running() is True


def test_list_available_models_parses_tags_response(monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: _FakeResponse(status_code=200, json_data={"models": [{"name": "llama3:latest"}, {"name": "mistral"}]})
    )
    models = llm_service.list_available_models()
    assert "llama3:latest" in models
    assert "mistral" in models


def test_is_configured_model_available_matches_prefix(monkeypatch):
    monkeypatch.setattr(llm_service, "list_available_models", lambda: ["llama3:latest"])
    monkeypatch.setattr(llm_service, "get_model_name", lambda: "llama3")
    assert llm_service.is_configured_model_available() is True


def test_is_configured_model_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(llm_service, "list_available_models", lambda: ["mistral"])
    monkeypatch.setattr(llm_service, "get_model_name", lambda: "llama3")
    assert llm_service.is_configured_model_available() is False


def test_generate_success_returns_response_text(monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: _FakeResponse(status_code=200, json_data={"response": "Here is the answer."})
    )
    result = llm_service.generate("some prompt")
    assert result == "Here is the answer."


def test_generate_raises_on_model_not_found(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=404, text="not found"))
    with pytest.raises(OllamaUnavailableError, match="not found"):
        llm_service.generate("some prompt")


def test_generate_raises_on_non_200(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=500, text="server error"))
    with pytest.raises(OllamaUnavailableError):
        llm_service.generate("some prompt")


def test_generate_raises_on_malformed_json(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=200, json_data=None))
    with pytest.raises(OllamaUnavailableError):
        llm_service.generate("some prompt")


def test_generate_raises_on_empty_response_field(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(status_code=200, json_data={"response": ""}))
    with pytest.raises(OllamaUnavailableError):
        llm_service.generate("some prompt")


def test_generate_raises_on_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")
    monkeypatch.setattr(requests, "post", fake_post)
    with pytest.raises(OllamaUnavailableError, match="timed out"):
        llm_service.generate("some prompt", timeout=1)
