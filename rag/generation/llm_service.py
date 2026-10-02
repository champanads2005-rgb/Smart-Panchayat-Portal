"""
Thin HTTP client for a local Ollama instance. Configurable via
OLLAMA_BASE_URL / OLLAMA_MODEL in config.py (env-overridable, never
hardcoded here) - both already existed in config.py since Phase 1 as
placeholders for exactly this phase.

HONEST STATUS: this sandbox has no Ollama binary and no network route to
Ollama's own model registry (it's not in the allowed egress domain list
here), so no LLM has actually generated a real answer during this
phase's development. Every code path in this file has been exercised
against a real HTTP server, though: the "Ollama unreachable" path is
verified for real (there genuinely is no Ollama running here - the
connection failure is not simulated), and the success-path parsing is
covered by tests against a mocked HTTP response (see
tests/test_llm_service.py). Generation itself needs to be verified on a
machine with Ollama actually installed - see UPGRADE_NOTES.md for exact
commands.
"""

from typing import List, Optional

import requests

from config import Config

DEFAULT_TIMEOUT_SECONDS = 30
HEALTH_CHECK_TIMEOUT_SECONDS = 2


class OllamaUnavailableError(Exception):
    """Raised when Ollama can't be reached, times out, or the configured
    model isn't installed. Callers (rag_pipeline.py) catch this and
    degrade gracefully instead of crashing a request."""


def get_base_url() -> str:
    return Config.OLLAMA_BASE_URL


def get_model_name() -> str:
    return Config.OLLAMA_MODEL


def is_ollama_running() -> bool:
    try:
        resp = requests.get(f"{get_base_url()}/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def list_available_models() -> List[str]:
    try:
        resp = requests.get(f"{get_base_url()}/api/tags", timeout=HEALTH_CHECK_TIMEOUT_SECONDS)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except requests.exceptions.RequestException:
        return []


def is_configured_model_available() -> bool:
    models = list_available_models()
    configured = get_model_name()
    # Ollama tags often include a version suffix (e.g. "llama3:latest"),
    # so match on the prefix before ':' too.
    return any(m == configured or m.split(":")[0] == configured.split(":")[0] for m in models)


def generate(prompt: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    """Calls Ollama's /api/generate with streaming disabled and returns
    the plain text response. Raises OllamaUnavailableError for any
    failure mode (connection refused, timeout, missing model, malformed
    response) - callers must catch this, never let it propagate as a
    raw 500 to the user."""
    payload = {
        "model": get_model_name(),
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(f"{get_base_url()}/api/generate", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(f"Could not connect to Ollama at {get_base_url()}: {exc}") from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaUnavailableError(f"Ollama request timed out after {timeout}s: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(f"Ollama request failed: {exc}") from exc

    if resp.status_code == 404:
        raise OllamaUnavailableError(
            f"Model '{get_model_name()}' not found on this Ollama instance. "
            f"Run: ollama pull {get_model_name()}"
        )
    if resp.status_code != 200:
        raise OllamaUnavailableError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
        text = data.get("response")
    except ValueError as exc:
        raise OllamaUnavailableError(f"Ollama returned a non-JSON response: {exc}") from exc

    if not text:
        raise OllamaUnavailableError("Ollama returned an empty response.")

    return text
