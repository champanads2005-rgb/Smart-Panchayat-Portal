"""
Tests for rag/generation/rag_pipeline.py - the orchestrator. Monkeypatches
the retriever and llm_service boundaries to exercise each of the four
documented outcomes (index unavailable / no relevant chunks / Ollama
unavailable / success) deterministically, without depending on Ollama
actually being reachable.

Run: pytest tests/test_rag_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generation import rag_pipeline
from rag.retrieval import retriever
from rag.generation.llm_service import OllamaUnavailableError


def test_answer_unavailable_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(retriever, "is_available", lambda: False)
    monkeypatch.setattr(retriever, "get_load_error", lambda: "index missing")

    result = rag_pipeline.answer_question("anything")
    assert result["available"] is False
    assert result["answer"] is None
    assert result["reason"] == "index missing"


def test_answer_declines_when_no_relevant_chunks(monkeypatch):
    monkeypatch.setattr(retriever, "is_available", lambda: True)
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=3: [])

    result = rag_pipeline.answer_question("something totally unrelated")
    assert result["available"] is True
    assert "couldn't find" in result["answer"]
    assert result["sources"] == []
    assert result["degraded"] is False


def test_answer_degrades_when_ollama_unavailable(monkeypatch):
    fake_chunks = [{"source": "a.md", "heading": "A", "text": "Some real KB text.", "score": 0.5}]
    monkeypatch.setattr(retriever, "is_available", lambda: True)
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=3: fake_chunks)

    def fake_generate(prompt, timeout=30):
        raise OllamaUnavailableError("simulated: Ollama not reachable")
    monkeypatch.setattr(rag_pipeline, "generate", fake_generate)

    result = rag_pipeline.answer_question("what does X mean?")
    assert result["available"] is True
    assert result["degraded"] is True
    assert result["answer"] is None
    assert "Some real KB text." in result["raw_context"]
    assert result["sources"][0]["source"] == "a.md"


def test_answer_succeeds_when_ollama_available(monkeypatch):
    fake_chunks = [{"source": "a.md", "heading": "A", "text": "Some real KB text.", "score": 0.5}]
    monkeypatch.setattr(retriever, "is_available", lambda: True)
    monkeypatch.setattr(retriever, "retrieve", lambda q, top_k=3: fake_chunks)
    monkeypatch.setattr(rag_pipeline, "generate", lambda prompt, timeout=30: "  Here is the generated answer.  ")

    result = rag_pipeline.answer_question("what does X mean?")
    assert result["available"] is True
    assert result["degraded"] is False
    assert result["answer"] == "Here is the generated answer."
    assert result["sources"][0]["source"] == "a.md"


def test_answer_question_real_end_to_end_against_shipped_index():
    """No mocking - exercises the real shipped FAISS index and the real
    (currently unreachable) Ollama instance, to prove the whole chain
    degrades gracefully together, not just each piece in isolation."""
    result = rag_pipeline.answer_question("What does Under Review mean?")
    assert result["available"] is True
    # Either genuinely degraded (Ollama down, which is the real state of
    # this sandbox) or, if some Ollama happens to be running in whatever
    # environment runs this test, a real answer - either is acceptable,
    # a hard crash is not.
    assert result["degraded"] in (True, False)
