"""
Orchestrates the knowledge-base RAG path: retrieve -> build prompt ->
generate -> attach sources. Every failure mode degrades gracefully
instead of raising:

  - retriever not built/empty KB  -> available=False, clear reason
  - no relevant chunks found      -> available=True, explicit "couldn't
                                      find this" answer, no LLM call made
  - Ollama unreachable/model      -> available=True, degraded=True,
    missing/times out               returns the raw retrieved source
                                     text instead of a synthesized answer
                                     (retrieval still worked - only the
                                     natural-language generation step
                                     didn't)
  - generation succeeds           -> available=True, degraded=False,
                                      answer + sources
"""

from typing import List

from rag.retrieval import retriever
from rag.generation.prompt_builder import build_prompt
from rag.generation.llm_service import generate, OllamaUnavailableError


def _sources_for_response(chunks: List[dict]) -> List[dict]:
    return [{"source": c["source"], "heading": c["heading"], "score": c["score"]} for c in chunks]


def answer_question(question: str, top_k: int = 3) -> dict:
    if not retriever.is_available():
        return {
            "available": False,
            "reason": retriever.get_load_error(),
            "answer": None,
            "sources": [],
        }

    chunks = retriever.retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "available": True,
            "answer": "I couldn't find this information in the Panchayat knowledge base.",
            "sources": [],
            "grounded": True,
            "degraded": False,
        }

    prompt = build_prompt(question, chunks)

    try:
        text = generate(prompt)
    except OllamaUnavailableError as exc:
        return {
            "available": True,
            "answer": None,
            "degraded": True,
            "degraded_reason": str(exc),
            "sources": _sources_for_response(chunks),
            "raw_context": [c["text"] for c in chunks],
            "grounded": True,
        }

    return {
        "available": True,
        "answer": text.strip(),
        "degraded": False,
        "sources": _sources_for_response(chunks),
        "grounded": True,
    }
