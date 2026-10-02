"""
Tests for rag/retrieval/retriever.py, run against the actual FAISS index
shipped in rag/index/ (built by rag/ingestion/ingest.py). These exercise
the real retrieval quality, not a mocked stand-in.

Run: pytest tests/test_rag_retrieval.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retrieval import retriever


def test_index_is_available():
    """Smoke test that the shipped index actually loads."""
    assert retriever.is_available() is True


def test_on_topic_query_returns_results():
    results = retriever.retrieve("What complaint categories are available?", top_k=3)
    assert len(results) > 0
    assert results[0]["source"] == "complaint_categories.md"


def test_off_topic_query_returns_no_results_at_default_threshold():
    """Regression test for the real finding in Phase 6: with the
    TF-IDF+SVD embedder this used to fail (off-topic scored higher than
    on-topic). With plain TF-IDF, completely unrelated questions should
    score below the calibrated 0.20 threshold and return nothing."""
    results = retriever.retrieve("What is the capital of France?", top_k=3)
    assert results == []


def test_results_are_sorted_by_score_descending():
    results = retriever.retrieve("How do I submit a complaint?", top_k=5, min_similarity=0.0)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_no_results():
    assert retriever.retrieve("", top_k=3) == []
    assert retriever.retrieve("   ", top_k=3) == []


def test_min_similarity_threshold_is_respected():
    results = retriever.retrieve("How do I submit a complaint?", top_k=5, min_similarity=0.9)
    assert all(r["score"] >= 0.9 for r in results)


def test_top_k_limits_result_count():
    results = retriever.retrieve("complaint", top_k=2, min_similarity=0.0)
    assert len(results) <= 2
