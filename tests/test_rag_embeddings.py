"""
Tests for rag/embeddings/embedder.py.

Run: pytest tests/test_rag_embeddings.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from rag.embeddings.embedder import LocalEmbedder


SAMPLE_DOCS = [
    "The water supply is disrupted in several villages this week.",
    "Drinking water pipelines need urgent repair near the tank.",
    "The road has a large pothole causing accidents near the school.",
    "Street lights are broken and the area is dark at night.",
    "Garbage collection has been delayed in the sanitation department.",
]


def test_fit_returns_normalized_vectors():
    embedder = LocalEmbedder()
    vectors = embedder.fit(SAMPLE_DOCS)
    assert vectors.shape[0] == len(SAMPLE_DOCS)
    norms = np.linalg.norm(vectors, axis=1)
    # Rows should be unit-length (or zero for an all-stopword doc, not the case here)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_before_fit_raises():
    embedder = LocalEmbedder()
    with pytest.raises(RuntimeError):
        embedder.embed(["some query"])


def test_embed_matches_fit_dimensionality():
    embedder = LocalEmbedder()
    embedder.fit(SAMPLE_DOCS)
    query_vec = embedder.embed(["Is there a water problem?"])
    assert query_vec.shape[1] == embedder.vectorizer.transform(SAMPLE_DOCS).shape[1]


def test_save_and_load_round_trip(tmp_path):
    embedder = LocalEmbedder()
    embedder.fit(SAMPLE_DOCS)
    path = tmp_path / "embedder_test.joblib"
    embedder.save(path)

    loaded = LocalEmbedder.load(path)
    original_vec = embedder.embed(["water problem in the village"])
    loaded_vec = loaded.embed(["water problem in the village"])
    assert np.allclose(original_vec, loaded_vec)


def test_on_topic_query_scores_higher_than_off_topic():
    """Regression test for the real bug found during Phase 6 development:
    the first version of this embedder used TF-IDF + TruncatedSVD, and on
    this small a corpus, a completely unrelated query ('What is the
    capital of France?') scored HIGHER cosine similarity than a genuinely
    relevant one. This test locks in the fix (plain TF-IDF, no SVD) by
    asserting the on-topic query wins - it would fail again if SVD were
    reintroduced without enough data to support it."""
    embedder = LocalEmbedder()
    doc_vectors = embedder.fit(SAMPLE_DOCS)

    on_topic = embedder.embed(["Is there a drinking water shortage?"])[0]
    off_topic = embedder.embed(["What is the capital of France?"])[0]

    on_topic_best = max(float(doc_vectors[i] @ on_topic) for i in range(len(SAMPLE_DOCS)))
    off_topic_best = max(float(doc_vectors[i] @ off_topic) for i in range(len(SAMPLE_DOCS)))

    assert on_topic_best > off_topic_best
    assert on_topic_best > 0.15  # genuinely relevant match, not a coincidental sliver
