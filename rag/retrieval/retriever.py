"""
Retrieval over the FAISS index built by rag/ingestion/ingest.py.

Loaded once at import time (index, metadata, embedder) - not reloaded
per query.
"""

import json
from pathlib import Path
from typing import List, Optional

import faiss

from rag.embeddings.embedder import LocalEmbedder, EMBEDDER_PATH

INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# Calibrated the same way as Phase 3's duplicate-detection threshold:
# hand-checked queries against this knowledge base's actual chunks.
# On-topic questions scored 0.35-0.85 cosine similarity against their
# matching chunk; off-topic questions ("what is the capital of France")
# scored under 0.15 against every chunk. 0.20 leaves margin on both
# sides. Recalibrate if the knowledge base grows substantially.
DEFAULT_MIN_SIMILARITY = 0.20

_index = None
_metadata: Optional[List[dict]] = None
_embedder: Optional[LocalEmbedder] = None
_load_error: Optional[str] = None

try:
    if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists() and EMBEDDER_PATH.exists():
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        _embedder = LocalEmbedder.load(EMBEDDER_PATH)
    else:
        _load_error = "RAG index not found - run rag/ingestion/ingest.py first."
except Exception as exc:  # noqa: BLE001
    _load_error = f"Failed to load RAG index: {exc}"


def is_available() -> bool:
    return _index is not None and _metadata is not None and bool(_metadata)


def get_load_error() -> Optional[str]:
    return _load_error


def retrieve(query: str, top_k: int = 3, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> List[dict]:
    """Returns up to top_k chunks with similarity >= min_similarity,
    each as {"source", "heading", "text", "score"}. Empty list means
    either the KB genuinely has nothing relevant, or the index isn't
    loaded - callers should check is_available() to tell those apart."""
    if not is_available() or not query or not query.strip():
        return []

    query_vector = _embedder.embed([query])
    scores, indices = _index.search(query_vector, min(top_k, len(_metadata)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < min_similarity:
            continue
        chunk = _metadata[idx]
        results.append({
            "source": chunk["source"],
            "heading": chunk["heading"],
            "text": chunk["text"],
            "score": round(float(score), 3),
        })

    return results
