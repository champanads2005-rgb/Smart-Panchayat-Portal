"""
Document ingestion pipeline: reads every .md/.txt file in
rag/knowledge_base/, chunks it, embeds the chunks, and builds a FAISS
vector index + a metadata sidecar file mapping each vector back to its
source document, heading, and text.

CHUNK SIZE (60 words, 15 overlap) was tuned empirically, not guessed:
the default 120-word chunking put an entire bulleted list of six status
definitions into one chunk, so a query like "What does Under Review
mean?" diluted "Under Review" among five other unrelated status names
and scored too low to be retrieved (0.139, below the 0.20 threshold in
retriever.py). Shrinking to 60 words split that list roughly one
bullet per chunk, and the same query correctly retrieved the right
chunk afterward. See tests/test_rag_retrieval.py for the regression
test encoding this.

Run:
    python rag/ingestion/ingest.py
Produces:
    rag/index/faiss.index
    rag/index/metadata.json
    rag/index/embedder.joblib
"""

import json
import sys
from pathlib import Path

import faiss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.ingestion.chunker import chunk_markdown_file
from rag.embeddings.embedder import LocalEmbedder, EMBEDDER_PATH

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"
INDEX_DIR = Path(__file__).resolve().parent.parent / "index"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"


def load_documents():
    """Skips README.md - that's ingestion-pipeline documentation for
    developers, not citizen-facing knowledge base content."""
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        if path.name == "README.md":
            continue
        docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def main():
    documents = load_documents()
    if not documents:
        print(f"No knowledge base documents found in {KB_DIR} - nothing to ingest.")
        return

    all_chunks = []  # list of {"source", "heading", "text"}
    for doc in documents:
        for chunk in chunk_markdown_file(doc["text"], chunk_size=60, overlap=15):
            all_chunks.append({
                "source": doc["source"],
                "heading": chunk["heading"],
                "text": chunk["text"],
            })

    if not all_chunks:
        print("Documents were found but produced zero chunks - nothing to index.")
        return

    print(f"Loaded {len(documents)} documents -> {len(all_chunks)} chunks.")

    texts = [c["text"] for c in all_chunks]
    embedder = LocalEmbedder()
    vectors = embedder.fit(texts)

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # inner product on normalized vectors = cosine similarity
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    embedder.save(EMBEDDER_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"Wrote FAISS index ({dimension}-dim, {index.ntotal} vectors) to {FAISS_INDEX_PATH}")
    print(f"Wrote metadata for {len(all_chunks)} chunks to {METADATA_PATH}")
    print(f"Wrote embedder to {EMBEDDER_PATH}")


if __name__ == "__main__":
    main()
