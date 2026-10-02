"""
Local embedding model for the RAG pipeline.

HONEST LIMITATION, same as ml/inference/duplicate.py in Phase 3: the
master-prompt ideal is a transformer sentence-embedding model, but this
sandbox has no network route to a model hub (huggingface.co is not
reachable), so a pretrained embedding model cannot be downloaded. This
uses TF-IDF vectors as the "embedding" instead - real, offline,
lexical-similarity-based, not deep semantic understanding.

A REAL FINDING FROM TESTING THIS, not a design choice made in the
abstract: the first version of this file used TF-IDF + TruncatedSVD
(classic LSA) to get dense, lower-dimensional vectors, which is normally
a reasonable move. On this specific knowledge base (14 chunks total), it
backfired badly - completely off-topic queries like "What is the capital
of France?" scored HIGHER similarity than genuinely relevant queries.
With this few documents, SVD has nowhere near enough data to learn a
meaningful latent-topic space; it ends up fitting noise, and the
resulting vectors put unrelated short queries close together. Verified
by comparing raw TF-IDF cosine similarity against the same query set: on
topic queries scored 0.23-0.53, off-topic queries scored 0.00-0.15 - a
real, checkable gap. So this version drops SVD and uses plain
stopword-filtered, unigram TF-IDF vectors directly. See
tests/test_rag_retrieval.py for the regression test that encodes this.

Revisit SVD (or a real embedding model) once the knowledge base is much
larger - LSA needs enough documents to find real latent structure, and a
future larger corpus is exactly the case where it would help rather than
overfit.

UPGRADE PATH (documented, not implemented): once real internet access or
a local model file is available, replace fit()/embed() with
sentence-transformers encode() and keep the same on-disk contract
(embedder.joblib) - nothing in retriever.py or ingest.py would need to
change beyond this file.
"""

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

EMBEDDER_PATH = Path(__file__).resolve().parent.parent / "index" / "embedder.joblib"


class LocalEmbedder:
    def __init__(self):
        # Unigrams only + English stopwords removed: bigrams and stopwords
        # both diluted the signal on this small a corpus during testing.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True, stop_words="english")
        self._fitted = False

    def fit(self, texts: List[str]) -> np.ndarray:
        matrix = self.vectorizer.fit_transform(texts)
        self._fitted = True
        return self._to_dense(matrix)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder has not been fit yet - call fit() during ingestion first.")
        matrix = self.vectorizer.transform(texts)
        return self._to_dense(matrix)

    @staticmethod
    def _to_dense(matrix) -> np.ndarray:
        """TfidfVectorizer L2-normalizes rows by default, so these are
        already unit vectors - inner product == cosine similarity,
        exactly what FAISS's IndexFlatIP expects."""
        return matrix.toarray().astype("float32")

    def save(self, path: Path = EMBEDDER_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer}, path)

    @classmethod
    def load(cls, path: Path = EMBEDDER_PATH) -> "LocalEmbedder":
        bundle = joblib.load(path)
        instance = cls()
        instance.vectorizer = bundle["vectorizer"]
        instance._fitted = True
        return instance
