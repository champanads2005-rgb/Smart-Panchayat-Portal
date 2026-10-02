"""
Duplicate / related complaint detection.

HONEST LIMITATION UP FRONT: the master-prompt ideal here is transformer
sentence embeddings (e.g. Sentence-BERT). This sandbox has no network
access to a model hub (huggingface.co is not reachable from here), so a
pretrained embedding model cannot be downloaded. Rather than fake
"semantic similarity" with a model that was never actually loaded, this
phase implements a genuinely-computed, weaker but real alternative:

    similarity(a, b) = 0.5 * cosine(TF-IDF word 1-2gram)
                      + 0.5 * cosine(TF-IDF char 3-5gram)

Word n-grams catch shared vocabulary; character n-grams catch shared
morphology/phrasing even when word choice differs slightly ("hasn't been
coming" vs "no ... supply"). This is still lexical, not semantic - it
will miss true paraphrases that share no surface form at all. It is a
real, inspectable, offline-computable signal, not a placeholder.

THRESHOLD: 0.12, chosen from an actual manual calibration (see
UPGRADE_NOTES.md) comparing hand-written duplicate pairs (scored
0.13-0.44) against hand-written unrelated pairs (scored 0.00-0.05) on a
small test set - not an arbitrary guess, but also not derived from real
complaint data (none existed to calibrate against). Recalibrate once real
complaint history accumulates.

UPGRADE PATH (documented, not implemented here): once real internet
access / a model file is available, replace `_similarity_matrix` below
with sentence-transformers encode() + cosine_similarity and keep the same
`find_similar_complaints` interface - nothing else in app.py would need
to change.
"""

from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_THRESHOLD = 0.12
MAX_RESULTS = 5


def _cosine_to_all(query_text: str, candidate_texts: List[str], analyzer: str, ngram_range: tuple):
    """Fits a TF-IDF vectorizer over [query] + candidates and returns the
    cosine similarity of the query against every candidate. Fit fresh each
    call - fine at this data scale; would be cached/batched for a large
    complaint volume."""
    if not candidate_texts:
        return []
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    matrix = vec.fit_transform([query_text] + candidate_texts)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return sims.tolist()


def find_similar_complaints(
    new_text: str,
    candidates: List[Dict],
    threshold: float = DEFAULT_THRESHOLD,
    max_results: int = MAX_RESULTS,
) -> List[Dict]:
    """
    candidates: list of dicts, each with at least {"id", "description"}
                and any extra display fields (title, status, created_at)
                to pass through untouched.
    Returns candidates with a similarity score above `threshold`, sorted
    descending, each with a "similarity" key added.
    """
    if not new_text or not new_text.strip() or not candidates:
        return []

    texts = [c["description"] for c in candidates]

    word_sims = _cosine_to_all(new_text, texts, "word", (1, 2))
    char_sims = _cosine_to_all(new_text, texts, "char_wb", (3, 5))

    results = []
    for candidate, w_sim, c_sim in zip(candidates, word_sims, char_sims):
        combined = 0.5 * w_sim + 0.5 * c_sim
        if combined >= threshold:
            enriched = dict(candidate)
            enriched["similarity"] = round(combined, 3)
            results.append(enriched)

    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:max_results]
