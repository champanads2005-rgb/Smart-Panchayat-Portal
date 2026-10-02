"""
Pure text-chunking logic for the RAG ingestion pipeline. No I/O, no
embedding - directly unit-testable (see tests/test_rag_chunker.py).
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    """
    Splits `text` into overlapping word-count windows. Word-count based
    (not character-based) chunking keeps chunks roughly sentence-aligned
    for prose documents like these knowledge-base files.

    chunk_size: target words per chunk.
    overlap: words repeated between consecutive chunks, so a sentence
             split across a chunk boundary still appears whole in at
             least one chunk.
    """
    if not text or not text.strip():
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        window = words[start:start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_markdown_file(text: str, chunk_size: int = 120, overlap: int = 30) -> List[dict]:
    """
    Splits a markdown document by its '# ' / '## ' headers first (so a
    chunk doesn't straddle two unrelated sections), then applies
    chunk_text within each section. Returns a list of
    {"heading": str, "text": str} dicts.
    """
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    sections = []
    current_heading = "(document start)"
    current_lines: List[str] = []

    def flush():
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))

    for line in lines:
        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()

    results = []
    for heading, section_text in sections:
        for chunk in chunk_text(section_text, chunk_size=chunk_size, overlap=overlap):
            results.append({"heading": heading, "text": chunk})

    return results
