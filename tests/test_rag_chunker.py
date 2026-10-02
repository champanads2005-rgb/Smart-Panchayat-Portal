"""
Tests for rag/ingestion/chunker.py.

Run: pytest tests/test_rag_chunker.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from rag.ingestion.chunker import chunk_text, chunk_markdown_file


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert chunks == [text]


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text_with_overlap():
    words = [f"word{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every word should appear in at least one chunk
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_chunk_text_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_markdown_file_splits_by_heading():
    md = "# Heading One\nSome text under heading one.\n\n## Heading Two\nSome text under heading two."
    chunks = chunk_markdown_file(md, chunk_size=120, overlap=30)
    headings = {c["heading"] for c in chunks}
    assert "Heading One" in headings
    assert "Heading Two" in headings


def test_chunk_markdown_file_empty_input():
    assert chunk_markdown_file("") == []


def test_chunk_markdown_file_small_chunks_isolate_list_items():
    """Regression test for the real retrieval-quality finding in Phase 6:
    a bulleted list of six status definitions needs small chunk_size to
    avoid diluting each individual status's keywords across one giant
    chunk (see rag/ingestion/ingest.py docstring)."""
    md = (
        "# Status\n"
        "- Submitted: complaint received.\n"
        "- Under Review: officer checking details.\n"
        "- Resolved: issue fixed.\n"
    )
    chunks = chunk_markdown_file(md, chunk_size=8, overlap=2)
    assert len(chunks) > 1
    assert any("Under Review" in c["text"] for c in chunks)
