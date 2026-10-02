"""
Tests for rag/generation/prompt_builder.py - pure function, no I/O.

Run: pytest tests/test_rag_prompt_builder.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generation.prompt_builder import build_prompt, SYSTEM_INSTRUCTIONS


def test_prompt_includes_system_instructions():
    prompt = build_prompt("What is Under Review?", [])
    assert SYSTEM_INSTRUCTIONS in prompt


def test_prompt_includes_question():
    question = "What does the Sanitation category cover?"
    prompt = build_prompt(question, [])
    assert question in prompt


def test_prompt_with_no_chunks_tells_model_context_is_empty():
    prompt = build_prompt("anything", [])
    assert "no relevant context was found" in prompt


def test_prompt_includes_chunk_text_and_source():
    chunks = [
        {"source": "complaint_categories.md", "heading": "Complaint Categories", "text": "Water means drinking water issues.", "score": 0.5},
    ]
    prompt = build_prompt("What is Water category?", chunks)
    assert "complaint_categories.md" in prompt
    assert "Water means drinking water issues." in prompt


def test_prompt_includes_multiple_chunks_with_separate_source_labels():
    chunks = [
        {"source": "a.md", "heading": "A", "text": "Text A", "score": 0.5},
        {"source": "b.md", "heading": "B", "text": "Text B", "score": 0.4},
    ]
    prompt = build_prompt("q", chunks)
    assert "a.md" in prompt and "b.md" in prompt
    assert "Text A" in prompt and "Text B" in prompt


def test_prompt_instructs_model_to_decline_when_unsure():
    prompt = build_prompt("q", [])
    assert "I couldn't find this information in the Panchayat knowledge base." in prompt


def test_prompt_instructs_model_not_to_use_outside_knowledge():
    prompt = build_prompt("q", [])
    assert "ONLY" in prompt
    assert "outside knowledge" in prompt
