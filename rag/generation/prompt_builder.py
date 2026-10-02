"""
Prompt construction for the RAG assistant. Pure function - no I/O - so
it's directly unit-testable (see tests/test_rag_prompt_builder.py)
without needing Ollama or the retriever running.
"""

from typing import List

SYSTEM_INSTRUCTIONS = (
    "You are the Smart Panchayat AI assistant. Answer the citizen's question "
    "using ONLY the information in the CONTEXT section below. "
    "Do not use any outside knowledge about government schemes, laws, or "
    "procedures that is not explicitly stated in the CONTEXT. "
    "If the CONTEXT does not contain enough information to answer the "
    "question, respond with exactly this sentence and nothing else: "
    "\"I couldn't find this information in the Panchayat knowledge base.\" "
    "When you do answer from the CONTEXT, mention which source document(s) "
    "you used."
)


def build_prompt(question: str, chunks: List[dict]) -> str:
    """
    chunks: list of {"source", "heading", "text", "score"} as returned by
    rag/retrieval/retriever.py. If empty, the prompt still explicitly
    tells the model there is no context, so it must decline rather than
    invent an answer from its own training data.
    """
    if not chunks:
        context_block = "(no relevant context was found in the knowledge base)"
    else:
        parts = []
        for chunk in chunks:
            parts.append(f"[Source: {chunk['source']} — {chunk['heading']}]\n{chunk['text']}")
        context_block = "\n\n".join(parts)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:"
    )
