"""
Evaluates the RAG pipeline against rag/evaluation/eval_dataset.json.

HONEST SCOPE: this sandbox has no reachable Ollama instance, so
generation/groundedness/hallucination-rate cannot be measured here -
those require an actual LLM to produce actual text to score. What CAN be
measured for real, with no LLM needed, is retrieval quality:

  - Retrieval hit rate: for in-scope questions, does the expected source
    document appear in the top-k retrieved chunks?
  - Correct-decline rate: for out-of-scope questions, does retrieval
    correctly return NOTHING above the similarity threshold (i.e. would
    the pipeline correctly say "I couldn't find this" instead of
    hallucinating from irrelevant context)?

If Ollama IS reachable when this script runs (e.g. on a machine with it
installed), it also runs the full generate() path for each in-scope
question and reports whether the source document name appears in the
model's answer, as a crude, real, groundedness proxy (not a substitute
for human evaluation, and reported as a separate, clearly-labeled
section, not blended into the retrieval numbers).

Run:
    python rag/evaluation/evaluate_rag.py
Produces:
    rag/evaluation/results.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag.retrieval import retriever
from rag.generation.llm_service import is_ollama_running, generate, OllamaUnavailableError
from rag.generation.prompt_builder import build_prompt

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

TOP_K = 3


def load_dataset():
    with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(dataset):
    in_scope_results = []
    out_of_scope_results = []

    for item in dataset:
        results = retriever.retrieve(item["question"], top_k=TOP_K)
        retrieved_sources = [r["source"] for r in results]

        if item["in_scope"]:
            hit = item["expected_source"] in retrieved_sources
            in_scope_results.append({
                "question": item["question"],
                "expected_source": item["expected_source"],
                "retrieved_sources": retrieved_sources,
                "hit": hit,
            })
        else:
            correctly_declined = len(results) == 0
            out_of_scope_results.append({
                "question": item["question"],
                "retrieved_sources": retrieved_sources,
                "correctly_declined": correctly_declined,
            })

    hit_rate = sum(r["hit"] for r in in_scope_results) / len(in_scope_results) if in_scope_results else None
    decline_rate = (
        sum(r["correctly_declined"] for r in out_of_scope_results) / len(out_of_scope_results)
        if out_of_scope_results else None
    )

    return {
        "in_scope": in_scope_results,
        "out_of_scope": out_of_scope_results,
        "retrieval_hit_rate_at_k": hit_rate,
        "correct_decline_rate": decline_rate,
        "top_k": TOP_K,
    }


def evaluate_generation(dataset):
    """Only runs if Ollama is actually reachable. Returns None otherwise
    - never fabricates a groundedness number without a real model."""
    if not is_ollama_running():
        return None

    generation_results = []
    for item in dataset:
        if not item["in_scope"]:
            continue
        chunks = retriever.retrieve(item["question"], top_k=TOP_K)
        if not chunks:
            continue
        prompt = build_prompt(item["question"], chunks)
        try:
            answer = generate(prompt)
        except OllamaUnavailableError as exc:
            generation_results.append({"question": item["question"], "error": str(exc)})
            continue

        mentions_source = item["expected_source"] in answer if item["expected_source"] else None
        generation_results.append({
            "question": item["question"],
            "answer": answer,
            "expected_source": item["expected_source"],
            "answer_mentions_expected_source": mentions_source,
        })

    return generation_results


def main():
    dataset = load_dataset()
    retrieval_report = evaluate_retrieval(dataset)

    print("=== Retrieval Evaluation (real, computed against the shipped index) ===")
    print(f"In-scope questions: {len(retrieval_report['in_scope'])}")
    print(f"Retrieval hit rate @ top-{TOP_K}: {retrieval_report['retrieval_hit_rate_at_k']:.2%}")
    print(f"Out-of-scope questions: {len(retrieval_report['out_of_scope'])}")
    print(f"Correct-decline rate: {retrieval_report['correct_decline_rate']:.2%}")
    print()
    for r in retrieval_report["in_scope"]:
        status = "HIT " if r["hit"] else "MISS"
        print(f"  [{status}] {r['question'][:55]:55s} expected={r['expected_source']} got={r['retrieved_sources']}")
    for r in retrieval_report["out_of_scope"]:
        status = "OK  " if r["correctly_declined"] else "LEAK"
        print(f"  [{status}] {r['question'][:55]:55s} retrieved={r['retrieved_sources']}")

    generation_report = evaluate_generation(dataset)
    print()
    if generation_report is None:
        print("=== Generation Evaluation: SKIPPED (Ollama is not reachable in this environment) ===")
    else:
        print("=== Generation Evaluation (real Ollama calls) ===")
        for r in generation_report:
            print(r)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval": retrieval_report,
            "generation": generation_report,
        }, f, indent=2)
    print(f"\nSaved full results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
