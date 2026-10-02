# Knowledge Base Scope

No official Panchayat scheme documents, RTI procedures, or government
policy PDFs were provided with this project. Rather than invent
plausible-sounding government rules (which risks a citizen being told a
fabricated policy as if it were real), every document in this folder
describes **this portal's own actual behavior** — verified against the
real application code (`app.py`, `complaint.sql`, and the ML phases
implemented in Phases 1-5) — not external government policy.

This means the RAG assistant can answer accurately and safely:
  "What complaint categories exist?", "How is priority decided?",
  "What does 'Under Review' mean?", "Is the resolution-time estimate a
  guarantee?", "What happens if my complaint looks like a duplicate?"

It will correctly say it doesn't know for anything outside that scope
(e.g. a specific government scheme's eligibility rules) rather than
generate a plausible-sounding but unverified answer.

When real official Panchayat documents become available, drop them into
this folder as `.md` or `.txt` files and re-run
`rag/ingestion/ingest.py` — no code changes needed.
