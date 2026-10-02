"""
Extracts a REAL resolution-time training set from the live database, if
enough history exists. This is the "use real data when it exists" half of
the pipeline - ml/data/generate_resolution_time_data.py is the synthetic
fallback for when it doesn't (a fresh install has zero resolved
complaints, so that fallback is what Phase 4 actually trains on right
now - see UPGRADE_NOTES.md).

For each complaint that has a 'Resolved' row in complaint_updates:
  - resolution_hours = time from complaints.created_at to that row's
    updated_at  (via temporal_features.resolution_hours)
  - category, priority = the complaint's OWN values (known at submission
    time - priority may have been changed later by an officer, which
    would be a leakage risk; see the note in the docstring below)
  - village = the submitting citizen's village
  - recurrence_count = same-category/same-village complaints already
    created before this one, in the preceding 30 days (leakage-safe, via
    temporal_features.compute_recurrence_count)

LEAKAGE CAVEAT (documented, not silently ignored): this schema has no
history table for priority changes, only for status changes. If an
officer edits a complaint's priority after creation, the `priority`
value read here is whatever it currently is, not what it was at
submission - a mild leakage risk. Flagged here rather than fixed, since
fixing it would need a priority-history table (schema change) which
is out of scope for this phase; noted as a limitation in UPGRADE_NOTES.md.

Run:
    python ml/data/extract_resolution_history.py
Produces:
    ml/data/resolution_real.csv  (only if MIN_ROWS_REQUIRED is met)
Exits gracefully with a clear message + non-zero-row CSV if not enough
data exists yet - this is expected on a fresh install.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import Config
from ml.data.temporal_features import compute_recurrence_count, resolution_hours

MIN_ROWS_REQUIRED = 50  # below this, a trained model would be unreliable
OUT_PATH = Path(__file__).parent / "resolution_real.csv"


def main():
    import mysql.connector

    try:
        db = mysql.connector.connect(
            host=Config.DB_HOST,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            database=Config.DB_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Could not connect to the database: {exc}")
        print("Skipping real-data extraction - the synthetic bootstrap dataset will be used instead.")
        return

    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT complaints.id, complaints.category, complaints.priority,
               complaints.created_at, users.village
        FROM complaints
        JOIN users ON complaints.user_id = users.id
        ORDER BY complaints.created_at ASC
    """)
    all_complaints = cursor.fetchall()

    cursor.execute("""
        SELECT complaint_id, updated_at
        FROM complaint_updates
        WHERE status = 'Resolved'
        ORDER BY updated_at ASC
    """)
    resolved_rows = {row["complaint_id"]: row["updated_at"] for row in cursor.fetchall()}

    cursor.close()
    db.close()

    # Records used for leakage-safe recurrence lookups (only needs
    # category/village/created_at).
    history_records = [
        {"category": c["category"], "village": c["village"], "created_at": c["created_at"]}
        for c in all_complaints
    ]

    rows = []
    for complaint in all_complaints:
        resolved_at = resolved_rows.get(complaint["id"])
        hours = resolution_hours(complaint["created_at"], resolved_at)
        if hours is None:
            continue  # not resolved yet, or bad timestamp - excluded, not imputed

        recurrence_count = compute_recurrence_count(
            history_records,
            category=complaint["category"],
            village=complaint["village"],
            before_timestamp=complaint["created_at"],
        )

        rows.append({
            "category": complaint["category"],
            "priority": complaint["priority"],
            "recurrence_count": recurrence_count,
            "resolution_hours": hours,
        })

    if len(rows) < MIN_ROWS_REQUIRED:
        print(
            f"Only {len(rows)} resolved complaints with valid timestamps found "
            f"(need at least {MIN_ROWS_REQUIRED} for a trustworthy model). "
            "This is expected on a new/low-volume install. "
            "Falling back to the documented synthetic bootstrap dataset - "
            "see ml/data/generate_resolution_time_data.py."
        )
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} REAL resolution-time training rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
