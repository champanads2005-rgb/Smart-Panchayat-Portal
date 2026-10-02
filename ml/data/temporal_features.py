"""
Pure, DB-independent temporal feature logic for resolution-time modeling.

Kept separate from any DB access so it can be unit-tested directly (see
tests/test_resolution_time.py) without a MySQL connection, and reused by
both:
  - ml/data/extract_resolution_history.py (training-time, historical data)
  - app.py's get_recent_similar_count() equivalent at inference time

LEAKAGE RULE: recurrence_count for a given complaint must only count
OTHER complaints whose created_at is strictly BEFORE this complaint's own
created_at (or "now", at inference time). Counting complaints created
after it (including complaints that arrived later but got resolved
sooner) would leak future information into a feature meant to represent
"how much of a pattern already existed when this complaint came in".
"""

from datetime import timedelta
from typing import List, Dict


def compute_recurrence_count(
    records: List[Dict],
    category: str,
    village: str,
    before_timestamp,
    window_days: int = 30,
) -> int:
    """
    records: list of dicts with at least {"category", "village", "created_at"}
             (created_at must be a datetime).
    Returns how many of those records share `category` and `village` and
    have created_at in (before_timestamp - window_days, before_timestamp).
    Strictly before `before_timestamp` - this is what prevents leakage.
    """
    if not category or not village or before_timestamp is None:
        return 0

    window_start = before_timestamp - timedelta(days=window_days)

    count = 0
    for r in records:
        if r.get("category") != category:
            continue
        if r.get("village") != village:
            continue
        created_at = r.get("created_at")
        if created_at is None:
            continue
        if window_start <= created_at < before_timestamp:
            count += 1
    return count


def resolution_hours(created_at, resolved_at) -> float:
    """Hours between submission and the complaint_updates row where
    status='Resolved'. Returns None if either timestamp is missing or the
    result would be non-positive (data-quality guard - a resolved
    timestamp before or equal to creation is invalid and should be
    dropped from training, not silently kept)."""
    if created_at is None or resolved_at is None:
        return None
    delta = (resolved_at - created_at).total_seconds() / 3600.0
    if delta <= 0:
        return None
    return delta
