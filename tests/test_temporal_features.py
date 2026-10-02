"""
Unit tests for ml/data/temporal_features.py — the leakage-prevention
logic behind the resolution-time feature pipeline. No database needed;
these are pure functions.

Run: pytest tests/test_temporal_features.py -v
"""

from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.data.temporal_features import compute_recurrence_count, resolution_hours


def _dt(days_ago):
    return datetime(2026, 1, 31) - timedelta(days=days_ago)


def test_recurrence_count_only_counts_matching_category_and_village():
    records = [
        {"category": "Water", "village": "Kadri", "created_at": _dt(5)},
        {"category": "Road", "village": "Kadri", "created_at": _dt(5)},       # wrong category
        {"category": "Water", "village": "Bejai", "created_at": _dt(5)},      # wrong village
        {"category": "Water", "village": "Kadri", "created_at": _dt(10)},
    ]
    count = compute_recurrence_count(
        records, category="Water", village="Kadri", before_timestamp=_dt(0)
    )
    assert count == 2


def test_recurrence_count_excludes_future_records_no_leakage():
    """The core leakage guard: a record created AFTER the target complaint
    must never be counted as 'recurrence' for it."""
    target_time = _dt(5)
    records = [
        {"category": "Water", "village": "Kadri", "created_at": _dt(10)},  # before -> counts
        {"category": "Water", "village": "Kadri", "created_at": _dt(2)},   # after target -> must NOT count
    ]
    count = compute_recurrence_count(
        records, category="Water", village="Kadri", before_timestamp=target_time
    )
    assert count == 1


def test_recurrence_count_respects_window():
    records = [
        {"category": "Water", "village": "Kadri", "created_at": _dt(29)},  # inside 30-day window
        {"category": "Water", "village": "Kadri", "created_at": _dt(45)},  # outside window
    ]
    count = compute_recurrence_count(
        records, category="Water", village="Kadri", before_timestamp=_dt(0), window_days=30
    )
    assert count == 1


def test_recurrence_count_handles_missing_inputs_gracefully():
    assert compute_recurrence_count([], category="", village="Kadri", before_timestamp=_dt(0)) == 0
    assert compute_recurrence_count([], category="Water", village="", before_timestamp=_dt(0)) == 0
    assert compute_recurrence_count([{"category": "Water", "village": "Kadri", "created_at": _dt(1)}],
                                     category="Water", village="Kadri", before_timestamp=None) == 0


def test_resolution_hours_normal_case():
    created = datetime(2026, 1, 1, 8, 0, 0)
    resolved = datetime(2026, 1, 3, 8, 0, 0)
    assert resolution_hours(created, resolved) == 48.0


def test_resolution_hours_rejects_invalid_ordering():
    """A 'resolved' timestamp before or equal to creation is bad data and
    must be dropped (return None), not silently clamped to 0 or negative."""
    created = datetime(2026, 1, 3, 8, 0, 0)
    resolved = datetime(2026, 1, 1, 8, 0, 0)  # before creation - invalid
    assert resolution_hours(created, resolved) is None

    same_time = datetime(2026, 1, 1, 8, 0, 0)
    assert resolution_hours(same_time, same_time) is None


def test_resolution_hours_handles_missing_timestamps():
    assert resolution_hours(None, datetime.now()) is None
    assert resolution_hours(datetime.now(), None) is None
