"""
Tests for ml/analytics/hotspot_anomaly.py.

Run: pytest tests/test_hotspot_anomaly.py -v
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.analytics.hotspot_anomaly import (
    detect_isolationforest_anomalies,
    detect_statistical_anomalies,
    get_hotspots,
)


def _baseline_records(n_weeks=10, seed=1):
    random.seed(seed)
    start = datetime(2026, 1, 1)
    villages = ["Kadri", "Bejai", "Surathkal"]
    categories = ["Water", "Road", "Sanitation"]
    records = []
    for week in range(n_weeks):
        for v in villages:
            for c in categories:
                n = random.randint(1, 4)
                for _ in range(n):
                    day_offset = week * 7 + random.randint(0, 6)
                    records.append({"village": v, "category": c, "created_at": start + timedelta(days=day_offset)})
    return records, start


def test_get_hotspots_ranks_by_volume():
    records, start = _baseline_records()
    # Add a clear extra burst in Kadri/Water within the last 30 days.
    latest = max(r["created_at"] for r in records)
    for _ in range(20):
        records.append({"village": "Kadri", "category": "Water", "created_at": latest - timedelta(days=1)})

    hotspots = get_hotspots(records, days=30)
    assert hotspots[0]["village"] == "Kadri"
    assert hotspots[0]["category"] == "Water"
    assert hotspots[0]["count"] >= 20


def test_get_hotspots_empty_input():
    assert get_hotspots([]) == []


def test_statistical_anomaly_requires_minimum_history():
    """A group with only 2 weeks of data must NOT be flagged - there's no
    real baseline to compare against yet."""
    start = datetime(2026, 1, 1)
    records = [
        {"village": "NewWard", "category": "Water", "created_at": start},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=7)},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=8)},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=9)},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=10)},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=11)},
        {"village": "NewWard", "category": "Water", "created_at": start + timedelta(days=12)},
    ]
    anomalies = detect_statistical_anomalies(records)
    assert all(a["village"] != "NewWard" for a in anomalies)


def test_statistical_anomaly_catches_spike_that_spans_two_calendar_weeks():
    """Regression test for a real finding during development: an
    injected 85-complaint spike happened to fall across two pandas
    weekly-period boundaries (56 in one week, 30 in the next). A plain
    mean/std z-score MISSED the second week, because the first week's
    56 dragged the baseline mean/std up enough to make 30 look normal.
    Median/MAD (the modified z-score actually implemented) must catch
    it - this test would fail if the implementation regressed back to
    mean/std."""
    records, start = _baseline_records(n_weeks=10, seed=3)
    spike_week_start = start + timedelta(days=10 * 7)
    for _ in range(85):
        day_offset = random.randint(0, 6)
        records.append({
            "village": "Kadri",
            "category": "Water",
            "created_at": spike_week_start + timedelta(days=day_offset),
        })

    anomalies = detect_statistical_anomalies(records)
    kadri_water_flags = [a for a in anomalies if a["village"] == "Kadri" and a["category"] == "Water"]
    assert len(kadri_water_flags) >= 1
    assert kadri_water_flags[0]["z_score"] > 3.5


def test_statistical_anomaly_flat_history_edge_case():
    """MAD == 0 (perfectly flat history) must not raise a division error,
    and should only flag a genuine change, not zero-vs-zero."""
    start = datetime(2026, 1, 1)
    records = []
    # Exactly 2 complaints/week for 6 weeks - a perfectly flat baseline.
    for week in range(6):
        for _ in range(2):
            records.append({
                "village": "FlatWard", "category": "Road",
                "created_at": start + timedelta(days=week * 7 + 1),
            })
    anomalies = detect_statistical_anomalies(records)
    assert all(a["village"] != "FlatWard" for a in anomalies)  # last week also == 2, no change


def test_isolationforest_flags_the_same_injected_spike():
    records, start = _baseline_records(n_weeks=10, seed=3)
    spike_week_start = start + timedelta(days=10 * 7)
    for _ in range(85):
        day_offset = random.randint(0, 6)
        records.append({
            "village": "Kadri",
            "category": "Water",
            "created_at": spike_week_start + timedelta(days=day_offset),
        })

    anomalies = detect_isolationforest_anomalies(records)
    assert any(a["village"] == "Kadri" and a["category"] == "Water" for a in anomalies)


def test_isolationforest_handles_too_little_data():
    """Fewer than 10 (group, week) observations overall - too little for
    IsolationForest to be meaningful; must return empty, not crash or
    fit garbage."""
    start = datetime(2026, 1, 1)
    records = [{"village": "TinyWard", "category": "Water", "created_at": start}]
    assert detect_isolationforest_anomalies(records) == []


def test_empty_input_does_not_crash_any_function():
    assert get_hotspots([]) == []
    assert detect_statistical_anomalies([]) == []
    assert detect_isolationforest_anomalies([]) == []
