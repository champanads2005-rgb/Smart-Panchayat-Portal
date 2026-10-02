"""
Tests for the resolution-time synthetic data generator (the
"preprocessing pipeline" for this phase, since there's no real historical
data yet to preprocess - see UPGRADE_NOTES.md).

Run: pytest tests/test_resolution_data_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from ml.data.generate_resolution_time_data import (
    CATEGORIES,
    PRIORITIES,
    sample_row,
)


def test_sample_row_has_expected_schema():
    row = sample_row()
    assert set(row.keys()) == {"category", "priority", "recurrence_count", "resolution_hours"}


def test_sample_row_uses_valid_categories_and_priorities():
    for _ in range(200):
        row = sample_row()
        assert row["category"] in CATEGORIES
        assert row["priority"] in PRIORITIES


def test_resolution_hours_always_positive():
    """A resolution time of zero or negative hours is not physically
    meaningful and would corrupt training - the generator floors at 1.0."""
    for _ in range(500):
        row = sample_row()
        assert row["resolution_hours"] >= 1.0


def test_recurrence_count_within_declared_bucket():
    valid_values = {0, 1, 2, 3, 5, 8, 12}
    for _ in range(200):
        row = sample_row()
        assert row["recurrence_count"] in valid_values


def test_generated_csv_is_loadable_and_well_formed(tmp_path, monkeypatch):
    """Runs the actual generator end-to-end against a temp path and checks
    the CSV it writes is well-formed - a real integration check of the
    preprocessing/generation pipeline, not just the row sampler."""
    import ml.data.generate_resolution_time_data as gen

    out_path = tmp_path / "resolution_test.csv"
    monkeypatch.setattr(gen, "OUT_PATH", out_path)

    gen.main(n_rows=50)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 50
    assert list(df.columns) == ["category", "priority", "recurrence_count", "resolution_hours"]
    assert df["resolution_hours"].min() > 0
    assert df["category"].isin(CATEGORIES).all()
    assert df["priority"].isin(PRIORITIES).all()
