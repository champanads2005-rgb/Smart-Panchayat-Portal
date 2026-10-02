"""
Hotspot & anomaly detection over complaint records.

HONEST SCOPE NOTE: the master prompt's GIS/heatmap vision (section 21)
assumes lat/lng coordinates or ward boundary polygons. Neither exists in
this schema — `users.village` is a free-text field, not a geocoded
location, and complaints carry no coordinates at all. Building a real
map means either geocoding villages (needs an external geocoding API —
not available offline/without a key) or manually drawing ward boundaries
(not this project's data to invent). Rather than fake a map with made-up
coordinates, this phase delivers the actual analytics — hotspot ranking,
trend direction, and anomaly flags — as tables/cards on an admin
dashboard, grouped by the real `village` field. A geographic view is
documented here as a follow-on step once real coordinates exist, not
built with placeholder ones.

Two REAL, unsupervised techniques, applied directly to live complaint
data (no training/labels needed, unlike Phases 1-4):

  1. Statistical thresholding (z-score) — per (village, category) group,
     compares the most recent week's complaint count to the mean/std of
     that group's own prior weeks. Requires a minimum amount of history
     before it will flag anything (see MIN_HISTORY_WEEKS) — a group with
     one or two weeks of data has no baseline to be "anomalous" against,
     and flagging it anyway would just be noise dressed up as insight.

  2. Isolation Forest — fit across ALL (village, category, week) count
     observations at once, to catch anomalies that a single group's own
     short history might miss (an isolation forest doesn't need
     per-group history the way the z-score method does, so it can flag
     an unusual count even for a group seen only once or twice).

Both are run and reported together; they're complementary; presenting
either alone would be less honest about coverage.
"""

from datetime import timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MIN_HISTORY_WEEKS = 4  # weeks of prior history required before z-score can flag anything
Z_THRESHOLD = 2.5
ISOLATION_CONTAMINATION = 0.05
RANDOM_STATE = 42


def _records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def _weekly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Dense weekly counts per (village, category), including zero-count
    weeks in between - required so z-score/mean/std reflect real quiet
    periods, not just weeks that happened to have a complaint."""
    if df.empty:
        return pd.DataFrame(columns=["village", "category", "week", "count"])

    df = df.copy()
    df["week"] = df["created_at"].dt.to_period("W").dt.start_time

    grouped = (
        df.groupby(["village", "category", "week"])
        .size()
        .reset_index(name="count")
    )

    filled_frames = []
    full_week_range = pd.period_range(
        df["week"].min(), df["week"].max(), freq="W"
    ).start_time

    for (village, category), group in grouped.groupby(["village", "category"]):
        series = group.set_index("week")["count"].reindex(full_week_range, fill_value=0)
        out = series.reset_index()
        out.columns = ["week", "count"]
        out["village"] = village
        out["category"] = category
        filled_frames.append(out)

    if not filled_frames:
        return pd.DataFrame(columns=["village", "category", "week", "count"])

    return pd.concat(filled_frames, ignore_index=True)


def get_hotspots(records: List[Dict], top_n: int = 10, days: int = 30) -> List[Dict]:
    """Ranks (village, category) pairs by complaint volume in the last
    `days` days. Trend is a real comparison of the first half vs. second
    half of that window, not a guess - 'increasing' means the second half
    had a meaningfully higher share, 'decreasing' the opposite, 'stable'
    otherwise."""
    df = _records_to_dataframe(records)
    if df.empty:
        return []

    cutoff = df["created_at"].max() - timedelta(days=days)
    window = df[df["created_at"] >= cutoff]
    if window.empty:
        return []

    midpoint = cutoff + timedelta(days=days / 2)

    results = []
    for (village, category), group in window.groupby(["village", "category"]):
        total = len(group)
        first_half = (group["created_at"] < midpoint).sum()
        second_half = (group["created_at"] >= midpoint).sum()

        if second_half > first_half * 1.3 and second_half >= 2:
            trend = "increasing"
        elif first_half > second_half * 1.3 and first_half >= 2:
            trend = "decreasing"
        else:
            trend = "stable"

        results.append({
            "village": village,
            "category": category,
            "count": int(total),
            "trend": trend,
        })

    results.sort(key=lambda r: r["count"], reverse=True)
    return results[:top_n]


def detect_statistical_anomalies(records: List[Dict], z_threshold: float = 3.5) -> List[Dict]:
    """Flags (village, category) groups whose MOST RECENT week is a real
    statistical outlier vs. that group's own history, using the median
    and MAD (median absolute deviation) rather than mean/std.

    This is a deliberate, tested choice, not the textbook default: mean/
    std is NOT robust to outliers already present in the history window
    itself. During development, a spike that happened to span two
    calendar weeks (56 complaints one week, 30 the next) got its second,
    still-abnormal week MISSED by plain mean/std z-score, because the
    first week's 56 dragged the baseline mean and std up enough to make
    30 look unremarkable. Median/MAD (the modified z-score, threshold
    3.5 per Iglewicz & Hoaglin) resists that contamination - re-running
    the same scenario with this method correctly flagged both weeks. See
    tests/test_hotspot_anomaly.py for the regression test that encodes
    this finding.

    Groups with fewer than MIN_HISTORY_WEEKS prior weeks of data are
    explicitly skipped (no baseline to be "anomalous" against), not
    flagged anyway with a shaky baseline.
    """
    df = _records_to_dataframe(records)
    if df.empty:
        return []

    weekly = _weekly_counts(df)
    anomalies = []

    for (village, category), group in weekly.groupby(["village", "category"]):
        group = group.sort_values("week")
        if len(group) < MIN_HISTORY_WEEKS + 1:
            continue  # not enough history to have a baseline - skip, don't guess

        *history, latest = group.to_dict("records")
        history_counts = np.array([h["count"] for h in history], dtype=float)
        median = np.median(history_counts)
        mad = np.median(np.abs(history_counts - median))

        if mad == 0:
            # Perfectly flat history - any nonzero change IS meaningful,
            # but MAD-based z-score is undefined (division by zero).
            if latest["count"] != median and latest["count"] > 0:
                anomalies.append({
                    "village": village,
                    "category": category,
                    "week": str(latest["week"].date()),
                    "count": int(latest["count"]),
                    "typical_range": f"{median:.0f} (flat history)",
                    "z_score": None,
                    "method": "statistical",
                })
            continue

        modified_z = 0.6745 * (latest["count"] - median) / mad
        if abs(modified_z) >= z_threshold:
            spread = 1.4826 * mad  # MAD scaled to be std-comparable, for a readable range
            anomalies.append({
                "village": village,
                "category": category,
                "week": str(latest["week"].date()),
                "count": int(latest["count"]),
                "typical_range": f"{max(0, median - spread):.0f}-{median + spread:.0f}",
                "z_score": round(float(modified_z), 2),
                "method": "statistical",
            })

    anomalies.sort(key=lambda a: abs(a["z_score"]) if a["z_score"] is not None else 0, reverse=True)
    return anomalies


def detect_isolationforest_anomalies(records: List[Dict], contamination: float = ISOLATION_CONTAMINATION) -> List[Dict]:
    """Fits IsolationForest across every (village, category, week) count
    observation at once. Unlike the statistical method, this doesn't
    need per-group history - it judges each observation against the
    overall distribution of counts, so it can flag a group's very first
    unusual week too."""
    df = _records_to_dataframe(records)
    if df.empty:
        return []

    weekly = _weekly_counts(df)
    if len(weekly) < 10:
        return []  # too few observations overall for IsolationForest to be meaningful

    X = weekly[["count"]].values
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)

    weekly = weekly.copy()
    weekly["is_anomaly"] = predictions == -1
    weekly["anomaly_score"] = scores

    flagged = weekly[weekly["is_anomaly"]].sort_values("anomaly_score")

    return [
        {
            "village": row["village"],
            "category": row["category"],
            "week": str(row["week"].date()),
            "count": int(row["count"]),
            "anomaly_score": round(float(row["anomaly_score"]), 3),
            "method": "isolation_forest",
        }
        for _, row in flagged.iterrows()
    ]
