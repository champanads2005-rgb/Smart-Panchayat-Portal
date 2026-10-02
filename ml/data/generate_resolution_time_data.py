"""
Synthetic bootstrap dataset for RESOLUTION-TIME regression.

WHY SYNTHETIC: ml/data/extract_resolution_history.py is the real
extraction path, but a freshly-installed Smart Panchayat Portal has zero
resolved complaints with real timestamps (it may not even have had the
complaint_updates table until this phase's schema patch - see
scripts/schema_patches.sql). There is nothing real to train on yet. This
generator exists so the modeling pipeline can be built, tested, and
demonstrated now, and swapped for ml/data/resolution_real.csv the moment
enough real history exists (train_resolution.py checks for that file
first automatically).

DOCUMENTED ASSUMPTIONS (all invented for bootstrap purposes only, not
measured from any real Panchayat data - do not quote these as real
performance figures):
  - Baseline resolution time by category, in hours, reflecting typical
    civil-works complexity (e.g. water/drainage infrastructure work
    plausibly takes longer than a street-light bulb swap).
  - Higher priority complaints resolve FASTER on average (assumption:
    officers triage and act on High-priority complaints sooner). This is
    an assumption, not an observed fact - flagged in UPGRADE_NOTES.md.
  - Higher recurrence_count (more similar recent complaints in the same
    ward) resolves marginally FASTER (assumption: a visible pattern of
    repeated complaints in one area draws more attention/pressure to fix
    it, e.g. a burst pipe reported by 10 households vs. 1).
  - Resolution time is log-normally distributed (always positive,
    right-skewed - a few complaints take much longer than the rest),
    which is why we sample noise in log-space rather than adding
    symmetric Gaussian noise directly to hours.

Run:
    python ml/data/generate_resolution_time_data.py
Produces:
    ml/data/resolution_synthetic.csv
"""

import csv
import math
import random
from pathlib import Path

random.seed(21)

OUT_PATH = Path(__file__).parent / "resolution_synthetic.csv"

CATEGORIES = ["Water", "Electricity", "Road", "Sanitation", "Street Light", "Drainage", "Other"]
PRIORITIES = ["Low", "Medium", "High"]

# Baseline MEDIAN resolution time in hours, by category (documented assumption).
CATEGORY_BASE_HOURS = {
    "Water": 60,
    "Drainage": 72,
    "Road": 120,
    "Electricity": 36,
    "Sanitation": 30,
    "Street Light": 24,
    "Other": 48,
}

# Multiplier applied to the baseline for each priority (documented assumption:
# High priority -> resolved faster -> multiplier < 1).
PRIORITY_MULTIPLIER = {
    "High": 0.55,
    "Medium": 1.0,
    "Low": 1.35,
}


def sample_row():
    category = random.choice(CATEGORIES)
    priority = random.choices(PRIORITIES, weights=[0.2, 0.5, 0.3])[0]
    recurrence_count = random.choices(
        [0, 1, 2, 3, 5, 8, 12], weights=[30, 25, 20, 10, 8, 4, 3]
    )[0]

    median_hours = CATEGORY_BASE_HOURS[category] * PRIORITY_MULTIPLIER[priority]
    # Recurrence gives a mild speed-up, capped so it can't dominate.
    recurrence_speedup = max(0.6, 1.0 - 0.03 * recurrence_count)
    median_hours *= recurrence_speedup

    # Log-normal noise: sigma controls spread. This is what makes the
    # target genuinely hard to predict exactly - not a deterministic
    # function of the inputs.
    sigma = 0.5
    noise = random.gauss(0, sigma)
    resolution_hours = median_hours * math.exp(noise)
    resolution_hours = max(1.0, resolution_hours)  # floor at 1 hour

    return {
        "category": category,
        "priority": priority,
        "recurrence_count": recurrence_count,
        "resolution_hours": round(resolution_hours, 1),
    }


def main(n_rows=1000):
    rows = [sample_row() for _ in range(n_rows)]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "priority", "recurrence_count", "resolution_hours"])
        writer.writeheader()
        writer.writerows(rows)

    hours = [r["resolution_hours"] for r in rows]
    print(f"Wrote {len(rows)} synthetic rows to {OUT_PATH}")
    print(f"resolution_hours: min={min(hours):.1f} median={sorted(hours)[len(hours)//2]:.1f} max={max(hours):.1f}")


if __name__ == "__main__":
    main()
