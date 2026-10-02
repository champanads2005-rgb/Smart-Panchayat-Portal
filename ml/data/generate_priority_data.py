"""
Synthetic dataset for PRIORITY prediction (Low / Medium / High).

IMPORTANT: SYNTHETIC data, same caveat as ml/data/generate_synthetic_data.py.
Reuses the same complaint text templates so the priority model sees
realistic complaint phrasing, but the LABEL here is generated from a
combination of:
  - a text severity score (urgency/emergency language present in the text)
  - the complaint category's baseline risk (e.g. Water/Drainage tend to be
    more urgent than "Other")
  - a recurrence_count feature (how many similar complaints already exist
    in that ward/category recently)
  - random noise

This mirrors the real workflow in app.py, where recurrence_count is
computed for real from the complaints table (see get_recent_similar_count
in app.py) and passed alongside the text at inference time — only the
LABEL used to train on is synthetic, not the feature computation itself.

We deliberately do NOT reduce this to a single if/else rule: the label is
a noisy combination of three signals bucketed into three classes, and the
model has to learn the combined pattern from TF-IDF text + category +
recurrence_count as inputs - it is not told the severity score directly.

Priority is kept to 3 levels (Low/Medium/High) to match the existing
`complaints.priority` ENUM in complaint.sql, rather than introducing a
4th "Critical" level that would require a schema migration.

Run:
    python ml/data/generate_priority_data.py
Produces:
    ml/data/priority_synthetic.csv
"""

import csv
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_synthetic_data import TEMPLATES, WARDS, DURATIONS  # noqa: E402

random.seed(7)

OUT_PATH = Path(__file__).parent / "priority_synthetic.csv"

CATEGORY_BASELINE_RISK = {
    "Water": 0.65,
    "Drainage": 0.55,
    "Electricity": 0.55,
    "Sanitation": 0.45,
    "Road": 0.40,
    "Street Light": 0.30,
    "Other": 0.20,
}

URGENT_PHRASES = [
    "This is very urgent.", "This is a serious problem.", "Please act quickly.",
    "This is affecting many families.", "This needs immediate attention.",
]

CATEGORIES = list(TEMPLATES.keys())


def make_text_and_severity():
    category = random.choice(CATEGORIES)
    template = random.choice(TEMPLATES[category])
    text = template.format(ward=random.choice(WARDS), duration=random.choice(DURATIONS))

    severity = 0.0
    n_urgent = 0
    for _ in range(random.choice([0, 0, 1, 1, 2])):
        phrase = random.choice(URGENT_PHRASES)
        if phrase not in text:
            text = phrase + " " + text
            n_urgent += 1
    severity = min(1.0, n_urgent * 0.35)

    return text, category, severity


def bucket_priorities_by_quantile(scores):
    """Buckets by quantile so classes stay evaluable (real complaint data is
    imbalanced too, but 900 rows split roughly 45/35/20 keeps enough samples
    per class for a meaningful train/test evaluation)."""
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    low_cut = sorted_scores[int(n * 0.45)]
    medium_cut = sorted_scores[int(n * 0.80)]

    labels = []
    for s in scores:
        if s <= low_cut:
            labels.append("Low")
        elif s <= medium_cut:
            labels.append("Medium")
        else:
            labels.append("High")
    return labels


def main(n_rows=900):
    raw = []
    for _ in range(n_rows):
        text, category, severity = make_text_and_severity()
        recurrence_count = random.choices(
            [0, 1, 2, 3, 5, 8, 12], weights=[30, 25, 20, 10, 8, 4, 3]
        )[0]
        recurrence_component = min(1.0, recurrence_count / 10.0)

        base = CATEGORY_BASELINE_RISK[category]
        noise = random.gauss(0, 0.12)

        score = 0.45 * severity + 0.30 * base + 0.25 * recurrence_component + noise
        score = max(0.0, min(1.0, score))

        raw.append((text, category, recurrence_count, score))

    scores = [r[3] for r in raw]
    priorities = bucket_priorities_by_quantile(scores)
    rows = [(text, category, recurrence_count, priority)
            for (text, category, recurrence_count, _), priority in zip(raw, priorities)]

    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category", "recurrence_count", "priority"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {OUT_PATH}")
    print("Priority counts:")
    for p in ["Low", "Medium", "High"]:
        print(f"  {p}: {sum(1 for r in rows if r[3] == p)}")


if __name__ == "__main__":
    main()
