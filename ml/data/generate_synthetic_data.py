"""
Synthetic complaint dataset generator.

IMPORTANT: This data is SYNTHETIC. It is generated from templated sentence
patterns with randomized variation (locality names, durations, intensifiers)
because no real historical Panchayat complaint corpus was provided with this
project. It is used only to bootstrap a genuine, trainable text-classification
pipeline. It must never be presented as real government data.

Categories match the existing `complaints.category` ENUM in complaint.sql:
Water, Electricity, Road, Sanitation, Street Light, Drainage, Other.

Run:
    python ml/data/generate_synthetic_data.py
Produces:
    ml/data/complaints_synthetic.csv
"""

import csv
import random
from pathlib import Path

random.seed(42)  # reproducibility

OUT_PATH = Path(__file__).parent / "complaints_synthetic.csv"

WARDS = [f"Ward {i}" for i in range(1, 11)]
DURATIONS = ["one day", "two days", "three days", "four days", "a week",
             "five days", "since yesterday", "several days"]
INTENSIFIERS = ["", "This is very urgent. ", "This is a serious problem. ",
                "Please act quickly. ", "This is affecting many families. ",
                "This needs immediate attention. "]

TEMPLATES = {
    "Water": [
        "We have not received drinking water for {duration} in {ward}.",
        "The water supply pipeline near {ward} is leaking badly.",
        "No water is coming to our taps for {duration}.",
        "The water tank in {ward} has not been filled for {duration}.",
        "Drinking water is contaminated and smells bad in {ward}.",
        "There is an irregular water supply schedule in {ward} for {duration}.",
        "The borewell motor in {ward} has stopped working since {duration}.",
    ],
    "Electricity": [
        "There has been a power cut in {ward} for {duration}.",
        "The electricity transformer near {ward} is making sparking sounds.",
        "Frequent power fluctuations are damaging appliances in {ward}.",
        "The electric pole near our house in {ward} is leaning dangerously.",
        "We have had no electricity supply for {duration} in {ward}.",
        "Exposed electrical wires are hanging near the road in {ward}.",
        "The electricity meter reading seems incorrect for {duration} now.",
    ],
    "Road": [
        "There is a huge pothole on the main road in {ward}.",
        "The road in {ward} has been damaged for {duration} after the rain.",
        "The newly laid road near {ward} is already cracking.",
        "Vehicles are getting stuck in mud on the unpaved road in {ward}.",
        "The road divider near {ward} is broken and unsafe.",
        "There are no speed breakers near the school in {ward}, causing accidents.",
        "The road near {ward} has not been repaired for {duration}.",
    ],
    "Sanitation": [
        "Garbage has not been collected in {ward} for {duration}.",
        "There is a bad smell from uncollected waste near {ward}.",
        "The public toilet in {ward} is dirty and not maintained.",
        "Stray animals are scattering garbage all over {ward}.",
        "Waste is piling up on the street corner in {ward} for {duration}.",
        "The garbage truck has not come to {ward} for {duration}.",
        "Sewage is overflowing onto the street in {ward}.",
    ],
    "Street Light": [
        "The street light near {ward} has not been working for {duration}.",
        "Several street lights are off in {ward}, making it unsafe at night.",
        "The street light pole in {ward} is damaged and flickering.",
        "It is completely dark near {ward} at night due to broken lights.",
        "New street lights are needed near the bus stop in {ward}.",
        "The street light in {ward} has been broken since {duration}.",
    ],
    "Drainage": [
        "The drainage in {ward} is blocked and water is stagnating for {duration}.",
        "Rainwater is flooding the street in {ward} due to poor drainage.",
        "The open drain near {ward} is overflowing with dirty water.",
        "Mosquitoes are breeding in the stagnant drain water in {ward}.",
        "The drainage cover is missing near {ward}, which is dangerous.",
        "Drain water has been stagnant in {ward} for {duration}.",
    ],
    "Other": [
        "There is a stray dog menace in {ward} for {duration}.",
        "We need a new community hall in {ward}.",
        "The park in {ward} has broken equipment and needs repair.",
        "There is noise pollution from a nearby function in {ward}.",
        "We request a new bus stop shelter to be built in {ward}.",
        "The panchayat office in {ward} is not open during listed hours.",
    ],
}

CATEGORIES = list(TEMPLATES.keys())


def generate_row():
    category = random.choice(CATEGORIES)
    template = random.choice(TEMPLATES[category])
    text = template.format(
        ward=random.choice(WARDS),
        duration=random.choice(DURATIONS),
    )
    intensifier = random.choice(INTENSIFIERS)
    text = (intensifier + text).strip()
    return text, category


def main(n_per_category=120):
    rows = []
    for category in CATEGORIES:
        seen = set()
        attempts = 0
        while len([r for r in rows if r[1] == category]) < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            template = random.choice(TEMPLATES[category])
            text = template.format(
                ward=random.choice(WARDS),
                duration=random.choice(DURATIONS),
            )
            intensifier = random.choice(INTENSIFIERS)
            text = (intensifier + text).strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, category))

    random.shuffle(rows)

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {OUT_PATH}")
    print("Category counts:")
    for c in CATEGORIES:
        print(f"  {c}: {sum(1 for r in rows if r[1] == c)}")


if __name__ == "__main__":
    main()
