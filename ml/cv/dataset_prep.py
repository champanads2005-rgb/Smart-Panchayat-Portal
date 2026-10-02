"""
Builds ml/cv/data/manifest.csv from the externally-cloned raw pothole
dataset (see ml/cv/data/README.md for why the raw images themselves are
not shipped in this repo).

What this script does, and why:
    1. Walks {raw-dir}/train/{Plain,Pothole} and {raw-dir}/test/{Plain,Pothole}
       (the source repo's own folder layout).
    2. Hashes every file's content (SHA-256) and drops exact duplicates -
       706 unique files after dedup as of this writing (8 duplicates
       found in the 714-file source, see README.md).
    3. Ignores the source repo's own train/test split (its test side is
       only 18 images - too small to evaluate anything meaningfully) and
       does its OWN stratified 80/10/10 train/val/test split with a
       fixed seed, so results are reproducible.
    4. Writes ml/cv/data/manifest.csv with columns:
       relative_path, label, split, sha256
       (relative_path is relative to --raw-dir - no pixel data is stored
       in this repo, only this metadata).

Run:
    python ml/cv/dataset_prep.py --raw-dir "/path/to/My Dataset"
"""

import argparse
import csv
import hashlib
import random
from pathlib import Path

SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
LABELS = ["Plain", "Pothole"]
MANIFEST_PATH = Path(__file__).parent / "data" / "manifest.csv"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_source_files(raw_dir: Path):
    """Source repo layout: {raw_dir}/{train,test}/{Plain,Pothole}/*.jpg
    We ignore the train/test grouping and just collect every file per
    label - our own split is computed fresh below."""
    files_by_label = {label: [] for label in LABELS}
    for source_split in ("train", "test"):
        for label in LABELS:
            folder = raw_dir / source_split / label
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    files_by_label[label].append(path)
    return files_by_label


def build_manifest(raw_dir: Path) -> dict:
    files_by_label = _collect_source_files(raw_dir)
    total_found = sum(len(v) for v in files_by_label.values())
    if total_found == 0:
        raise FileNotFoundError(
            f"No images found under {raw_dir}/{{train,test}}/{{Plain,Pothole}}. "
            "Did you pass the right --raw-dir (the 'My Dataset' folder)?"
        )

    seen_hashes = {}
    duplicates_dropped = 0
    rows = []

    rng = random.Random(SEED)

    for label, paths in files_by_label.items():
        unique_paths = []
        for path in paths:
            digest = _hash_file(path)
            if digest in seen_hashes:
                duplicates_dropped += 1
                continue
            seen_hashes[digest] = str(path)
            unique_paths.append((path, digest))

        rng.shuffle(unique_paths)
        n = len(unique_paths)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])
        # remainder goes to test, so rounding never drops a sample
        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )

        for (path, digest), split in zip(unique_paths, splits):
            rows.append(
                {
                    "relative_path": str(path.relative_to(raw_dir)),
                    "label": label,
                    "split": split,
                    "sha256": digest,
                }
            )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "label", "split", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_found": total_found,
        "duplicates_dropped": duplicates_dropped,
        "unique_total": len(rows),
        "per_label": {label: len(v) for label, v in files_by_label.items()},
        "per_split": {
            split: sum(1 for r in rows if r["split"] == split) for split in SPLIT_RATIOS
        },
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Path to the cloned source dataset's 'My Dataset' folder "
        "(containing train/ and test/ subfolders).",
    )
    args = parser.parse_args()

    result = build_manifest(Path(args.raw_dir).expanduser().resolve())

    print(f"Found {result['total_found']} source files.")
    print(f"Dropped {result['duplicates_dropped']} exact-duplicate files (by content hash).")
    print(f"Wrote {result['unique_total']} unique rows to {MANIFEST_PATH}")
    print(f"Per-label counts: {result['per_label']}")
    print(f"Per-split counts: {result['per_split']}")
