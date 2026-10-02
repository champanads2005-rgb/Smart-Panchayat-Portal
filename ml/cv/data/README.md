# Phase 7 — Image dataset documentation

## What this model actually detects

**One class of civic issue: road potholes**, as a binary
"Pothole" vs "Plain / no-pothole road surface" whole-image classifier.

It does **not** detect garbage accumulation, generic road damage other than
potholes, broken street lights, or drainage problems. The master prompt for
this phase asked for all five, but this project was built in a sandboxed
environment with no route to Kaggle, Google Drive, or Hugging Face Hub, and
no labeled dataset of adequate size and quality for the other four classes
could be obtained here (see "What was tried and why it didn't work" below).
Rather than fabricate classes the model was never trained or evaluated on —
which the master prompt explicitly forbids — this phase scopes the real,
working feature to the one class for which a genuine, if imperfect, dataset
was actually obtainable, and wires the rest of the pipeline (upload,
security, confirmation workflow, API) to support additional classes the
moment a suitable dataset exists. `ml/inference/image_classify.py`'s
`SUPPORTED_CLASSES` is the single source of truth read by both the API and
the UI, so the "supported classes" list shown to citizens can never drift
from what the model actually knows.

## Source dataset

- **Origin**: a public GitHub repository,
  `Somashekarbm/PotholeDetection` ("Pothole Detection System — Real-time
  Image Classification"), folder `My Dataset/{train,test}/{Plain,Pothole}`.
- **Provenance, stated by the dataset's own author**: the images were
  *web-scraped from Google Images*, not captured by a controlled
  camera/survey protocol. The author's own README states this makes the
  set "highly inconsistent" and recommends a proper dataset for real
  research use. This project repeats that caveat rather than hiding it.
- **No explicit open-source license file** ships with that repository (no
  `LICENSE`). Because of that, **the raw images are not redistributed
  inside this project** — they are not committed to this repo and not
  shipped in `ml/cv/data/`. Reproducing training requires cloning the
  source repository yourself (see "How to reproduce" below); this
  project ships only the *code*, the *manifest metadata* (file hashes,
  labels, split assignment — no pixel data), and the *evaluation
  results* it produced.
- **Size after dedup**: 714 files as shipped by the source repo, 706
  unique files after exact-duplicate removal by SHA-256 content hash (8
  byte-for-byte duplicates were found and dropped — see
  `dataset_prep.py`). Class balance: 348 "Plain", 358 "Pothole" after
  dedup (close to balanced, no resampling needed).
- **Image quality**: sizes range from 160×120 to 5760×3840 (the author
  mixed phone photos and downloaded web images of very different
  resolutions). All 706 files opened and verified successfully with
  Pillow — no corrupt files.
- **A real finding from actually running the validator, not assumed
  away**: 23 of the 706 files have a `.jpg` extension but decode as
  **WEBP**, not JPEG (an artifact of how they were originally
  downloaded/renamed during scraping). `ml/cv/preprocessing.py`
  validates the image's real decoded format, not its extension or
  claimed Content-Type — exactly the "file type validation" the master
  prompt requires for uploads (see requirement 10) — so these 23 files
  are correctly rejected and excluded from training rather than
  silently trusted. This is why the training run reports 545/68/70
  train/val/test images used, not the full 706.

## What was tried and why the other four classes aren't supported

- **Garbage/sanitation**: TrashNet (`garythung/trashnet`) is a real,
  commonly-used dataset for waste material classification, but its
  image archive is stored via Git LFS and downloaded from Hugging Face /
  a GitHub LFS media host — both outside this sandbox's allowed network
  domains (confirmed by a direct connection test: `huggingface.co` and
  `download.pytorch.org` both return `host_not_allowed`). It also
  classifies material type (glass/paper/cardboard/plastic/metal/trash)
  photographed on a posterboard, not "garbage accumulated on a street",
  so it wouldn't have been a good class match even if reachable.
- **Broken street lights / drainage**: no dataset of usable size and
  clear provenance was found that didn't require Kaggle, Google Drive,
  or Hugging Face Hub access, all unreachable here.
- **A pretrained ImageNet backbone for transfer learning** (the
  preferred approach per the master prompt) was also not usable:
  `torchvision`/`timm` fetch pretrained weights from
  `download.pytorch.org` at first use, which this sandbox's egress proxy
  blocks (`x-deny-reason: host_not_allowed`, verified directly). This is
  the same category of environment constraint the project's own Phase 3
  (no route to a sentence-embedding model hub) and Phase 6 (no reachable
  Ollama) already document — this phase follows the same policy: state
  the gap, ship a real, weaker, fully-offline alternative instead of
  faking the ideal one.

## Model family used instead

Because a pretrained CNN backbone could not be downloaded, this phase
uses **classical, hand-engineered computer-vision features + a shallow
classifier** — the same "real, weaker, fully offline" trade-off the
existing codebase already made for duplicate detection (TF-IDF instead
of Sentence-BERT) and RAG retrieval (TF-IDF instead of a hosted
embedding model). Concretely (`ml/cv/preprocessing.py`):

1. Decode, EXIF-transpose, and resize to 128×128 RGB.
2. **HOG** (Histogram of Oriented Gradients, `skimage.feature.hog`) on
   the grayscale image — captures edge/texture structure, which is the
   dominant visual signal for a pothole (a dark, irregular depression
   with a distinct edge boundary against surrounding pavement texture).
3. **HSV color histogram** (8 bins/channel) — captures the duller,
   often wet/shadowed coloring inside a pothole versus uniform pavement.
4. Concatenate → a fixed-length feature vector → scikit-learn classifier.

This is a real, trained, evaluated pipeline, not a keyword match or a
hardcoded rule. It is also a materially weaker model than a fine-tuned
CNN would be — this is stated plainly in "Limitations" below, not
papered over.

## How to reproduce

```bash
# 1. Fetch the raw source dataset (not shipped in this repo - see above)
git clone --depth 1 https://github.com/Somashekarbm/PotholeDetection.git /tmp/pothole_src

# 2. Build the manifest: dedup by content hash, then a fresh stratified
#    80/10/10 train/val/test split with a fixed seed (the source repo's
#    own train/test split is too small on the test side - 18 images -
#    to evaluate meaningfully, so this project does its own split).
python ml/cv/dataset_prep.py --raw-dir "/tmp/pothole_src/My Dataset"

# 3. Train + evaluate (compares multiple classical models, saves the best)
python ml/cv/train_image_classifier.py

# Results land in ml/cv/evaluation/image_classifier_metrics.{json,txt}
# and the trained model in ml/cv/models/pothole_classifier.joblib
```

## Limitations (stated plainly, not hidden)

- Single class (pothole vs plain road) — see above.
- Source images are web-scraped, not from a controlled acquisition
  protocol; the author's own documentation calls the set "inconsistent".
- Classical HOG+color features are a real but weaker signal than a
  fine-tuned CNN; expect more false positives/negatives on unusual
  lighting, wet roads, shadows, or non-road objects than a deep model
  would give, especially on images very unlike this dataset's style
  (phone snapshots and web-downloaded stock/news photos).
- No object localization — this is whole-image classification, not
  bounding-box detection, so it cannot say *where* in the photo a
  pothole is or how many there are. Because of this, **mAP does not
  apply** (mAP measures localization quality against ground-truth boxes,
  which this pipeline has none of); the evaluation instead reports
  accuracy, precision/recall/F1 per class, macro-F1, and ROC-AUC — the
  metrics that actually apply to a binary image classifier.
- The dataset is small (706 images). Metrics are real and were computed
  on a genuine held-out split, but should not be read as a claim of
  production-grade robustness.
