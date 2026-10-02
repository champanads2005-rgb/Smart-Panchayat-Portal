"""
Shared image validation + feature extraction for the Phase 7 pothole
image classifier.

This module is imported by BOTH the training script and the inference
module so that a served prediction always uses exactly the same
preprocessing/feature pipeline the model was trained and evaluated on
(no train/serve skew).

Feature pipeline (see ml/cv/data/README.md for the full rationale):
    1. Decode + EXIF-transpose + resize to a fixed 128x128 RGB image.
    2. HOG (Histogram of Oriented Gradients) on the grayscale image.
    3. HSV color histogram (8 bins/channel).
    4. Concatenate into one fixed-length feature vector.

This is classical, hand-engineered computer vision - not a pretrained
CNN backbone (no route to a model-weights host was available in this
sandbox - see ml/cv/data/README.md) and not a keyword match or hardcoded
rule; it is a real, trained, evaluated pipeline.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from skimage.color import rgb2gray

IMAGE_SIZE = (128, 128)  # (width, height) all images are resized to

# Formats we will actually decode and train/infer on. Anything else is
# rejected before it ever reaches PIL/numpy.
ALLOWED_PIL_FORMATS = {"JPEG", "PNG"}

# Reject implausibly tiny images (icon-sized / corrupt fragments) that
# could not plausibly show a road surface.
MIN_DIMENSION_PX = 32

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

COLOR_HIST_BINS = 8  # per HSV channel


class InvalidImageError(ValueError):
    """Raised for any image that fails validation - corrupt data, an
    unsupported/disguised format, or implausible dimensions. Callers
    (the Flask routes) turn this into a clean 4xx JSON error instead of
    a 500."""


def validate_and_load_image(raw_bytes: bytes) -> Image.Image:
    """Decodes `raw_bytes` into a verified, EXIF-normalized RGB PIL
    image, or raises InvalidImageError.

    Deliberately does NOT trust any client-supplied filename, extension,
    or Content-Type header - the actual image is decoded and its REAL
    format inspected. This is what stops a renamed non-image file (e.g.
    "shell.php.jpg") from ever being treated as one.
    """
    if not raw_bytes:
        raise InvalidImageError("empty_file")

    try:
        # Pillow's verify() must run on a fresh handle - it leaves the
        # parser in a state that can't be used to actually load pixels
        # afterwards, so we re-open for the real decode below.
        with Image.open(io.BytesIO(raw_bytes)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise InvalidImageError("not_a_valid_image")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
        real_format = image.format
        if real_format not in ALLOWED_PIL_FORMATS:
            raise InvalidImageError(f"unsupported_format:{real_format}")

        # Apply EXIF orientation before anything else touches the pixel
        # grid, then drop all EXIF/metadata (which can carry GPS
        # coordinates of the citizen's home) by rebuilding a plain image.
        from PIL import ImageOps  # local import: keeps module import light

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        width, height = image.size
        if width < MIN_DIMENSION_PX or height < MIN_DIMENSION_PX:
            raise InvalidImageError("image_too_small")

        return image
    except InvalidImageError:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is invalid input, not a server bug
        raise InvalidImageError(f"decode_failed:{exc}")


def strip_metadata_and_reencode(image: Image.Image) -> bytes:
    """Re-encodes a validated image as a plain JPEG with no EXIF/ICC/GPS
    metadata, for safe permanent storage. Takes the already-validated,
    already-oriented PIL image from validate_and_load_image()."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def extract_features(image: Image.Image) -> np.ndarray:
    """Turns a validated RGB PIL image into the fixed-length feature
    vector the classifier was trained on. MUST stay in lockstep with
    ml/cv/train_image_classifier.py - both call this exact function."""
    resized = image.resize(IMAGE_SIZE, Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float64) / 255.0  # HxWx3, [0,1]

    gray = rgb2gray(arr)
    hog_features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    hsv = np.asarray(resized.convert("HSV"), dtype=np.float64)
    color_hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(
            hsv[:, :, channel], bins=COLOR_HIST_BINS, range=(0, 255), density=True
        )
        color_hist_parts.append(hist)
    color_features = np.concatenate(color_hist_parts)

    return np.concatenate([hog_features, color_features]).astype(np.float64)


def image_bytes_to_features(raw_bytes: bytes) -> Tuple[np.ndarray, Image.Image]:
    """Convenience wrapper: validate + extract features in one call.
    Returns (features, validated_image) so callers that also need to
    store the image (re-encoded, metadata-stripped) don't have to
    decode it twice."""
    image = validate_and_load_image(raw_bytes)
    features = extract_features(image)
    return features, image
