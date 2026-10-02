"""
Tests for ml/cv/preprocessing.py - image validation and feature
extraction. Uses small synthetic images generated on the fly (solid
colors / noise) purely to exercise the validation and feature-shape
mechanics; these are NOT used to claim anything about real-world
detection accuracy - the real, trained model's evaluated performance is
in ml/cv/evaluation/image_classifier_metrics.json (see
ml/cv/data/README.md for how that was produced on the real dataset).

Run: pytest tests/test_image_preprocessing.py -v
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from PIL import Image

from ml.cv.preprocessing import (
    InvalidImageError,
    IMAGE_SIZE,
    extract_features,
    image_bytes_to_features,
    strip_metadata_and_reencode,
    validate_and_load_image,
)


def _make_jpeg_bytes(size=(200, 150), color=(120, 120, 120)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(size=(200, 150), color=(80, 160, 40)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_webp_bytes(size=(200, 150)):
    img = Image.new("RGB", size, color=(10, 10, 10))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


# ---------------------------------------------------------------------
# validate_and_load_image
# ---------------------------------------------------------------------

def test_valid_jpeg_is_accepted():
    image = validate_and_load_image(_make_jpeg_bytes())
    assert image.mode == "RGB"


def test_valid_png_is_accepted():
    image = validate_and_load_image(_make_png_bytes())
    assert image.mode == "RGB"


def test_empty_bytes_rejected():
    with pytest.raises(InvalidImageError):
        validate_and_load_image(b"")


def test_garbage_bytes_rejected():
    with pytest.raises(InvalidImageError):
        validate_and_load_image(b"this is definitely not an image, just text bytes" * 5)


def test_truncated_jpeg_rejected():
    raw = _make_jpeg_bytes()
    truncated = raw[: len(raw) // 3]
    with pytest.raises(InvalidImageError):
        validate_and_load_image(truncated)


def test_disguised_webp_rejected_even_though_it_would_pass_as_an_extension():
    """A real finding from this phase's own dataset: 23 of 706 source
    files were WEBP data saved with a .jpg extension. The extension is
    never trusted - only the real decoded format matters."""
    with pytest.raises(InvalidImageError) as exc_info:
        validate_and_load_image(_make_webp_bytes())
    assert "unsupported_format" in str(exc_info.value)


def test_too_small_image_rejected():
    tiny = _make_jpeg_bytes(size=(8, 8))
    with pytest.raises(InvalidImageError):
        validate_and_load_image(tiny)


def test_exif_orientation_is_applied_and_stripped():
    """A rotated image should come back right-side-up (EXIF applied),
    and re-encoding must not carry the original EXIF/GPS block forward."""
    img = Image.new("RGB", (300, 200), color=(50, 50, 200))
    exif = img.getexif()
    exif[274] = 6  # Orientation: rotate 270
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    loaded = validate_and_load_image(buf.getvalue())
    # After exif_transpose for orientation 6, width/height swap (200x300)
    assert loaded.size == (200, 300)

    reencoded = strip_metadata_and_reencode(loaded)
    reencoded_image = Image.open(io.BytesIO(reencoded))
    assert reencoded_image.getexif() == {} or len(reencoded_image.getexif()) == 0


# ---------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------

def test_feature_vector_is_fixed_length_and_deterministic():
    image = validate_and_load_image(_make_jpeg_bytes(color=(90, 90, 90)))
    f1 = extract_features(image)
    f2 = extract_features(image)
    assert isinstance(f1, np.ndarray)
    assert f1.shape == f2.shape
    assert np.allclose(f1, f2)  # same image -> same features, no randomness


def test_feature_vector_shape_is_independent_of_input_image_size():
    small = validate_and_load_image(_make_jpeg_bytes(size=(64, 64)))
    large = validate_and_load_image(_make_jpeg_bytes(size=(1200, 900)))
    assert extract_features(small).shape == extract_features(large).shape


def test_different_images_give_different_features():
    gray = validate_and_load_image(_make_jpeg_bytes(color=(100, 100, 100)))
    colorful = validate_and_load_image(_make_png_bytes(color=(255, 0, 0)))
    f_gray = extract_features(gray)
    f_colorful = extract_features(colorful)
    assert not np.allclose(f_gray, f_colorful)


def test_image_bytes_to_features_matches_manual_pipeline():
    raw = _make_jpeg_bytes()
    features, image = image_bytes_to_features(raw)
    manual = extract_features(validate_and_load_image(raw))
    assert np.allclose(features, manual)
    assert image.size[0] > 0
