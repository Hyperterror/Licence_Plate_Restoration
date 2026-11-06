"""Tests for Step 5 -- Otsu binarization, morphology and segmentation.

Covers tasks 10.3/10.4 (Otsu), 11.4 (morphology) and 12.2/12.3 (segmentation).

Polarity note (verified against OpenCV, which treats white as the object)::

    erode(img, k)  == 255 - cv2.erode(255 - img, k)
    dilate(img, k) == 255 - cv2.dilate(255 - img, k)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import binary_kernel_strategy, binary_uint8_image_strategy, float32_image_strategy
from hypothesis import given, settings

from license_plate_restoration.binarizer import (
    DEFAULT_POLARITY,
    binarize,
    character_boxes,
    closing,
    dilate,
    erode,
    opening,
    otsu_threshold,
    segment_characters,
    square_kernel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATES_NPZ = REPO_ROOT / "license_plate_restoration" / "data" / "plates.npz"


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Same quantisation the binarizer uses, so cv2 sees identical input."""
    return (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255).astype(np.uint8)


def _representative_plates() -> list[np.ndarray]:
    """5 plate images: 3 synthetic + 2 CCPD (skipped if the dataset is absent)."""
    from license_plate_restoration.renderer import generate_plates

    plates = [img for img, _ in generate_plates(3, seed=11)]
    if PLATES_NPZ.is_file():
        with np.load(PLATES_NPZ) as data:
            base = data["base_imgs"]
        plates.extend([base[0], base[7]])
    return plates


def _character_plate() -> tuple[np.ndarray, list[int]]:
    """Binary plate with a border plus 7 well-separated character blobs."""
    plate = np.full((92, 256), 255, dtype=np.uint8)
    plate[:2, :] = 0
    plate[-2:, :] = 0
    plate[:, :2] = 0
    plate[:, -2:] = 0

    xs = [20, 55, 90, 125, 160, 195, 230]
    for x in xs:
        plate[25:75, x:x + 18] = 0           # 50 x 18 -> aspect 2.8, area 900
    return plate, xs


# --------------------------------------------------------------------------- #
# Otsu
# --------------------------------------------------------------------------- #
def test_otsu_matches_cv2_on_plates() -> None:
    for plate in _representative_plates():
        ours = otsu_threshold(plate) * 255.0
        reference, _ = cv2.threshold(_to_uint8(plate), 0, 255, cv2.THRESH_OTSU)
        assert abs(ours - reference) <= 1.0, (ours, reference)


def test_otsu_range() -> None:
    for plate in _representative_plates():
        assert 0.0 <= otsu_threshold(plate) <= 1.0


def test_binarize_gives_black_text_on_white() -> None:
    from license_plate_restoration.renderer import generate_plates

    plate, text = generate_plates(1, seed=21)[0]
    binary = binarize(plate)

    assert binary.dtype == np.uint8
    assert set(np.unique(binary)) <= {0, 255}

    # Glyph cores of the grayscale plate are black; they must stay ink.
    ink = binary == 0
    dark = plate <= plate.min() + 1e-6
    assert ink[dark].mean() > 0.9

    # The white plate interior away from the glyphs must be background.
    assert binary[12, 180] == 255
    assert binary[84, 180] == 255

    # Otsu splits two classes, so the mid-grey margin joins the dark class and
    # forms a ring.  That ring is the largest component and gets discarded, so
    # the character count is unaffected by it.
    assert ink[:8, :].all()
    crops = segment_characters(binary)
    assert 4 <= len(crops) <= 12, len(crops)


def test_binarize_polarity_modes() -> None:
    # A dark plate with bright characters, like a CCPD photograph.
    image = np.full((92, 256), 0.15, dtype=np.float32)
    image[30:70, 20:50] = 0.9                # bright text covering a minority

    # literal requirement rule: the lower class mean is the ink
    literal = binarize(image, polarity="dark")
    assert literal[50, 35] == 255             # bright text -> background
    assert literal[50, 200] == 0              # dark field -> ink

    # default: the minority class of the interior is the ink
    auto = binarize(image)
    assert auto[50, 35] == 0                  # bright text -> ink
    assert auto[50, 200] == 255               # dark field -> background

    with pytest.raises(ValueError):
        binarize(image, polarity="sideways")


def test_binarize_constant_image_has_no_ink() -> None:
    binary = binarize(np.full((20, 20), 0.5, dtype=np.float32))
    assert np.all(binary == 255)


def test_default_polarity_is_auto() -> None:
    """Team decision: ``auto`` is frozen as the default for every caller."""
    assert DEFAULT_POLARITY == "auto"
    assert inspect.signature(binarize).parameters["polarity"].default == DEFAULT_POLARITY


def test_literal_rule_inverts_ccpd_plates() -> None:
    """Requirement 8.3's rule empties a CCPD plate; ``auto`` recovers the text.

    These are the numbers behind the team decision: on ten ccpd_base plates the
    literal "lower class mean = ink" rule keeps at most two character-sized
    components, because the dark field becomes the ink, while ``auto`` keeps
    seven or eight.
    """
    if not PLATES_NPZ.is_file():
        pytest.skip("plates.npz has not been built")

    with np.load(PLATES_NPZ) as data:
        plates = data["base_imgs"][:10]

    auto = [len(segment_characters(binarize(plate))) for plate in plates]
    literal = [len(segment_characters(binarize(plate, polarity="dark"))) for plate in plates]

    assert min(auto) >= 6, auto
    assert max(literal) <= 2, literal


# --------------------------------------------------------------------------- #
# Morphology
# --------------------------------------------------------------------------- #
def test_opening_removes_speck() -> None:
    image = np.full((31, 31), 255, dtype=np.uint8)
    image[10:20, 10:20] = 0                  # 10 x 10 blob
    image[27, 3] = 0                         # single-pixel speck
    kernel = square_kernel(3)

    opened = opening(image, kernel)
    assert opened[10:20, 10:20].max() == 0 or (opened[10:20, 10:20] == 0).any()
    assert opened[27, 3] == 255              # speck removed


def test_closing_joins_broken_stroke() -> None:
    image = np.full((31, 31), 255, dtype=np.uint8)
    image[10:20, 10:13] = 0
    image[10:20, 15:18] = 0                  # 1 px vertical gap at column 13-14
    kernel = square_kernel(3)

    closed = closing(image, kernel)
    assert np.all(closed[10:20, 10:18] == 0)


def test_morphology_matches_opencv_duality() -> None:
    rng = np.random.default_rng(0)
    kernel = square_kernel(3)
    for _ in range(10):
        image = np.where(rng.random((15, 19)) < 0.5, 0, 255).astype(np.uint8)
        inverted = 255 - image
        assert np.array_equal(erode(image, kernel), 255 - cv2.erode(inverted, kernel))
        assert np.array_equal(dilate(image, kernel), 255 - cv2.dilate(inverted, kernel))


def test_morphology_rejects_empty_kernel() -> None:
    with pytest.raises(ValueError):
        erode(np.zeros((5, 5), np.uint8), np.zeros((3, 3), np.uint8))


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def test_segment_characters_sorted_order() -> None:
    plate, xs = _character_plate()
    boxes = character_boxes(plate)
    crops = segment_characters(plate)

    assert len(boxes) == len(xs)
    assert [box[0] for box in boxes] == xs          # left to right
    assert len(crops) == len(xs)
    for crop, (x, _y, w, _h) in zip(crops, boxes):
        assert crop.dtype == np.uint8
        assert crop.shape == (50, w)
        assert np.all(crop == 0)


def test_segment_characters_rejects_background() -> None:
    plate, _ = _character_plate()
    ink = (plate == 0).astype(np.uint8)
    _count, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, 4]
    largest = 1 + int(np.argmax(areas))
    largest_box = tuple(int(v) for v in stats[largest, :4])

    boxes = character_boxes(plate)
    assert largest_box not in boxes                  # the frame is excluded
    assert all(box != largest_box for box in boxes)


def test_segment_characters_crops_are_tight() -> None:
    plate, _ = _character_plate()
    for crop in segment_characters(plate):
        # A tight bounding box has ink on every one of its four edges.
        assert (crop[0] == 0).any() and (crop[-1] == 0).any()
        assert (crop[:, 0] == 0).any() and (crop[:, -1] == 0).any()
        assert crop.shape == (50, 18)


def test_segment_characters_on_blank_plate() -> None:
    assert segment_characters(np.full((92, 256), 255, dtype=np.uint8)) == []


def test_segment_characters_tolerates_single_blob() -> None:
    plate = np.full((92, 256), 255, dtype=np.uint8)
    plate[10:80, 10:240] = 0                         # one huge component only
    assert segment_characters(plate) == []           # largest is rejected


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 16: otsu_threshold accuracy vs. OpenCV reference
@given(img=float32_image_strategy())
@settings(max_examples=100, deadline=None)
def test_otsu_accuracy_vs_cv2(img: np.ndarray) -> None:
    ours = otsu_threshold(img) * 255.0
    reference, _ = cv2.threshold(_to_uint8(img), 0, 255, cv2.THRESH_OTSU)
    assert abs(ours - reference) <= 1.0


# Feature: license-plate-restoration, Property 17: binarize output invariants
@given(img=float32_image_strategy())
@settings(max_examples=60, deadline=None)
def test_binarize_output_values(img: np.ndarray) -> None:
    binary = binarize(img)

    assert binary.dtype == np.uint8
    assert binary.shape == img.shape
    assert set(np.unique(binary)) <= {0, 255}


# Feature: license-plate-restoration, Property 18: erode and dilate match OpenCV reference
@given(img=binary_uint8_image_strategy(), kernel=binary_kernel_strategy())
@settings(max_examples=100, deadline=None)
def test_erode_matches_cv2(img: np.ndarray, kernel: np.ndarray) -> None:
    kernel = kernel.astype(np.uint8)
    inverted = 255 - img
    assert np.array_equal(erode(img, kernel), 255 - cv2.erode(inverted, kernel))


@given(img=binary_uint8_image_strategy(), kernel=binary_kernel_strategy())
@settings(max_examples=100, deadline=None)
def test_dilate_matches_cv2(img: np.ndarray, kernel: np.ndarray) -> None:
    kernel = kernel.astype(np.uint8)
    inverted = 255 - img
    assert np.array_equal(dilate(img, kernel), 255 - cv2.dilate(inverted, kernel))


# Feature: license-plate-restoration, Property 19: opening and closing composition
@given(img=binary_uint8_image_strategy(), kernel=binary_kernel_strategy())
@settings(max_examples=60, deadline=None)
def test_opening_composition(img: np.ndarray, kernel: np.ndarray) -> None:
    assert np.array_equal(opening(img, kernel), dilate(erode(img, kernel), kernel))


@given(img=binary_uint8_image_strategy(), kernel=binary_kernel_strategy())
@settings(max_examples=60, deadline=None)
def test_closing_composition(img: np.ndarray, kernel: np.ndarray) -> None:
    assert np.array_equal(closing(img, kernel), erode(dilate(img, kernel), kernel))


# Feature: license-plate-restoration, Property 20: segment_characters left-to-right ordering and no exception
@given(plate=binary_uint8_image_strategy(shape=(92, 256)))
@settings(max_examples=30, deadline=None)
def test_segment_characters_no_exception_sorted(plate: np.ndarray) -> None:
    boxes = character_boxes(plate)
    crops = segment_characters(plate)

    assert len(crops) == len(boxes)
    xs = [box[0] for box in boxes]
    assert xs == sorted(xs), "crops must be ordered left to right"
    for crop, (x, y, w, h) in zip(crops, boxes):
        assert crop.dtype == np.uint8
        assert crop.shape == (h, w)
        assert np.array_equal(crop, plate[y:y + h, x:x + w])
