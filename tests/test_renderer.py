"""Tests for Step 1.1 -- synthetic Indian plate rendering.

Covers the unit checks (task 2.4) and the four property tests (task 2.3).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import invalid_plate_text_strategy, valid_plate_text_strategy
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.renderer import (
    GREY,
    MARGIN,
    OUT_H,
    OUT_W,
    PLATE_H,
    PLATE_W,
    STATE_CODES,
    generate_plates,
    render_plate,
)

TEXT_RE = re.compile(r"^[A-Z]{2} [0-9]{2} [A-Z]{2} [0-9]{4}$")

# Rendering is fast but not free; give Hypothesis room without a deadline.
SETTINGS = settings(max_examples=100, deadline=None)


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_render_plate_shape_dtype_range() -> None:
    plate = render_plate("KA 05 MN 1234", seed=7)

    assert plate.shape == (OUT_H, OUT_W) == (96, 336)
    assert plate.dtype == np.float32
    assert plate.min() >= 0.0 and plate.max() <= 1.0


def test_render_plate_white_background() -> None:
    plate = render_plate("KA 05 MN 1234", seed=0)

    # 8 px grey margin on every side.
    assert np.allclose(plate[:MARGIN, :], GREY)
    assert np.allclose(plate[-MARGIN:, :], GREY)
    assert np.allclose(plate[:, :MARGIN], GREY)
    assert np.allclose(plate[:, -MARGIN:], GREY)

    # 2 px black frame just inside the white plate area.
    top, left = MARGIN, MARGIN
    assert np.all(plate[top:top + 2, left:left + PLATE_W] == 0.0)
    assert np.all(plate[top:top + PLATE_H, left:left + 2] == 0.0)

    # White plate background: a strip just inside the frame, above the glyphs.
    inner_white = plate[top + 4, left + 4:left + PLATE_W - 4]
    assert np.all(inner_white == 1.0)


def test_render_plate_text_pixels_present() -> None:
    plate = render_plate("KA 05 MN 1234", seed=0)
    inner = plate[MARGIN:MARGIN + PLATE_H, MARGIN:MARGIN + PLATE_W]

    # Glyph cores are pure black and must exist inside the plate.
    assert (inner == 0.0).sum() > 200
    # Anti-aliased glyph edges give intermediate values (not a hard mask).
    edge = (inner > 0.0) & (inner < 1.0)
    assert edge.sum() > 0


def test_render_plate_invalid_raises() -> None:
    with pytest.raises(ValueError, match="state code"):
        render_plate("XX 00 AA 1234", seed=0)   # format fine, state does not exist

    with pytest.raises(ValueError, match="expected the format"):
        render_plate("KA-05-MN-1234", seed=0)


def test_generate_plates_returns_pairs() -> None:
    plates = generate_plates(600, seed=42)

    assert len(plates) == 600
    texts = [text for _, text in plates]
    assert len(set(texts)) == 600, "generated texts must be unique"
    assert all(img.shape == (OUT_H, OUT_W) and img.dtype == np.float32
               for img, _ in plates)


def test_generate_plates_is_deterministic() -> None:
    first = generate_plates(5, seed=3)
    second = generate_plates(5, seed=3)

    assert [t for _, t in first] == [t for _, t in second]
    for (img_a, _), (img_b, _) in zip(first, second):
        assert np.array_equal(img_a, img_b)


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 1: render_plate output shape and dtype
@given(text=valid_plate_text_strategy(), seed=st.integers(0, 2**31 - 1))
@SETTINGS
def test_render_plate_shape_and_dtype(text: str, seed: int) -> None:
    plate = render_plate(text, seed)
    assert plate.shape == (96, 336)
    assert plate.dtype == np.float32


# Feature: license-plate-restoration, Property 2: render_plate determinism
@given(text=valid_plate_text_strategy(), seed=st.integers(0, 2**31 - 1))
@SETTINGS
def test_render_plate_deterministic(text: str, seed: int) -> None:
    assert np.array_equal(render_plate(text, seed), render_plate(text, seed))


# Feature: license-plate-restoration, Property 3: generate_plates text format
@given(n=st.integers(1, 50), seed=st.integers(0, 2**31 - 1))
@SETTINGS
def test_generate_plates_text_format(n: int, seed: int) -> None:
    for _, text in generate_plates(n, seed):
        assert TEXT_RE.match(text), text
        assert text[:2] in STATE_CODES


# Feature: license-plate-restoration, Property 4: render_plate rejects invalid text
@given(text=invalid_plate_text_strategy())
@SETTINGS
def test_render_plate_raises_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        render_plate(text, 0)
