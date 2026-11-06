"""Shared Hypothesis strategies for the license-plate restoration test suite.

Only the strategies needed by the current step are defined here; the image and
kernel strategies get appended as steps 2 and 5 are implemented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

# Make ``license_plate_restoration`` importable when pytest is run from the
# repository root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from license_plate_restoration.renderer import (  # noqa: E402
    PLATE_PATTERN,
    SERIAL_LETTERS,
    STATE_CODES,
)


# --------------------------------------------------------------------------- #
# Plate-text strategies
# --------------------------------------------------------------------------- #
@st.composite
def valid_plate_text_strategy(draw: st.DrawFn) -> str:
    """Any string accepted by ``render_plate``: real state code + full format."""
    state = draw(st.sampled_from(STATE_CODES))
    district = draw(st.integers(min_value=0, max_value=99))
    series = draw(st.text(alphabet=SERIAL_LETTERS, min_size=2, max_size=2))
    serial = draw(st.integers(min_value=0, max_value=9999))
    return f"{state} {district:02d} {series} {serial:04d}"


def _looks_renderable(text: str) -> bool:
    match = PLATE_PATTERN.match(text)
    return match is not None and match.group(1) in STATE_CODES


_INVALID_EXAMPLES = [
    "",
    "KA 05 MN 123",            # too few digits
    "KA 05 MN 12345",          # too many digits
    "ka 05 mn 1234",           # lower case
    "KA-05-MN-1234",           # wrong separator
    "XX 00 AA 1234",           # format ok, state code does not exist
    "KA 005 MN 1234",          # district is 3 digits
    "KA 05 M 1234",            # one series letter
    "KAA 05 MN 1234",          # state code is 3 chars
    "KA 05 MN 12a4",           # alpha inside serial
    " KA 05 MN 1234",          # leading space
    "KA 05 MN 1234 ",          # trailing space
]


@st.composite
def invalid_plate_text_strategy(draw: st.DrawFn) -> str:
    """Strings that ``render_plate`` must reject with ``ValueError``."""
    return draw(
        st.one_of(
            st.sampled_from(_INVALID_EXAMPLES),
            st.text(min_size=0, max_size=24).filter(lambda t: not _looks_renderable(t)),
        )
    )


# --------------------------------------------------------------------------- #
# Image / kernel strategies
# --------------------------------------------------------------------------- #
@st.composite
def float32_image_strategy(
    draw: st.DrawFn,
    min_h: int = 4,
    min_w: int = 4,
    max_h: int = 32,
    max_w: int = 32,
    even: bool = False,
) -> np.ndarray:
    """Random float32 image in [0, 1]; ``even`` forces both sides even."""
    h = draw(st.integers(min_value=min_h, max_value=max_h))
    w = draw(st.integers(min_value=min_w, max_value=max_w))
    if even:
        h -= h % 2
        w -= w % 2
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    image = np.random.default_rng(seed).random((h, w), dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


@st.composite
def float32_image_batch_strategy(draw: st.DrawFn, max_n: int = 4) -> np.ndarray:
    """Small batch of float32 images, all the same shape."""
    n = draw(st.integers(min_value=1, max_value=max_n))
    h = draw(st.integers(min_value=4, max_value=16))
    w = draw(st.integers(min_value=4, max_value=16))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    batch = np.random.default_rng(seed).random((n, h, w), dtype=np.float32)
    return np.clip(batch, 0.0, 1.0)


@st.composite
def binary_uint8_image_strategy(
    draw: st.DrawFn,
    shape: tuple[int, int] | None = None,
    max_h: int = 24,
    max_w: int = 24,
    foreground: float = 0.5,
) -> np.ndarray:
    """Binary uint8 image with values in {0, 255} (0 = foreground)."""
    if shape is None:
        shape = (draw(st.integers(min_value=3, max_value=max_h)),
                 draw(st.integers(min_value=3, max_value=max_w)))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    mask = np.random.default_rng(seed).random(shape) < foreground
    return np.where(mask, np.uint8(0), np.uint8(255))


@st.composite
def binary_kernel_strategy(draw: st.DrawFn, max_size: int = 5) -> np.ndarray:
    """Odd-sized binary uint8 structuring element with values in {0, 1}."""
    h = draw(st.integers(min_value=1, max_value=(max_size + 1) // 2)) * 2 - 1
    w = draw(st.integers(min_value=1, max_value=(max_size + 1) // 2)) * 2 - 1
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    mask = np.random.default_rng(seed).random((h, w)) < 0.75
    mask[h // 2, w // 2] = True                     # always include the anchor
    return mask.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """Per-test scratch directory for artifacts."""
    return tmp_path
