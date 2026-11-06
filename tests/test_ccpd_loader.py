"""Tests for Step 1.2 -- CCPD plate loading.

Covers the unit checks (task 3.4) and properties 5-7 (task 3.3).  The ground
truth in ``test_ccpd_parse_known_filename`` was verified by eye against the
warped image (the plate reads 皖A 65L62 -> "A65L62").
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.ccpd_loader import (
    ADS,
    ALPHABETS,
    OUT_H,
    OUT_W,
    load_ccpd_plate,
    parse_ccpd_filename,
    sample_ccpd,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CCPD_BASE = REPO_ROOT / "CCPD2019" / "ccpd_base"
CCPD_BLUR = REPO_ROOT / "CCPD2019" / "ccpd_blur"

KNOWN_NAME = ("00245450191571-90_90-276&480_370&520-364&524_274&519_275&486_365&491"
              "-0_0_30_29_10_30_26-80-16.jpg")
KNOWN_TEXT = "A65L62"

requires_ccpd = pytest.mark.skipif(
    not CCPD_BASE.is_dir(), reason="CCPD2019/ccpd_base is not available"
)

# A handful of real files, used by the file-based property tests.
_SAMPLE_FILES = sorted(CCPD_BASE.glob("*.jpg"))[:40] if CCPD_BASE.is_dir() else []
ccpd_path_strategy = st.sampled_from([str(p) for p in _SAMPLE_FILES] or ["missing.jpg"])


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_ccpd_parse_known_filename() -> None:
    parsed = parse_ccpd_filename(KNOWN_NAME)
    assert parsed is not None
    corners, text = parsed

    assert text == KNOWN_TEXT
    assert corners.shape == (4, 2)
    assert corners.dtype == np.float32
    # bottom-right corner of the plate (first vertex in CCPD order)
    assert tuple(corners[0]) == (364.0, 524.0)


@pytest.mark.parametrize(
    "name",
    [
        "not-a-ccpd-name.jpg",
        "00245450191571-90_90-276&480_370&520-364&524-274&519_275&486_365&491"
        "-0_0_30_29_10_30_26-80-16.jpg",          # only 3 vertex pairs
        "00245450191571-90_90-276&480_370&520-364&524_274&519_275&486_365&491"
        "-0_0_30_29_10_30-80-16.jpg",             # only 6 char indices
        "00245450191571-90_90-276&480_370&520-364&524_274&519_275&486_365&491"
        "-0_99_30_29_10_30_26-80-16.jpg",         # alphabet index out of range
        "00245450191571-90_90-a&b_c&d_e&f_g&h-0_0_30_29_10_30_26-80-16.jpg",  # not numeric
    ],
)
def test_ccpd_parse_rejects_malformed(name: str) -> None:
    assert parse_ccpd_filename(name) is None


@requires_ccpd
def test_load_ccpd_plate_invariants_on_known_file() -> None:
    loaded = load_ccpd_plate(str(CCPD_BASE / KNOWN_NAME))
    assert loaded is not None
    image, text = loaded

    assert image.shape == (OUT_H, OUT_W) == (92, 256)
    assert image.dtype == np.float32
    assert 0.0 <= image.min() and image.max() <= 1.0
    assert text == KNOWN_TEXT

    # 8 px grey margin on all four sides, same value as the synthetic plates.
    expected = np.float32(128 / 255.0)
    assert np.allclose(image[:8, :], expected, atol=1e-6)
    assert np.allclose(image[-8:, :], expected, atol=1e-6)
    assert np.allclose(image[:, :8], expected, atol=1e-6)
    assert np.allclose(image[:, -8:], expected, atol=1e-6)


def test_load_ccpd_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_ccpd_plate(str(tmp_path / KNOWN_NAME)) is None
    assert load_ccpd_plate(str(tmp_path / "garbage.jpg")) is None


@requires_ccpd
def test_text_decoding_uses_ccpd_tables() -> None:
    # Index 24 of ALPHABETS is "O" and index 34 of ADS is "O": they must map.
    assert ALPHABETS[0] == "A" and ALPHABETS[-1] == "O" and len(ALPHABETS) == 25
    assert ADS[:24] == list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    assert ADS[24] == "0" and ADS[33] == "9" and ADS[34] == "O" and len(ADS) == 35


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 5: load_ccpd_plate output invariants
@requires_ccpd
@given(path=ccpd_path_strategy)
@settings(max_examples=20, deadline=None)
def test_load_ccpd_plate_invariants(path: str) -> None:
    loaded = load_ccpd_plate(path)
    if loaded is None:                        # allowed by the contract
        return
    image, text = loaded

    assert image.shape == (92, 256)
    assert image.dtype == np.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert len(text) == 6
    assert all(ch in ALPHABETS or ch in ADS for ch in text)


# Feature: license-plate-restoration, Property 6: sample_ccpd returns exactly n plates
@requires_ccpd
@given(n=st.integers(1, 8), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=5, deadline=None)
def test_sample_ccpd_returns_n(n: int, seed: int) -> None:
    images, texts, names = sample_ccpd(str(CCPD_BASE), n, seed)

    assert images.shape == (n, 92, 256)
    assert images.dtype == np.float32
    assert texts.shape == (n,)
    assert len(names) == n
    assert all(len(t) == 6 for t in texts)


@requires_ccpd
def test_sample_ccpd_is_deterministic() -> None:
    _, first, _ = sample_ccpd(str(CCPD_BASE), 4, seed=11)
    _, second, _ = sample_ccpd(str(CCPD_BASE), 4, seed=11)
    assert list(first) == list(second)


# Feature: license-plate-restoration, Property 7: malformed CCPD filenames are silently skipped
@requires_ccpd
def test_sample_ccpd_skips_malformed(tmp_path: Path) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()

    valid = _SAMPLE_FILES[:3]
    for path in valid:
        shutil.copy(path, folder / path.name)
    (folder / "broken.jpg").write_bytes(b"not an image")
    (folder / "a-b-c.jpg").write_bytes(b"not an image")

    images, texts, names = sample_ccpd(str(folder), 3, seed=0)
    assert images.shape == (3, 92, 256)
    assert len(names) == 3
    assert set(names) == {p.name for p in valid}

    # Asking for more than the folder holds must not raise either.
    fewer, _, _ = sample_ccpd(str(folder), 10, seed=0)
    assert fewer.shape[0] == 3


def test_sample_ccpd_empty_folder(tmp_path: Path) -> None:
    images, texts, names = sample_ccpd(str(tmp_path), 5, seed=0)
    assert images.shape == (0, 92, 256)
    assert texts.shape == (0,)
    assert names == []
