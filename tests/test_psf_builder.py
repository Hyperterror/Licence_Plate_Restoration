"""Tests for Step 2.1 -- PSF construction.

Covers unit checks (task 6.4) and Property 10 (task 6.3).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
    psf_is_valid,
)


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_motion_psf_zero_length() -> None:
    psf = make_motion_psf(0, 37.0)

    assert psf.shape == (1, 1)
    assert psf.dtype == np.float32
    assert psf[0, 0] == pytest.approx(1.0)


def test_defocus_psf_zero_radius() -> None:
    psf = make_defocus_psf(0)

    assert psf.shape == (1, 1)
    assert psf.dtype == np.float32
    assert psf[0, 0] == pytest.approx(1.0)


def test_motion_psf_shape_follows_length() -> None:
    for length in (1, 2, 7, 15, 21):
        psf = make_motion_psf(length, 12.0)
        assert psf.shape == (2 * length + 1, 2 * length + 1)


def test_motion_psf_direction() -> None:
    # 0 degrees -> smear along x (single row); 90 degrees -> along y (one column).
    horizontal = make_motion_psf(4, 0.0)
    vertical = make_motion_psf(4, 90.0)

    row_energy = horizontal.sum(axis=1)
    col_energy = vertical.sum(axis=0)

    assert row_energy.argmax() == 4 and row_energy.max() == pytest.approx(1.0, abs=1e-5)
    assert col_energy.argmax() == 4 and col_energy.max() == pytest.approx(1.0, abs=1e-5)
    # 45 degrees spreads over both axes
    diagonal = make_motion_psf(4, 45.0)
    assert (diagonal.sum(axis=1) > 1e-6).sum() > 3
    assert (diagonal.sum(axis=0) > 1e-6).sum() > 3


def test_motion_psf_is_anti_aliased() -> None:
    # A 4 px smear spans 5 pixels; the endpoints cover only half a pixel, so an
    # exactly area-weighted kernel has half-weight ends (not a flat 1/5 each).
    psf = make_motion_psf(2, 0.0)
    centre_row = psf[2]

    assert centre_row[0] == pytest.approx(0.125, abs=1e-3)
    assert centre_row[2] == pytest.approx(0.25, abs=1e-3)
    assert centre_row[-1] == pytest.approx(0.125, abs=1e-3)


def test_defocus_psf_is_a_disc() -> None:
    radius = 3
    psf = make_defocus_psf(radius)

    assert psf.shape == (2 * radius + 1, 2 * radius + 1)
    assert abs(psf.sum() - 1.0) < 1e-5
    # centre value is the largest; the four corners are outside the circle
    assert psf[radius, radius] == psf.max()
    assert psf[0, 0] == 0.0 and psf[0, -1] == 0.0
    assert psf[-1, 0] == 0.0 and psf[-1, -1] == 0.0
    # left/right symmetry
    assert np.allclose(psf, psf[::-1, :])
    assert np.allclose(psf, psf[:, ::-1])


def test_negative_arguments_raise() -> None:
    with pytest.raises(ValueError):
        make_motion_psf(-1, 0.0)
    with pytest.raises(ValueError):
        make_defocus_psf(-0.5)


def test_combine_psfs_keeps_normalisation() -> None:
    motion = make_motion_psf(3, 8.0)
    defocus = make_defocus_psf(2)
    combined = combine_psfs(motion, defocus)

    assert combined.shape == (7 + 5 - 1, 7 + 5 - 1)
    assert abs(combined.sum() - 1.0) < 1e-5
    assert psf_is_valid(combined)


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 10: PSF kernels are normalised
@given(length=st.integers(0, 30), angle=st.floats(0, 360))
@settings(max_examples=100, deadline=None)
def test_motion_psf_sum(length: int, angle: float) -> None:
    psf = make_motion_psf(length, angle)

    assert psf.dtype == np.float32
    assert psf.shape == (2 * length + 1, 2 * length + 1)
    assert abs(float(psf.sum()) - 1.0) < 1e-5
    assert np.all(psf >= 0.0)
    assert psf_is_valid(psf)


@given(radius=st.floats(0, 10))
@settings(max_examples=100, deadline=None)
def test_defocus_psf_sum(radius: float) -> None:
    psf = make_defocus_psf(radius)

    assert psf.dtype == np.float32
    assert psf.shape[0] % 2 == 1 and psf.shape[1] % 2 == 1
    assert abs(float(psf.sum()) - 1.0) < 1e-5
    assert np.all(psf >= 0.0)
    assert psf_is_valid(psf)


# Feature: license-plate-restoration, Property 15: combined PSF from grid loop is normalised
@given(length=st.integers(0, 21),
       angle=st.floats(0, 180),
       radius=st.integers(0, 5))
@settings(max_examples=60, deadline=None)
def test_combined_psf_normalised(length: int, angle: float, radius: int) -> None:
    combined = combine_psfs(make_motion_psf(length, angle), make_defocus_psf(radius))

    assert combined.dtype == np.float32
    assert abs(float(combined.sum()) - 1.0) < 1e-5
