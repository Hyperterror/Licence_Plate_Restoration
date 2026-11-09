"""Tests for Step 4.1 -- the restoration foundation.

Covers validation, the PSF frequency response, edge tapering and the shared
numerical helpers.  Everything runs on small synthetic arrays: the foundation
is resolution agnostic, so a 9x13 array and a 96x336 plate exercise exactly the
same code paths.  No file from ``data/degraded/`` is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import float32_image_strategy
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.degradation import blur_fft, effective_psf
from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)
from license_plate_restoration.restoration import (
    DEFAULT_EPS,
    DEFAULT_TAPER_FRACTION,
    clip_unit_interval,
    edge_taper_window,
    near_zero_mask,
    psf_frequency_response,
    safe_divide,
    taper_edges,
    validate_image,
    validate_image_shape,
    validate_psf,
)

DELTA = np.array([[1.0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #
def _circular_convolve(image, response: np.ndarray) -> np.ndarray:
    """Apply a frequency response the way the restoration stages will."""
    spectrum = np.fft.fft2(np.asarray(image, dtype=np.float64))
    return np.real(np.fft.ifft2(spectrum * response))


def _spatial_convolve_centred(image, psf) -> np.ndarray:
    """Centred 2-D convolution, valid region only -- the ground truth.

    The full linear convolution ``y[p, q] = sum_ij h[i, j] x[p - i, q - j]`` is
    accumulated first; the centred result of an unshifted kernel is then the
    window starting at the kernel centre, ``y[c + m, c + n]``.
    """
    kernel = np.asarray(psf, dtype=np.float64)
    source = np.asarray(image, dtype=np.float64)
    rows, cols = source.shape
    kernel_rows, kernel_cols = kernel.shape
    centre_r, centre_c = kernel_rows // 2, kernel_cols // 2

    full = np.zeros((rows + kernel_rows - 1, cols + kernel_cols - 1))
    for i in range(kernel_rows):
        for j in range(kernel_cols):
            full[i:i + rows, j:j + cols] += kernel[i, j] * source

    return full[centre_r:centre_r + rows - kernel_rows + 1,
                centre_c:centre_c + cols - kernel_cols + 1]


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity centroid ``(row, col)``; only meaningful for positive images."""
    rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    total = float(image.sum())
    return float((image * rows).sum() / total), float((image * cols).sum() / total)


def _random_psf(seed: int, size: int = 5) -> np.ndarray:
    """Random odd-sized, non-negative, normalised kernel."""
    raw = np.random.default_rng(seed).random((size, size))
    return (raw / raw.sum()).astype(np.float32)


# --------------------------------------------------------------------------- #
# Image validation
# --------------------------------------------------------------------------- #
def test_validate_image_returns_float32_copy() -> None:
    image = np.random.default_rng(0).random((7, 11))
    checked = validate_image(image)

    assert checked.shape == image.shape
    assert checked.dtype == np.float32
    assert np.allclose(checked, image, atol=1e-7)
    assert checked is not image


def test_validate_image_rejects_wrong_dimensionality() -> None:
    with pytest.raises(ValueError, match="2-D grayscale"):
        validate_image(np.zeros((4, 4, 3)))
    with pytest.raises(ValueError, match="2-D grayscale"):
        validate_image(np.zeros(9))


def test_validate_image_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_image(np.zeros((0, 5)))


def test_validate_image_rejects_non_finite_values() -> None:
    broken = np.zeros((4, 4), dtype=np.float32)
    broken[1, 2] = np.nan
    broken[3, 3] = np.inf

    with pytest.raises(ValueError, match="2 non-finite"):
        validate_image(broken)


def test_validate_image_rejects_values_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_image(np.full((3, 3), 1.5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_image(np.full((3, 3), -0.25))


def test_validate_image_tolerates_float32_round_trip() -> None:
    # 1.0000001 as float32 is exactly 1.0, but a float64 near-miss must pass.
    image = np.full((3, 3), 1.0 + 5e-5, dtype=np.float64)
    assert validate_image(image).shape == (3, 3)


def test_validate_image_shape_accepts_pairs_and_rejects_junk() -> None:
    assert validate_image_shape((92, 256)) == (92, 256)
    assert validate_image_shape([4, 5]) == (4, 5)

    for bad in ((5,), (5, 5, 5), 7, "ab"):
        with pytest.raises(ValueError):
            validate_image_shape(bad)
    with pytest.raises(ValueError, match=">= 1"):
        validate_image_shape((0, 5))
    with pytest.raises(ValueError, match="whole number"):
        validate_image_shape((5, 6.5))


# --------------------------------------------------------------------------- #
# PSF validation
# --------------------------------------------------------------------------- #
def test_validate_psf_accepts_kernels_built_by_step_2() -> None:
    for psf in (DELTA, make_motion_psf(3, 12.0), make_defocus_psf(2.5),
                combine_psfs(make_motion_psf(4, 30.0), make_defocus_psf(2))):
        checked = validate_psf(psf)
        assert checked.dtype == np.float32
        assert checked.shape == psf.shape
        assert abs(float(checked.sum()) - 1.0) < 1e-5


def test_validate_psf_does_not_renormalise() -> None:
    """Stored kernels must be used as-is, so validation may not rescale."""
    psf = make_motion_psf(3, 7.0)
    assert np.array_equal(validate_psf(psf), psf)

    # Still inside the 1e-5 tolerance -> accepted, and still untouched.
    nudged = psf.copy()
    nudged[3, 3] += 4e-6
    assert float(validate_psf(nudged).sum(dtype=np.float64)) == pytest.approx(
        float(nudged.sum(dtype=np.float64)), abs=0.0)


@pytest.mark.parametrize(
    "kernel, message",
    [
        (np.ones((4, 4), dtype=np.float32) / 16.0, "odd-sized"),
        (np.ones((3, 4), dtype=np.float32) / 12.0, "odd-sized"),
        (np.ones((3, 3, 3), dtype=np.float32) / 9.0, "2-D kernel"),
        (np.zeros((0, 0), dtype=np.float32), "empty"),
        (-0.1 * np.ones((3, 3), dtype=np.float32), "negative weight"),
        (np.full((3, 3), 0.5, dtype=np.float32), "must sum to 1"),
        (np.full((3, 3), 1.0 / 9.0, dtype=np.float32) * 0.0, "must sum to 1"),
    ],
)
def test_validate_psf_rejects_malformed_kernels(kernel: np.ndarray,
                                                message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_psf(kernel)


def test_validate_psf_rejects_non_finite_weights() -> None:
    kernel = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
    kernel[0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        validate_psf(kernel)


def test_validate_psf_rejects_ragged_input() -> None:
    with pytest.raises(ValueError, match="numeric array"):
        validate_psf([[1.0, 0.0], [0.0], [0.0, 1.0]])


# --------------------------------------------------------------------------- #
# Frequency response
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4, 4), (9, 13), (17, 31), (46, 168),
                                   (48, 168), (92, 256), (96, 336)])
def test_frequency_response_has_the_image_shape(shape: tuple[int, int]) -> None:
    psf = make_defocus_psf(1.0)
    response = psf_frequency_response(psf, shape)

    assert response.shape == shape
    assert np.iscomplexobj(response)
    assert response.dtype == np.complex128
    assert np.all(np.isfinite(response))


def test_delta_psf_response_is_unity() -> None:
    """A delta kernel is the identity filter, so H = 1 at every frequency."""
    response = psf_frequency_response(DELTA, (9, 13))

    assert np.allclose(response, 1.0 + 0.0j, atol=1e-12)


def test_delta_psf_leaves_the_image_unchanged() -> None:
    """In the response convention the delta PSF must be an exact identity."""
    image = np.random.default_rng(0).random((12, 20), dtype=np.float32)
    response = psf_frequency_response(DELTA, image.shape)

    restored = _circular_convolve(image, response)
    assert np.max(np.abs(restored - image)) <= 1e-9


def test_response_dc_gain_is_one_for_a_normalised_psf() -> None:
    psf = _random_psf(seed=1, size=7)
    response = psf_frequency_response(psf, (16, 24))

    # DC gain == sum(h) == 1: the mean brightness of the image is preserved.
    # 1e-6 leaves room for the float32 normalisation of the kernel.
    assert response[0, 0] == pytest.approx(1.0, abs=1e-6)
    # A non-negative kernel summing to 1 can never amplify any frequency.
    assert np.abs(response).max() <= 1.0 + 1e-9


def test_centred_psf_does_not_introduce_a_spatial_shift() -> None:
    image = np.zeros((41, 41), dtype=np.float32)
    image[20, 20] = 1.0                   # impulse exactly on the centre
    psf = make_defocus_psf(3.0)           # 7x7 symmetric disc

    response = psf_frequency_response(psf, image.shape)
    blurred = _circular_convolve(image, response)
    row, col = _centroid(blurred)

    assert row == pytest.approx(20.0, abs=1e-9)
    assert col == pytest.approx(20.0, abs=1e-9)

    # Sanity check on the check: skipping the origin shift really does move the
    # content by half a kernel, which is the artefact the roll removes.
    naive = np.zeros_like(image)
    naive[:psf.shape[0], :psf.shape[1]] = psf
    shifted = _circular_convolve(image, np.fft.fft2(naive))
    shifted_row, shifted_col = _centroid(shifted)
    assert shifted_row == pytest.approx(20.0 + psf.shape[0] // 2, abs=0.1)
    assert shifted_col == pytest.approx(20.0 + psf.shape[1] // 2, abs=0.1)


def test_response_matches_direct_spatial_convolution() -> None:
    """The response implements a centred convolution, not a correlation."""
    image = np.zeros((32, 40), dtype=np.float32)
    image[8:-8, 8:-8] = np.random.default_rng(2).random((16, 24), dtype=np.float32)
    psf = make_defocus_psf(3.0)

    circular = _circular_convolve(image, psf_frequency_response(psf, image.shape))
    direct = _spatial_convolve_centred(image, psf)
    margin = psf.shape[0] // 2

    # Both arrays are indexed by pixel: drop the rim where the circular copy
    # wraps, and stay inside the valid-convolution window of the direct sum.
    shared = (slice(margin, direct.shape[0]), slice(margin, direct.shape[1]))
    assert np.allclose(circular[shared], direct[shared], atol=1e-12)


def test_response_agrees_with_blur_fft_on_a_zero_bordered_plate() -> None:
    """Same convention as step 2.2: circular FFT == its reflect-padded blur."""
    plate = np.zeros((96, 336), dtype=np.float32)
    plate[8:-8, 8:-8] = np.random.default_rng(3).random(
        (80, 320), dtype=np.float32)
    psf = make_motion_psf(3, 8.0)

    circular = _circular_convolve(plate, psf_frequency_response(psf, plate.shape))
    reference = blur_fft(plate, psf)

    assert np.max(np.abs(circular - reference)) <= 1e-6


def test_both_dataset_resolutions_work_with_matching_kernels() -> None:
    """Full-resolution and factor-2 inputs, each with its own kernel width."""
    full_kernel = make_motion_psf(15, 8.0)                  # 31x31
    half_kernel = effective_psf(full_kernel, 2)             # 15x15

    full = psf_frequency_response(full_kernel, (96, 336))
    half = psf_frequency_response(half_kernel, (48, 168))
    ccpd = psf_frequency_response(
        combine_psfs(make_motion_psf(11, 20.0), make_defocus_psf(3)), (92, 256))

    for response, shape in ((full, (96, 336)), (half, (48, 168)),
                            (ccpd, (92, 256))):
        assert response.shape == shape
        assert response[0, 0].real == pytest.approx(1.0, abs=1e-6)
        assert np.all(np.isfinite(response))


def test_response_rejects_a_kernel_larger_than_the_image() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        psf_frequency_response(make_motion_psf(15, 0.0), (20, 20))


def test_response_rejects_an_invalid_psf() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        psf_frequency_response(np.full((5, 5), 0.08), (32, 32))
    with pytest.raises(ValueError, match="negative weight"):
        psf_frequency_response(-np.ones((5, 5)) / 25.0, (32, 32))
    with pytest.raises(ValueError, match="whole number"):
        psf_frequency_response(DELTA, (32, 32.5))
    with pytest.raises(ValueError, match="2-D kernel"):
        psf_frequency_response(np.ones((3, 3, 3)) / 27.0, (32, 32))


def test_response_accepts_stored_factor_two_kernels() -> None:
    """Stored factor-2 kernels are even-sized and carry FFT round-off lobes.

    ``run_degradation_grid`` stores ``effective_psf(combined, 2)``, whose size
    is ``(L + r, L + r)`` -- even for about half of the grid -- with ~1e-17
    negative weights.  Step 4 must consume those as-is instead of rejecting
    half of the low-resolution dataset.
    """
    stored = effective_psf(combine_psfs(make_motion_psf(15, 8.0),
                                        make_defocus_psf(3)), 2)

    assert stored.shape == (18, 18)                  # even side, by construction
    assert float(stored.min()) < 0.0                 # round-off, not a real lobe

    response = psf_frequency_response(stored, (48, 168))
    assert response.shape == (48, 168)
    assert response[0, 0].real == pytest.approx(1.0, abs=1e-6)

    # A genuinely negative kernel is still refused.
    with pytest.raises(ValueError, match="negative weight"):
        psf_frequency_response(-np.ones((18, 18)) / 324.0, (48, 168))


@pytest.mark.parametrize("kernel, message", [
    (np.ones((4, 4), dtype=np.float32) / 16.0, "odd-sized"),
    (np.ones((3, 4), dtype=np.float32) / 12.0, "odd-sized"),
])
def test_response_can_enforce_the_odd_kernel_convention(kernel: np.ndarray,
                                                        message: str) -> None:
    with pytest.raises(ValueError, match=message):
        psf_frequency_response(kernel, (32, 32), require_odd=True)


# --------------------------------------------------------------------------- #
# Edge tapering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(2, 2), (4, 5), (9, 13), (46, 168),
                                   (92, 256), (96, 336), (1, 17), (17, 1)])
def test_taper_preserves_shape_and_finiteness(shape: tuple[int, int]) -> None:
    image = np.random.default_rng(4).random(shape, dtype=np.float32)
    tapered = taper_edges(image)

    assert tapered.shape == shape
    assert tapered.dtype == np.float32
    assert np.all(np.isfinite(tapered))
    assert tapered.min() >= 0.0 and tapered.max() <= 1.0


def test_taper_window_is_unit_on_the_plateau_and_zero_on_the_border() -> None:
    width = 6
    window = edge_taper_window((40, 64), width=width)

    assert window.dtype == np.float32
    assert window.shape == (40, 64)
    assert window[0, 0] == 0.0 and window[-1, -1] == 0.0
    assert window[0, 32] == 0.0 and window[39, 32] == 0.0
    assert np.all(window[width:-width, width:-width] == 1.0)
    assert window.max() == 1.0
    # Monotone ramp: moving away from the border never decreases the weight.
    assert np.all(np.diff(window[:width, 32]) >= 0.0)


def test_taper_does_not_alter_the_central_plate_region() -> None:
    image = np.random.default_rng(5).random((96, 336), dtype=np.float32)
    width = int(round(DEFAULT_TAPER_FRACTION * min(image.shape)))
    tapered = taper_edges(image)

    assert width == 10
    assert np.array_equal(tapered[width:-width, width:-width],
                          image[width:-width, width:-width])


def test_taper_attenuates_only_the_border() -> None:
    image = np.full((92, 256), 0.5, dtype=np.float32)
    tapered = taper_edges(image)

    assert float(tapered.mean()) < float(image.mean())
    assert tapered[46, 128] == pytest.approx(0.5)        # centre untouched
    assert tapered[0, 0] == 0.0
    border = np.concatenate([tapered[0], tapered[-1],
                             tapered[:, 0], tapered[:, -1]])
    assert float(border.mean()) < 0.25 * float(image.mean())


def test_taper_width_is_configurable_in_pixels() -> None:
    shape = (46, 168)
    narrow = edge_taper_window(shape, width=2)
    wide = edge_taper_window(shape, width=8)

    # A wider ramp attenuates more of the image but keeps the same plateau rule.
    assert float(wide.sum()) < float(narrow.sum())
    assert np.all(wide[8:-8, 8:-8] == 1.0)
    assert np.all(narrow[2:-2, 2:-2] == 1.0)


def test_taper_fraction_scales_with_the_shorter_side() -> None:
    tall = edge_taper_window((96, 336), fraction=0.1)
    short = edge_taper_window((46, 168), fraction=0.1)

    # 10 % of 96 = 10 px, 10 % of 46 = 5 px: measured in pixels, not per axis.
    assert tall[10, 168] == 1.0 and tall[9, 168] < 1.0
    assert short[5, 84] == 1.0 and short[4, 84] < 1.0


def test_taper_fraction_at_the_limit_still_keeps_a_plateau() -> None:
    window = edge_taper_window((9, 9), fraction=0.5)

    assert window[4, 4] == pytest.approx(1.0)            # one-pixel plateau
    assert window[0, 0] == 0.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"width": 0}, ">= 1"),
        ({"width": -3}, ">= 1"),
        ({"width": 2.5}, "whole number"),
        ({"width": 99}, "no plateau"),
        ({"fraction": 0.0}, r"fraction must be in \(0, 0.5]"),
        ({"fraction": 0.75}, r"fraction must be in \(0, 0.5]"),
    ],
)
def test_taper_rejects_invalid_parameters(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        edge_taper_window((40, 40), **kwargs)
    with pytest.raises(ValueError, match=message):
        taper_edges(np.full((40, 40), 0.5, dtype=np.float32), **kwargs)


def test_taper_rejects_an_invalid_image() -> None:
    with pytest.raises(ValueError, match="2-D grayscale"):
        taper_edges(np.zeros((4, 4, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        taper_edges(np.full((4, 4), 2.0, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Shared numerical helpers
# --------------------------------------------------------------------------- #
def test_clip_unit_interval_clips_and_casts_to_float32() -> None:
    values = np.array([-0.4, 0.0, 0.25, 1.0, 1.7], dtype=np.float64)
    clipped = clip_unit_interval(values)

    assert clipped.dtype == np.float32
    assert np.array_equal(clipped, np.array([0.0, 0.0, 0.25, 1.0, 1.0],
                                            dtype=np.float32))


def test_clip_unit_interval_refuses_to_mask_non_finite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        clip_unit_interval(np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        clip_unit_interval(np.array([0.5, np.inf]))


def test_safe_divide_matches_the_plain_division_away_from_zero() -> None:
    numerator = np.array([1.0, -2.0, 3.0])
    denominator = np.array([2.0, 4.0, -8.0])

    assert np.allclose(safe_divide(numerator, denominator),
                       numerator / denominator, atol=0.0)


def test_safe_divide_keeps_the_quotient_finite_at_zero() -> None:
    numerator = np.array([1.0, -1.0, 0.0, 2.0])
    denominator = np.array([0.0, 0.0, 0.0, 1e-300])

    quotient = safe_divide(numerator, denominator)
    assert np.all(np.isfinite(quotient))
    assert quotient[0] == pytest.approx(1.0 / DEFAULT_EPS)
    assert quotient[1] == pytest.approx(-1.0 / DEFAULT_EPS)
    assert quotient[2] == 0.0
    # The clamped denominator keeps the sign of the original coefficient.
    assert quotient[1] < 0.0


def test_near_zero_mask_counts_the_clamped_coefficients() -> None:
    denominator = np.array([0.0, 1e-15, 1e-3, -1.0])

    mask = near_zero_mask(denominator)
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, np.array([True, True, False, False]))
    assert int(mask.sum()) == 2


def test_numerical_helpers_reject_invalid_input() -> None:
    with pytest.raises(ValueError, match="eps must be > 0"):
        safe_divide(np.ones(3), np.ones(3), eps=0.0)
    with pytest.raises(ValueError, match="eps must be > 0"):
        near_zero_mask(np.ones(3), eps=-1e-12)
    with pytest.raises(ValueError, match="non-finite"):
        safe_divide(np.array([1.0, np.nan]), np.ones(2))
    with pytest.raises(ValueError, match="non-finite"):
        near_zero_mask(np.array([1.0, np.inf]))


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 17: the PSF response is a
# unital, non-amplifying filter on the image's own DFT grid.
@given(seed=st.integers(0, 2**31 - 1),
       shape=st.tuples(st.integers(5, 24), st.integers(5, 24)))
@settings(max_examples=80, deadline=None)
def test_property_response_is_unital_and_bounded(seed: int,
                                                 shape: tuple[int, int]) -> None:
    response = psf_frequency_response(_random_psf(seed), shape)

    assert response.shape == shape
    assert response[0, 0].real == pytest.approx(1.0, abs=1e-6)
    assert response[0, 0].imag == pytest.approx(0.0, abs=1e-6)
    assert np.abs(response).max() <= 1.0 + 1e-6


# Feature: license-plate-restoration, Property 18: tapering preserves the image
# domain, stays inside [0, 1] and never touches the plateau.
@given(img=float32_image_strategy(), width=st.integers(1, 4))
@settings(max_examples=60, deadline=None)
def test_property_taper_preserves_domain(img: np.ndarray, width: int) -> None:
    width = min(width, min(img.shape) // 2)
    tapered = taper_edges(img, width=width)

    assert tapered.shape == img.shape
    assert tapered.dtype == np.float32
    assert np.all(np.isfinite(tapered))
    assert tapered.min() >= 0.0 and tapered.max() <= 1.0
    assert np.array_equal(tapered[width:-width, width:-width],
                          img[width:-width, width:-width])
