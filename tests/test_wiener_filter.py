"""Tests for Step 4.3 -- Wiener filtering.

Same policy as ``test_inverse_filter.py``: every case runs on small
deterministic arrays or on a plate rendered by step 1, so ``data/degraded.npz``
is never needed, and each docstring quotes the measurement it pins with at
least an order of magnitude of slack.

Two things to keep straight while reading:

* ``K`` is a **power** ratio (noise over signal), never a decibel value, and
  ``bsnr_db`` is the decibel value ``degradation.add_noise`` takes.
* ``K = 0`` is not a degenerate corner -- it is the inverse filter of step 4.2,
  reached through the Wiener expression instead of ``1 / H``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import float32_image_strategy
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.degradation import (
    add_noise,
    blur_fft,
    downsample,
    effective_psf,
)
from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)
from license_plate_restoration.renderer import generate_plates
from license_plate_restoration.restoration import (
    DEFAULT_EPS,
    DEFAULT_INVERSE_TAPER,
    DEFAULT_TAPER_FRACTION,
    inverse_filter,
    near_zero_mask,
    psf_centre_offset,
    psf_frequency_response,
    wiener_filter,
    wiener_k_from_bsnr,
)

DELTA = np.array([[1.0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Local helpers (deterministic by construction)
# --------------------------------------------------------------------------- #
def _textured(shape: tuple[int, int] = (40, 40), margin: int = 6,
              seed: int = 11) -> np.ndarray:
    """Flat uniform margin around a textured interior.

    The constant margin makes the circular model exact: opposite borders are
    equal, so no wrap discontinuity exists and any residual error belongs to
    the filter, not to the boundary.
    """
    rng = np.random.default_rng(seed)
    image = np.full(shape, 0.5, dtype=np.float64)
    image[margin:-margin, margin:-margin] = 0.25 + 0.5 * rng.random(
        (shape[0] - 2 * margin, shape[1] - 2 * margin))
    return image.astype(np.float32)


def _blob(shape: tuple[int, int] = (40, 40), row: float = 20.3,
          col: float = 20.0, sigma: float = 1.6) -> np.ndarray:
    """One bright dot at a sub-pixel position: a displacement probe."""
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    image = 0.9 * np.exp(-((rows - row) ** 2 + (cols - col) ** 2)
                         / (2.0 * sigma ** 2))
    return image.astype(np.float32)


def _gaussian(size: int, sigma: float = 1.0) -> np.ndarray:
    """Odd-sided, pixel-centred, sum-to-1 kernel."""
    axis = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2)
                    / sigma ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _even_gaussian(size: int = 8, sigma: float = 1.1) -> np.ndarray:
    """Even-sided kernel: its mass is centred *between* samples."""
    axis = np.arange(size) + 0.5 - size / 2.0
    kernel = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2)
                    / sigma ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _bilinear() -> np.ndarray:
    """2x2 linear interpolator: even-sided, and |H| vanishes at Nyquist."""
    return np.full((2, 2), 0.25, dtype=np.float32)


def _skewed() -> np.ndarray:
    """Asymmetric kernel, so its response is genuinely complex."""
    kernel = np.zeros((3, 3), dtype=np.float32)
    kernel[0, 0], kernel[1, 2], kernel[2, 1] = 0.5, 0.3, 0.2
    return kernel


def _circular_blur(image, response: np.ndarray) -> np.ndarray:
    """The forward model the filter inverts, evaluated without padding."""
    spectrum = np.fft.fft2(np.asarray(image, dtype=np.float64))
    return np.real(np.fft.ifft2(spectrum * response)).astype(np.float32)


def _rmse(reference, estimate, margin: int = 0) -> float:
    difference = (np.asarray(estimate, dtype=np.float64)
                  - np.asarray(reference, dtype=np.float64))
    if margin:
        difference = difference[margin:-margin, margin:-margin]
    return float(np.sqrt(np.mean(difference ** 2)))


def _centroid(image) -> float:
    rows = np.arange(np.shape(image)[0], dtype=np.float64)[:, None]
    mass = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    return float((mass * rows).sum() / mass.sum())


def _plate() -> np.ndarray:
    return generate_plates(1, seed=7)[0][0]


def _random_kernel(seed: int) -> np.ndarray:
    weights = np.random.default_rng(seed).random((3, 3))
    return (weights / weights.sum()).astype(np.float32)


def _clip32(values) -> np.ndarray:
    """The output convention, without importing the private helper."""
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0).astype(
        np.float32)


# --------------------------------------------------------------------------- #
# 1. Identity
# --------------------------------------------------------------------------- #
def test_delta_psf_with_zero_k_is_the_identity() -> None:
    """H == 1 everywhere and K == 0, so the quotient is a no-op, bit for bit."""
    image = _textured()

    restored = wiener_filter(image, DELTA, K=0.0)

    assert np.array_equal(restored, image)
    assert restored.dtype == np.float32
    assert restored.shape == image.shape


def test_delta_psf_scales_the_passband_by_one_over_one_plus_k() -> None:
    """H == 1 makes the gain exactly 1 / (1 + K): 0.4995 at K = 1e-3, not 0.5.

    The passband is passed with gain ``|H|^2 / (|H|^2 + K)``, so even a
    flat-transfer kernel loses 0.05% of the brightness at K = 1e-3.  That is
    the whole regularisation, visible on one pixel.
    """
    image = _textured()

    restored = wiener_filter(image, DELTA, K=1e-3)
    expected = image.astype(np.float64) / 1.001

    assert np.abs(restored.astype(np.float64) - expected).max() < 1e-6
    assert not np.array_equal(restored, image)


def test_default_arm_is_the_taper_free_one() -> None:
    """The default taper is `none`, exactly as step 4.2 measured it must be."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    assert DEFAULT_INVERSE_TAPER == "none"
    assert np.array_equal(wiener_filter(degraded, psf, K=1e-3),
                          wiener_filter(degraded, psf, K=1e-3,
                                        profile="none"))


# --------------------------------------------------------------------------- #
# 2. Noise-free controlled blur
# --------------------------------------------------------------------------- #
def test_noise_free_blur_is_recovered_by_a_tiny_k() -> None:
    """The blur costs 1.3e-1 RMS; K = 1e-12 puts 4.7e-6 of it back."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    restored = wiener_filter(degraded, psf, K=1e-12, profile="none")

    assert _rmse(image, degraded, 6) > 1e-2
    assert _rmse(image, restored, 6) < 1e-4
    assert _rmse(image, degraded, 6) > 100 * _rmse(image, restored, 6)


def test_tiny_k_agrees_with_the_inverse_filter_and_with_the_formula() -> None:
    """K -> 0 is the inverse filter: 1.8e-7 from step 4.2 and from H*G/|H|^2.

    The two programs are algebraically different routes to ``G / H`` --
    ``conj(H) G / |H|^2`` against ``G / H`` -- so they can only agree to the
    float32 round trip, one ULP at 0.5 (5.96e-8) either side.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)
    response = psf_frequency_response(psf, image.shape)

    restored = wiener_filter(degraded, psf, K=1e-12, profile="none")
    inverse = inverse_filter(degraded, psf, profile="none")
    spectrum = np.fft.fft2(degraded.astype(np.float64))
    formula = np.real(np.fft.ifft2(np.conj(response) * spectrum
                                   / (np.abs(response) ** 2 + 1e-12)))

    assert np.abs(restored.astype(np.float64)
                  - inverse.astype(np.float64)).max() < 1e-6
    assert np.abs(restored.astype(np.float64) - _clip32(formula)).max() < 1e-6


def test_estimate_is_the_inverse_filter_times_the_wiener_taper() -> None:
    """``conj(H) / (|H|^2 + K) == T * (1 / H)`` with ``T = |H|^2/(|H|^2+K)``.

    This is the identity the documentation builds the explanation on: the
    Wiener filter is the inverse filter multiplied by a real, frequency
    dependent taper in [0, 1).  Measured agreement 1e-7.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)
    response = psf_frequency_response(psf, image.shape)

    taper = np.abs(response) ** 2 / (np.abs(response) ** 2 + 1e-2)
    expected = np.real(np.fft.ifft2(
        taper * np.fft.fft2(degraded.astype(np.float64)) / response))
    restored = wiener_filter(degraded, psf, K=1e-2, profile="none")

    assert np.all(taper >= 0.0) and np.all(taper < 1.0)
    assert np.abs(restored.astype(np.float64) - _clip32(expected)).max() < 1e-6


# --------------------------------------------------------------------------- #
# 3. Noisy controlled blur
# --------------------------------------------------------------------------- #
def test_wiener_beats_both_the_degraded_image_and_the_inverse_filter() -> None:
    """At 20 dB BSNR: 1.26e-1 degraded, 4.41e-1 inverse, 1.07e-1 Wiener.

    The inverse filter is 3.5x worse than doing nothing -- it recovers detail
    and amplifies the noise riding on the nulls by more than the detail is
    worth.  The Wiener filter with the BSNR-derived K stays ahead of both.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    noisy = add_noise(blur_fft(image, psf), 20.0, seed=1)

    plain = inverse_filter(noisy, psf, profile="none")
    restored = wiener_filter(noisy, psf, bsnr_db=20.0, profile="none")

    assert _rmse(image, restored, 6) < _rmse(image, noisy, 6)
    assert 2 * _rmse(image, restored, 6) < _rmse(image, plain, 6)


def test_the_same_noise_admits_no_better_k_by_much() -> None:
    """The derived K is 1.2x the empirical optimum on this sample.

    Sweep at 20 dB: 1e-3 -> 8.6e-2, 1e-2 -> 1.07e-1, 1e-1 -> 1.3e-1.  The
    optimum sits below the derived value, which is the flat-spectrum model's
    known bias (section 14.3 of the docs), not a coding error.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    noisy = add_noise(blur_fft(image, psf), 20.0, seed=1)

    derived = wiener_filter(noisy, psf, bsnr_db=20.0, profile="none")
    best = wiener_filter(noisy, psf, K=1e-3, profile="none")

    assert _rmse(image, derived, 6) < 1.5 * _rmse(image, best, 6)


# --------------------------------------------------------------------------- #
# 4. Odd-sized kernels
# --------------------------------------------------------------------------- #
def test_odd_motion_psf_restores_a_plate() -> None:
    """A 7x7 `psf_builder` kernel is pixel-centred: 2.3e-3 RMS, no drift."""
    plate = _plate()
    psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))

    assert psf.shape == (7, 7)
    assert psf_centre_offset(psf) == (0.0, 0.0)

    degraded = blur_fft(plate, psf)
    restored = wiener_filter(degraded, psf, K=1e-8, profile="none")

    assert _rmse(plate, degraded, 12) > 1e-1                # measured 1.1e-1
    assert _rmse(plate, restored, 12) < 1e-2                # measured 2.3e-3
    assert abs(_centroid(restored) - _centroid(plate)) < 0.5


# --------------------------------------------------------------------------- #
# 5. Even-sized kernels and the half pixel
# --------------------------------------------------------------------------- #
def test_even_sized_psf_is_restored_without_a_half_pixel_shift() -> None:
    """`centre="auto"` lands on 20.3000 px at 6.5e-7 RMS; pixel is 0.5 off."""
    reference = _blob()
    psf = _even_gaussian()
    response = psf_frequency_response(psf, reference.shape)
    degraded = _circular_blur(reference, response)

    assert psf_centre_offset(psf) == pytest.approx((-0.5, -0.5), abs=1e-3)

    auto = wiener_filter(degraded, psf, K=1e-6, profile="none")
    pixel = wiener_filter(degraded, psf, K=1e-6, profile="none",
                          centre="pixel")

    assert _rmse(reference, auto, 6) < 1e-4                 # measured 6.5e-7
    assert abs(_centroid(auto) - _centroid(reference)) < 0.01
    assert abs(_centroid(pixel) - _centroid(reference)) > 0.4    # measured .5
    assert _rmse(reference, pixel, 6) > 1e-2                # measured 2.8e-2


def test_even_stored_kernel_keeps_the_decimated_plate_aligned() -> None:
    """The dataset's factor-2 path: auto 0.089 vs pixel 0.195 at K = 1e-4.

    ``effective_psf(combined, 2)`` is even-sided here, which is the geometry a
    stored ``_psf`` has in ``degraded.npz``.  ``centre="auto"`` halves the
    error and keeps the plate within 0.006 px of the reference; the pixel rule
    shifts it by 0.151 px.
    """
    plate = _plate()
    full = combine_psfs(make_motion_psf(15, 20.0), make_defocus_psf(3))
    stored = effective_psf(full, 2)

    assert stored.shape[0] % 2 == 0                     # even, hence ambiguous
    observed = downsample(blur_fft(plate, full), 2)
    reference = downsample(plate, 2)

    auto = wiener_filter(observed, stored, K=1e-4, profile="none")
    pixel = wiener_filter(observed, stored, K=1e-4, profile="none",
                          centre="pixel")

    assert _rmse(reference, auto, 12) < 0.5 * _rmse(reference, pixel, 12)
    assert abs(_centroid(auto) - _centroid(reference)) < 0.05
    assert abs(_centroid(pixel) - _centroid(reference)) > 0.1


# --------------------------------------------------------------------------- #
# 6. Complex-valued frequency response
# --------------------------------------------------------------------------- #
def test_complex_frequency_response_keeps_the_conjugate() -> None:
    """Without ``conj(H)`` the phase is lost: 4.8e-1 against 1.4e-2.

    An asymmetric kernel gives a response with |Im H| up to 0.766.  The Wiener
    numerator is the *complex conjugate*, not the magnitude and not the real
    part; dropping it applies an operator that neither deblurs nor preserves
    the plate, 33x worse than keeping it.
    """
    reference = _textured()
    psf = _skewed()
    response = psf_frequency_response(psf, reference.shape)

    assert np.iscomplexobj(response)
    assert float(np.abs(response.imag).max()) > 0.1
    degraded = _circular_blur(reference, response)

    restored = wiener_filter(degraded, psf, K=1e-6, profile="none")
    spectrum = np.fft.fft2(degraded.astype(np.float64))
    without_conjugate = np.real(np.fft.ifft2(
        spectrum / (np.abs(response) ** 2 + 1e-6)))

    assert _rmse(reference, restored, 6) < 0.1               # measured 1.4e-2
    assert _rmse(reference, without_conjugate, 6) > 10 * _rmse(
        reference, restored, 6)                             # measured 4.8e-1


# --------------------------------------------------------------------------- #
# 7. Exact zeros of H
# --------------------------------------------------------------------------- #
def test_exact_zero_frequency_response_stays_finite() -> None:
    """The 2x2 interpolator nulls Nyquist; every K arm stays finite, in range."""
    image = _textured()
    psf = _bilinear()
    response = psf_frequency_response(psf, image.shape)

    assert int(near_zero_mask(response, eps=DEFAULT_EPS).sum()) > 0
    assert float(np.abs(response).min()) < 1e-12
    degraded = _circular_blur(image, response)

    for constant in (0.0, 1e-12, 1e-3, 1e-1):
        restored = wiener_filter(degraded, psf, K=constant, profile="none")
        assert np.all(np.isfinite(restored))
        assert restored.dtype == np.float32
        assert restored.min() >= 0.0 and restored.max() <= 1.0


# --------------------------------------------------------------------------- #
# 8. K = 0
# --------------------------------------------------------------------------- #
def test_zero_k_is_the_inverse_filter_of_step_4_2() -> None:
    """K = 0 and `inverse_filter` agree to float32 round-off (measured 0.0).

    The two programs are algebraically different routes to ``G / H`` --
    ``conj(H) G / |H|^2`` against ``G / H`` -- and agree to 0.0 here, well
    inside the 1e-7 asserted.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    restored = wiener_filter(degraded, psf, K=0.0, profile="none")
    inverse = inverse_filter(degraded, psf, profile="none")

    assert np.abs(restored.astype(np.float64)
                  - inverse.astype(np.float64)).max() < 1e-7


def test_zero_k_at_an_exact_null_is_zero_gain_not_a_division() -> None:
    """At |H| == 0 the unguarded quotient is NaN; K = 0 returns 0, finite.

    This is the one coefficient the Wiener denominator does not protect,
    because at K = 0 it is 0/0 and the gain's limit ``1/|H|`` *diverges* -- the
    inverse filter's ill-conditioning in its purest form.  The zero is a
    stated convention (a destroyed frequency gets no gain), written into the
    division rather than patched afterwards.  The error stays large (0.523
    RMS): the destroyed band is not invented.
    """
    image = _textured()
    psf = _bilinear()
    response = psf_frequency_response(psf, image.shape)
    degraded = _circular_blur(image, response)
    spectrum = np.fft.fft2(degraded.astype(np.float64))

    with np.errstate(divide="ignore", invalid="ignore"):
        unguarded = np.real(np.fft.ifft2(np.conj(response) * spectrum
                                         / np.abs(response) ** 2))
    restored = wiener_filter(degraded, psf, K=0.0, profile="none")

    assert not np.all(np.isfinite(unguarded))
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0 and restored.max() <= 1.0
    assert _rmse(image, restored, 6) < 1.0                  # measured 5.2e-1


# --------------------------------------------------------------------------- #
# 9. The direct K parameter
# --------------------------------------------------------------------------- #
def test_larger_k_lowers_the_output_contrast() -> None:
    """The taper shrinks with K, so the estimate's spread shrinks too.

    Measured output std: 8.5e-2, 7.0e-2, 5.4e-2, 3.5e-2, 1.3e-2 for
    K = 1e-4 ... 1.0 -- strictly decreasing, which is the monotonicity the
    filter promises.
    """
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    spreads = [float(wiener_filter(degraded, psf, K=constant,
                                   profile="none").std())
               for constant in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]

    assert spreads == sorted(spreads, reverse=True)
    assert spreads[0] > 3 * spreads[-1]


def test_large_k_returns_an_almost_black_image() -> None:
    """K = 1e6 divides every coefficient by ~1e6: the estimate goes to 0."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    restored = wiener_filter(degraded, psf, K=1e6, profile="none")

    assert restored.max() < 1e-5                            # measured 5.6e-7
    assert np.all(np.isfinite(restored))


# --------------------------------------------------------------------------- #
# 10. K derived from BSNR
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bsnr_db", [40.0, 30.0, 25.0, 20.0])
def test_bsnr_conversion_matches_the_realised_noise_ratio(
        bsnr_db: float) -> None:
    """K = var(noise) / var(noisy), read off the dataset's own ``add_noise``.

    The repository's BSNR is defined against the *blurred* signal, so
    ``var(noisy) = var(blurred)(1 + rho)`` and the sharp-signal ratio is
    ``rho / (1 + rho)``.  ``add_noise``'s saturation compensation makes the
    realised residual variance match the target, so the identity is testable:
    measured agreement 1.0003, 1.0009, 1.0016, 1.0028 over the four levels.
    """
    image = _textured()
    blurred = blur_fft(image, _gaussian(5, 1.0))
    noisy = add_noise(blurred, bsnr_db, seed=5)

    residual = noisy.astype(np.float64) - blurred.astype(np.float64)
    measured = float(np.var(residual) / np.var(noisy.astype(np.float64)))

    assert wiener_k_from_bsnr(bsnr_db) == pytest.approx(measured, rel=0.01)


def test_bsnr_argument_is_exactly_the_equivalent_k_argument() -> None:
    """``bsnr_db=`` must be sugar for ``K=wiener_k_from_bsnr(...)``."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    noisy = add_noise(blur_fft(image, psf), 20.0, seed=1)

    assert np.array_equal(
        wiener_filter(noisy, psf, bsnr_db=20.0, profile="none"),
        wiener_filter(noisy, psf, K=wiener_k_from_bsnr(20.0),
                      profile="none"))


def test_k_from_bsnr_is_the_linear_ratio_over_one_plus_itself() -> None:
    """The conversion, at the values the project uses and at the limits.

    40 dB -> 9.999e-05, 20 dB -> 9.901e-03, 0 dB -> 0.5, -10 dB -> 0.909.
    ``K = rho`` (``1 / BSNR_linear``) would be within 1% of the first two and
    wrong by 2x at 0 dB, so the exact form is what is implemented.
    """
    assert wiener_k_from_bsnr(40.0) == pytest.approx(1e-4 / 1.0001, rel=1e-9)
    assert wiener_k_from_bsnr(20.0) == pytest.approx(1e-2 / 1.01, rel=1e-9)
    assert wiener_k_from_bsnr(0.0) == pytest.approx(0.5, rel=1e-12)
    assert wiener_k_from_bsnr(-10.0) == pytest.approx(10.0 / 11.0, rel=1e-12)

    levels = [wiener_k_from_bsnr(value) for value in (40.0, 30.0, 25.0, 20.0)]
    assert levels == sorted(levels)                 # more noise -> larger K
    assert all(0.0 <= constant < 1.0 for constant in levels)


# --------------------------------------------------------------------------- #
# 11. Invalid parameter combinations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs, message", [
    ({}, "needs a regularisation constant"),
    ({"K": None}, "needs a regularisation constant"),
    ({"K": 1e-3, "bsnr_db": 20.0}, "not both"),
    ({"K": -1e-3}, "K must be >= 0"),
    ({"K": float("nan")}, "K must be a finite number"),
    ({"K": np.inf}, "K must be a finite number"),
    ({"K": "1e-3"}, "K must be a number"),
    ({"bsnr_db": float("nan")}, "bsnr_db must be a finite number"),
    ({"bsnr_db": np.inf}, "bsnr_db must be a finite number"),
    ({"bsnr_db": "20"}, "bsnr_db must be a number"),
])
def test_wiener_rejects_invalid_regularisation(kwargs: dict,
                                               message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wiener_filter(_textured(), DELTA, **kwargs)


# --------------------------------------------------------------------------- #
# 12. Invalid image and PSF
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("image, psf, kwargs, message", [
    (np.zeros(8, dtype=np.float32), DELTA, {}, "2-D grayscale"),
    (np.full((8, 8), 2.0, dtype=np.float32), DELTA, {}, r"lie in \[0, 1\]"),
    (np.full((8, 8), np.nan, dtype=np.float32), DELTA, {}, "non-finite"),
    (np.zeros((8, 8), dtype=np.complex128), DELTA, {}, "must be real-valued"),
    (_textured(), np.full((8, 8), 1.0 / 64.0), {"require_odd": True},
     "must be odd-sized"),
    (_textured(), np.full((5, 5), 0.05), {}, "must sum to 1"),
    (_textured(), np.array([[1.2, -0.1], [-0.1, 0.0]]), {}, "negative weight"),
    (np.full((4, 4), 0.5, dtype=np.float32), np.full((7, 7), 1.0 / 49.0), {},
     "does not fit"),
    (_textured(), DELTA, {"profile": "blackman"}, "profile must be one of"),
    (_textured(), DELTA, {"centre": "middle"}, "centre must be one of"),
    (_textured(), DELTA, {"profile": "none", "width": 8}, "width is not used"),
    (_textured(), DELTA, {"profile": "tukey", "fraction": 0.9},
     r"fraction must be in \(0, 0.5\]"),
])
def test_wiener_rejects_invalid_input(image, psf, kwargs: dict,
                                      message: str) -> None:
    with pytest.raises(ValueError, match=message):
        wiener_filter(image, psf, K=1e-3, **kwargs)


# --------------------------------------------------------------------------- #
# 13. Output convention, both taper arms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ["none", "tukey"])
@pytest.mark.parametrize("constant", [1e-12, 1e-2, 1.0])
def test_output_keeps_the_image_convention(profile: str,
                                           constant: float) -> None:
    plate = _plate()
    psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))
    noisy = add_noise(blur_fft(plate, psf), 20.0, seed=2)

    restored = wiener_filter(noisy, psf, K=constant, profile=profile)

    assert restored.shape == plate.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_taper_choice_only_changes_the_border() -> None:
    """Tapering is a pre-multiplication, so it survives into the estimate.

    With a delta PSF the estimate is the windowed input divided by 1 + K:
    identical on the plateau, zeroed at the corner (0.0 against 0.4995), and
    the unwindowed image when the taper is off.
    """
    plate = _plate()
    width = int(round(DEFAULT_TAPER_FRACTION * min(plate.shape)))
    reference = plate.astype(np.float64) / 1.001

    plain = wiener_filter(plate, DELTA, K=1e-3, profile="none")
    windowed = wiener_filter(plate, DELTA, K=1e-3, profile="tukey")

    assert np.abs(plain.astype(np.float64) - reference).max() < 1e-6
    assert np.abs(windowed[width:-width, width:-width].astype(np.float64)
                  - reference[width:-width, width:-width]).max() < 1e-6
    assert windowed[0, 0] < 0.01 < 0.4
    assert not np.array_equal(windowed, plain)


# --------------------------------------------------------------------------- #
# Property test
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 22: Wiener filtering always
# returns a finite float32 image inside [0, 1], whatever the kernel and K are.
@given(image=float32_image_strategy(min_h=8, min_w=8, max_h=18, max_w=18),
       seed=st.integers(0, 2 ** 31 - 1),
       constant=st.sampled_from([0.0, 1e-12, 1e-6, 1e-3, 1e-1, 1.0]))
@settings(max_examples=40, deadline=None)
def test_property_output_is_always_a_valid_image(image: np.ndarray, seed: int,
                                                 constant: float) -> None:
    restored = wiener_filter(image, _random_kernel(seed), K=constant,
                             profile="none")

    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0 and restored.max() <= 1.0
