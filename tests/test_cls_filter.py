"""Tests for Constrained Least Squares (CLS) restoration (step 4.4).

Verifies:
* :data:`DEFAULT_LAPLACIAN` properties and signed operator response;
* :func:`operator_frequency_response` and :func:`laplacian_frequency_response`;
* :func:`cls_filter` input validation, edge cases, and API contract;
* Mathematical correctness against the analytical formula and inverse filter;
* Numerical stability at exact frequency nulls and near-zero denominators;
* Complex arithmetic and phase preservation on asymmetric kernels;
* Regularisation behaviour under increasing lambda;
* Property-based verification that outputs remain valid float32 images in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import float32_image_strategy
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
    DEFAULT_CENTRE_MODE,
    DEFAULT_INVERSE_TAPER,
    DEFAULT_LAPLACIAN,
    DEFAULT_TAPER_FRACTION,
    cls_filter,
    inverse_filter,
    laplacian_frequency_response,
    operator_frequency_response,
    psf_frequency_response,
    wiener_filter,
)

DELTA = np.array([[1.0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Local test fixtures and deterministic helpers
# --------------------------------------------------------------------------- #
def _textured(shape: tuple[int, int] = (40, 40), margin: int = 6,
              seed: int = 11) -> np.ndarray:
    """Flat uniform margin around a textured interior."""
    rng = np.random.default_rng(seed)
    image = np.full(shape, 0.5, dtype=np.float64)
    image[margin:-margin, margin:-margin] = 0.25 + 0.5 * rng.random(
        (shape[0] - 2 * margin, shape[1] - 2 * margin))
    return image.astype(np.float32)


def _gaussian(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Odd-sided, pixel-centred, sum-to-1 kernel."""
    axis = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (axis[:, None] ** 2 + axis[None, :] ** 2)
                    / sigma ** 2)
    return (kernel / kernel.sum()).astype(np.float32)


def _even_gaussian(size: int = 8, sigma: float = 1.1) -> np.ndarray:
    """Even-sided kernel: its mass is centred between samples."""
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
    """Circular convolution forward model."""
    spectrum = np.fft.fft2(np.asarray(image, dtype=np.float64))
    return np.real(np.fft.ifft2(spectrum * response)).astype(np.float32)


def _rmse(reference, estimate, margin: int = 0) -> float:
    difference = (np.asarray(estimate, dtype=np.float64)
                  - np.asarray(reference, dtype=np.float64))
    if margin:
        difference = difference[margin:-margin, margin:-margin]
    return float(np.sqrt(np.mean(difference ** 2)))


def _plate() -> np.ndarray:
    return generate_plates(1, seed=7)[0][0]


# --------------------------------------------------------------------------- #
# 1. Laplacian kernel and operator frequency response
# --------------------------------------------------------------------------- #
def test_default_laplacian_properties() -> None:
    """Laplacian kernel must be 3x3, zero-sum, with the 4-neighbour stencil."""
    assert DEFAULT_LAPLACIAN.shape == (3, 3)
    assert DEFAULT_LAPLACIAN.dtype == np.float32
    assert float(np.sum(DEFAULT_LAPLACIAN)) == pytest.approx(0.0, abs=1e-7)
    assert DEFAULT_LAPLACIAN[1, 1] == 4.0
    assert DEFAULT_LAPLACIAN[0, 1] == -1.0
    assert DEFAULT_LAPLACIAN[2, 1] == -1.0
    assert DEFAULT_LAPLACIAN[1, 0] == -1.0
    assert DEFAULT_LAPLACIAN[1, 2] == -1.0
    assert DEFAULT_LAPLACIAN[0, 0] == 0.0
    assert DEFAULT_LAPLACIAN[0, 2] == 0.0
    assert DEFAULT_LAPLACIAN[2, 0] == 0.0
    assert DEFAULT_LAPLACIAN[2, 2] == 0.0


def test_laplacian_frequency_response_dc_is_exactly_zero() -> None:
    """C(0, 0) must be 0 because sum(c) == 0: DC brightness is unpenalized."""
    for shape in ((16, 16), (40, 60), (96, 336)):
        C = laplacian_frequency_response(shape)
        assert C.shape == shape
        assert C.dtype == np.complex128
        assert C[0, 0] == pytest.approx(0.0 + 0.0j, abs=1e-12)


def test_laplacian_frequency_response_matches_analytical_formula() -> None:
    """C(u, v) = 4 - 2 cos(2 pi u / M) - 2 cos(2 pi v / N) in [0, 8]."""
    shape = (32, 48)
    C = laplacian_frequency_response(shape)

    # Purely real due to spatial symmetry
    assert float(np.max(np.abs(C.imag))) < 1e-12

    u = np.arange(shape[0], dtype=np.float64)[:, None]
    v = np.arange(shape[1], dtype=np.float64)[None, :]
    analytical = (4.0 - 2.0 * np.cos(2.0 * np.pi * u / shape[0])
                  - 2.0 * np.cos(2.0 * np.pi * v / shape[1]))

    assert np.allclose(C.real, analytical, atol=1e-12)
    assert float(np.min(C.real)) >= 0.0
    assert float(np.max(C.real)) <= 8.0


@pytest.mark.parametrize("bad_op, message", [
    (np.zeros(5, dtype=np.float32), "2-D array"),
    (np.zeros((2, 2, 2), dtype=np.float32), "2-D array"),
    (np.zeros((0, 3), dtype=np.float32), "empty"),
    (np.array([[1.0, np.nan]], dtype=np.float32), "non-finite"),
    (np.array([[1.0, np.inf]], dtype=np.float32), "non-finite"),
    (np.zeros((50, 10), dtype=np.float32), "does not fit"),
])
def test_operator_frequency_response_rejects_invalid_operator(
        bad_op, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operator_frequency_response(bad_op, (40, 40))


# --------------------------------------------------------------------------- #
# 2. Input validation for cls_filter
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_reg, message", [
    (-1e-3, "regularization must be >= 0"),
    (-1.0, "regularization must be >= 0"),
    (float("nan"), "regularization must be a finite number"),
    (float("inf"), "regularization must be a finite number"),
    (-float("inf"), "regularization must be a finite number"),
    ("0.01", "regularization must be a number"),
    (None, "regularization must be a number"),
])
def test_cls_rejects_invalid_regularization(bad_reg, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cls_filter(_textured(), DELTA, bad_reg)


@pytest.mark.parametrize("bad_image, message", [
    (np.zeros(10, dtype=np.float32), "2-D grayscale"),
    (np.zeros((10, 10, 3), dtype=np.float32), "2-D grayscale"),
    (np.zeros((0, 10), dtype=np.float32), "empty"),
    (np.full((10, 10), np.nan, dtype=np.float32), "non-finite"),
    (np.full((10, 10), np.inf, dtype=np.float32), "non-finite"),
    (np.full((10, 10), -0.1, dtype=np.float32), r"\[0, 1\]"),
    (np.full((10, 10), 1.1, dtype=np.float32), r"\[0, 1\]"),
])
def test_cls_rejects_invalid_image(bad_image, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cls_filter(bad_image, DELTA, 0.01)


@pytest.mark.parametrize("bad_psf, message", [
    (np.zeros(5, dtype=np.float32), "2-D kernel"),
    (np.zeros((0, 3), dtype=np.float32), "empty"),
    (np.full((3, 3), 0.2, dtype=np.float32), "must sum to 1"),
    (np.array([[-0.1, 1.1]], dtype=np.float32), "negative weight"),
])
def test_cls_rejects_invalid_psf(bad_psf, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cls_filter(_textured(), bad_psf, 0.01)


# --------------------------------------------------------------------------- #
# 3. Mathematical correctness
# --------------------------------------------------------------------------- #
def test_delta_psf_with_zero_lambda_is_the_identity() -> None:
    """H == 1 and lambda == 0: no blur, no regularisation -> exact identity."""
    image = _textured()
    restored = cls_filter(image, DELTA, 0.0)
    assert np.array_equal(restored, image)
    assert restored.dtype == np.float32


def test_constant_image_is_preserved_for_any_lambda() -> None:
    """Because C(0, 0) == 0, the DC gain is 1 / (1 + lambda * 0) == 1.0."""
    shape = (32, 32)
    for value in (0.1, 0.35, 0.5, 0.8):
        image = np.full(shape, value, dtype=np.float32)
        for lam in (0.0, 1e-4, 1e-2, 1.0, 100.0):
            restored = cls_filter(image, DELTA, lam)
            assert np.allclose(restored, value, atol=1e-6)


def test_cls_formula_agrees_with_manual_evaluation() -> None:
    """Step-by-step frequency-domain calculation matches cls_filter output."""
    image = _textured((32, 32))
    psf = _gaussian(5, 1.0)
    lam = 0.02

    response = psf_frequency_response(psf, image.shape)
    c_response = laplacian_frequency_response(image.shape)
    spectrum = np.fft.fft2(image.astype(np.float64))

    denominator = np.abs(response) ** 2 + lam * (np.abs(c_response) ** 2)
    manual_estimate = np.divide(np.conj(response) * spectrum, denominator,
                                out=np.zeros_like(spectrum), where=denominator > 0)
    manual_restored = np.clip(np.real(np.fft.ifft2(manual_estimate)), 0.0, 1.0).astype(np.float32)

    filter_restored = cls_filter(image, psf, regularization=lam)
    assert np.allclose(filter_restored, manual_restored, atol=1e-6)


def test_zero_lambda_matches_unregularised_inverse_filter() -> None:
    """At lambda == 0, CLS is G / H, matching inverse_filter on benign PSFs."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    cls_zero = cls_filter(degraded, psf, 0.0, profile="none")
    inv = inverse_filter(degraded, psf, profile="none")
    wiener_zero = wiener_filter(degraded, psf, K=0.0, profile="none")

    assert np.abs(cls_zero.astype(np.float64) - inv.astype(np.float64)).max() < 1e-7
    assert np.abs(cls_zero.astype(np.float64) - wiener_zero.astype(np.float64)).max() < 1e-7


# --------------------------------------------------------------------------- #
# 4. Numerical stability and frequency nulls
# --------------------------------------------------------------------------- #
def test_exact_zero_frequency_response_stays_finite() -> None:
    """The 2x2 interpolator has an exact zero at Nyquist; CLS stays finite."""
    image = _textured()
    psf = _bilinear()
    response = psf_frequency_response(psf, image.shape)
    degraded = _circular_blur(image, response)

    for lam in (0.0, 1e-12, 1e-4, 1e-2, 1.0):
        restored = cls_filter(degraded, psf, regularization=lam, profile="none")
        assert np.all(np.isfinite(restored))
        assert restored.dtype == np.float32
        assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0


def test_zero_lambda_at_exact_null_returns_zero_gain_not_nan() -> None:
    """At lambda == 0, 0 / 0 is evaluated as zero gain, never NaN or Inf."""
    image = _textured()
    psf = _bilinear()
    response = psf_frequency_response(psf, image.shape)
    degraded = _circular_blur(image, response)

    restored = cls_filter(degraded, psf, 0.0, profile="none")
    assert np.all(np.isfinite(restored))
    assert _rmse(image, restored, 6) < 1.0


def test_complex_frequency_response_preserves_conjugate_phase() -> None:
    """Without H* in the numerator, asymmetric phase is inverted: error is huge."""
    image = _textured()
    psf = _skewed()
    response = psf_frequency_response(psf, image.shape)
    degraded = _circular_blur(image, response)

    restored = cls_filter(degraded, psf, 1e-4, profile="none")
    spectrum = np.fft.fft2(degraded.astype(np.float64))
    c_resp = laplacian_frequency_response(image.shape)
    denom = np.abs(response) ** 2 + 1e-4 * (np.abs(c_resp) ** 2)

    # Deliberately omitting the conjugate
    without_conj = np.clip(np.real(np.fft.ifft2(spectrum / denom)), 0.0, 1.0)

    err_correct = _rmse(image, restored, 6)
    err_wrong = _rmse(image, without_conj, 6)
    assert err_correct < 0.1
    assert err_wrong > 5.0 * err_correct


def test_even_sized_psf_restores_without_shift() -> None:
    """Even-sized kernels with centre='auto' keep features aligned."""
    image = _textured()
    psf = _even_gaussian(8, 1.1)
    degraded = blur_fft(image, psf)

    restored = cls_filter(degraded, psf, 1e-4, centre="auto", profile="none")
    assert np.all(np.isfinite(restored))
    assert _rmse(image, restored, 8) < _rmse(image, degraded, 8)


# --------------------------------------------------------------------------- #
# 5. Behaviour under parameter variation
# --------------------------------------------------------------------------- #
def test_increasing_lambda_increases_smoothness_and_lowers_contrast() -> None:
    """Larger lambda penalises the Laplacian more heavily: output variance drops."""
    image = _textured((48, 48))
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    variances = []
    for lam in (1e-5, 1e-3, 1e-1, 10.0):
        restored = cls_filter(degraded, psf, lam)
        variances.append(float(np.var(restored[6:-6, 6:-6])))

    # Variance must decrease monotonically as regularisation strengthens
    assert variances == sorted(variances, reverse=True)


def test_cls_beats_inverse_filter_on_noisy_plate() -> None:
    """On a plate with noise (20 dB), CLS beats inverse filter decisively."""
    plate = _plate()
    psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))
    blurred = blur_fft(plate, psf)
    noisy = add_noise(blurred, 20.0, seed=3)

    err_deg = _rmse(plate, noisy, 12)
    err_inv = _rmse(plate, inverse_filter(noisy, psf), 12)
    err_cls = _rmse(plate, cls_filter(noisy, psf, 0.01), 12)

    assert err_inv > 5.0 * err_deg      # inverse explodes (> 0.70)
    assert err_cls < err_deg           # CLS deblurs and cleans (< 0.08)
    assert err_cls < 0.15 * err_inv    # CLS is > 6x better than inverse


@pytest.mark.parametrize("profile", ["none", "tukey", "hann", "linear"])
def test_cls_accepts_all_edge_taper_profiles(profile: str) -> None:
    """Taper profile is accepted and returns finite in-bounds image."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)
    restored = cls_filter(degraded, psf, 0.01, profile=profile)
    assert np.all(np.isfinite(restored))
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0


# --------------------------------------------------------------------------- #
# 6. Hypothesis property test
# --------------------------------------------------------------------------- #
@given(image=float32_image_strategy(min_h=8, min_w=8, max_h=24, max_w=24),
       reg=st.sampled_from([0.0, 1e-12, 1e-6, 1e-4, 1e-2, 1.0]))
@settings(max_examples=25, deadline=None)
def test_property_cls_output_is_always_valid_float32(image: np.ndarray, reg: float) -> None:
    """For any valid float32 image and non-negative lambda, output is valid."""
    restored = cls_filter(image, DELTA, regularization=reg)
    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0
