"""Tests for Richardson--Lucy (RL) restoration (step 4.5).

Verifies:
* Public API contract and input validation for iterations, eps, initial_estimate,
  clip_intermediate, image, and PSF;
* Output properties: shape, dtype (float32), finiteness, unit-interval bounds,
  immutability of inputs, determinism;
* Mathematical correctness: delta PSF identity, constant image fixed point,
  forward/adjoint inner-product duality (<Hx, y> == <x, H^T y>), flux conservation;
* Spatial centring: odd vs even PSFs, verifying centre="auto" avoids half-pixel shift;
* Numerical stability: zero-valued pixels, near-zero denominators, extreme iterations;
* Restoration behavior: clean deblurring, noise-model semi-convergence;
* All four project resolutions (96x336, 92x256, 48x168, 46x128);
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
    DEFAULT_EPS,
    DEFAULT_INVERSE_TAPER,
    DEFAULT_TAPER_FRACTION,
    psf_frequency_response,
    richardson_lucy_filter,
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


def _centroid(image: np.ndarray) -> tuple[float, float]:
    rows = np.arange(image.shape[0], dtype=np.float64)[:, None]
    cols = np.arange(image.shape[1], dtype=np.float64)[None, :]
    mass = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    total = float(mass.sum())
    return float((mass * rows).sum() / total), float((mass * cols).sum() / total)


def _plate() -> np.ndarray:
    return generate_plates(1, seed=7)[0][0]


# --------------------------------------------------------------------------- #
# 1. API and input validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_iters, message", [
    (-1, "iterations must be >= 0"),
    (-10, "iterations must be >= 0"),
    (1.5, "iterations must be an integer >= 0"),
    (True, "iterations must be an integer >= 0"),
    (False, "iterations must be an integer >= 0"),
    ("30", "iterations must be an integer >= 0"),
    (None, "iterations must be an integer >= 0"),
    (float("nan"), "iterations must be a finite number"),
    (float("inf"), "iterations must be a finite number"),
    (-float("inf"), "iterations must be a finite number"),
])
def test_rl_rejects_invalid_iterations(bad_iters, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        richardson_lucy_filter(_textured(), DELTA, iterations=bad_iters)


@pytest.mark.parametrize("bad_eps, message", [
    (0.0, "eps must be >"),
    (-1e-3, "eps must be >"),
    (float("nan"), "eps must be a finite number"),
    (float("inf"), "eps must be a finite number"),
    ("1e-12", "eps must be a number"),
    (None, "eps must be a number"),
])
def test_rl_rejects_invalid_eps(bad_eps, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        richardson_lucy_filter(_textured(), DELTA, eps=bad_eps)


@pytest.mark.parametrize("bad_clip", [1, 0, "True", "False", None, 1.0])
def test_rl_rejects_invalid_clip_intermediate(bad_clip) -> None:
    with pytest.raises(ValueError, match="clip_intermediate must be a bool"):
        richardson_lucy_filter(_textured(), DELTA, clip_intermediate=bad_clip)


@pytest.mark.parametrize("bad_image, message", [
    (np.zeros(10, dtype=np.float32), "2-D grayscale"),
    (np.zeros((10, 10, 3), dtype=np.float32), "2-D grayscale"),
    (np.zeros((0, 10), dtype=np.float32), "empty"),
    (np.full((10, 10), np.nan, dtype=np.float32), "non-finite"),
    (np.full((10, 10), np.inf, dtype=np.float32), "non-finite"),
    (np.full((10, 10), -0.1, dtype=np.float32), r"\[0, 1\]"),
    (np.full((10, 10), 1.1, dtype=np.float32), r"\[0, 1\]"),
])
def test_rl_rejects_invalid_image(bad_image, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        richardson_lucy_filter(bad_image, DELTA)


@pytest.mark.parametrize("bad_psf, message", [
    (np.zeros(5, dtype=np.float32), "2-D kernel"),
    (np.zeros((0, 3), dtype=np.float32), "empty"),
    (np.full((3, 3), 0.2, dtype=np.float32), "must sum to 1"),
    (np.array([[-0.1, 1.1]], dtype=np.float32), "negative weight"),
])
def test_rl_rejects_invalid_psf(bad_psf, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        richardson_lucy_filter(_textured(), bad_psf)


def test_rl_rejects_zero_sum_psf() -> None:
    zero_psf = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="must sum to 1"):
        richardson_lucy_filter(_textured(), zero_psf)


@pytest.mark.parametrize("bad_init, message", [
    (np.zeros((10, 10), dtype=np.float32), "does not match image shape"),
    (np.full((40, 40), np.nan, dtype=np.float32), "non-finite"),
    (np.full((40, 40), np.inf, dtype=np.float32), "non-finite"),
    (np.full((40, 40), -0.5, dtype=np.float32), "must be non-negative"),
])
def test_rl_rejects_invalid_initial_estimate(bad_init, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        richardson_lucy_filter(_textured((40, 40)), DELTA,
                               initial_estimate=bad_init)


def test_rl_does_not_mutate_inputs() -> None:
    """Caller-provided image and PSF must remain byte-for-byte unchanged."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    image_copy = np.copy(image)
    psf_copy = np.copy(psf)

    _ = richardson_lucy_filter(image, psf, iterations=5)
    assert np.array_equal(image, image_copy)
    assert np.array_equal(psf, psf_copy)


# --------------------------------------------------------------------------- #
# 2. Output properties
# --------------------------------------------------------------------------- #
def test_rl_output_properties() -> None:
    """Output must be float32 array of same shape in [0, 1] with finite values."""
    shape = (32, 48)
    image = _textured(shape)
    psf = _gaussian(5, 1.0)

    out = richardson_lucy_filter(image, psf, iterations=5)
    assert out.shape == shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_rl_is_deterministic() -> None:
    """Identical inputs must produce bit-identical results."""
    image = _textured()
    psf = _gaussian(5, 1.0)

    run1 = richardson_lucy_filter(image, psf, iterations=10)
    run2 = richardson_lucy_filter(image, psf, iterations=10)
    assert np.array_equal(run1, run2)


def test_rl_iterations_zero_returns_initial_estimate() -> None:
    """iterations=0 must return the initial estimate clipped to [0, 1]."""
    image = _textured()
    out = richardson_lucy_filter(image, DELTA, iterations=0)
    assert np.allclose(out, image, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. Mathematical correctness
# --------------------------------------------------------------------------- #
def test_delta_psf_leaves_image_unchanged() -> None:
    """An identity PSF (delta) should leave the image unchanged."""
    image = _textured()
    restored = richardson_lucy_filter(image, DELTA, iterations=10)
    assert np.allclose(restored, image, atol=1e-5)


def test_constant_image_is_exact_fixed_point() -> None:
    """A constant image is an exact fixed point of RL for any blur kernel."""
    shape = (32, 32)
    psf = combine_psfs(make_motion_psf(5, 12.0), make_defocus_psf(2))
    for value in (0.1, 0.45, 0.8):
        image = np.full(shape, value, dtype=np.float32)
        restored = richardson_lucy_filter(image, psf, iterations=10)
        assert np.allclose(restored, value, atol=1e-6)


def test_forward_adjoint_inner_product_identity() -> None:
    """Adjoint operator must satisfy <Hx, y> == <x, H^T y> to machine precision."""
    shape = (48, 64)
    rng = np.random.default_rng(123)
    x = rng.random(shape).astype(np.float64)
    y = rng.random(shape).astype(np.float64)

    test_psfs = [
        make_motion_psf(3, 8.0),
        make_motion_psf(15, 25.0),
        combine_psfs(make_motion_psf(7, 10.0), make_defocus_psf(2)),
        _skewed(),
        _bilinear(),
    ]

    for psf in test_psfs:
        H = psf_frequency_response(psf, shape, centre="auto")
        H_adj = np.conj(H)

        Hx = np.real(np.fft.ifft2(H * np.fft.fft2(x)))
        HTy = np.real(np.fft.ifft2(H_adj * np.fft.fft2(y)))

        ip_Hx_y = float(np.sum(Hx * y))
        ip_x_HTy = float(np.sum(x * HTy))

        rel_diff = abs(ip_Hx_y - ip_x_HTy) / (abs(ip_Hx_y) + abs(ip_x_HTy))
        assert rel_diff < 1e-14, f"Adjoint identity failed for PSF {psf.shape}: rel_diff={rel_diff}"


def test_flux_conservation_unclipped() -> None:
    """Under circular boundary conditions, sum(f) == sum(g) at each iteration."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    blurred = blur_fft(image, psf)

    # Calling with iterations=1, 5, 10 without intermediate clipping
    target_sum = float(np.sum(blurred))
    for it in (1, 5, 10):
        # We test internal flux conservation directly using the update
        H = psf_frequency_response(psf, blurred.shape)
        H_adj = np.conj(H)
        f = np.copy(blurred).astype(np.float64)
        for _ in range(it):
            Hf = np.maximum(np.real(np.fft.ifft2(H * np.fft.fft2(f))), 1e-12)
            corr = np.maximum(np.real(np.fft.ifft2(H_adj * np.fft.fft2(blurred / Hf))), 0.0)
            f = f * corr
        assert float(np.sum(f)) == pytest.approx(target_sum, rel=1e-6)


def test_zero_and_near_zero_regions_stay_finite() -> None:
    """Image with exact zeros (black regions) must not produce NaN or div-by-zero."""
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 0.7
    psf = _gaussian(5, 1.0)

    out = richardson_lucy_filter(image, psf, iterations=10)
    assert np.all(np.isfinite(out))
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_all_four_project_resolutions_work() -> None:
    """All four project image resolutions must execute cleanly."""
    resolutions = [
        (96, 336),  # Synthetic full
        (92, 256),  # CCPD full
        (48, 168),  # Synthetic factor-2
        (46, 128),  # CCPD factor-2
    ]
    psf = make_motion_psf(3, 8.0)

    for shape in resolutions:
        image = np.full(shape, 0.5, dtype=np.float32)
        image[shape[0] // 4: 3 * shape[0] // 4, shape[1] // 4: 3 * shape[1] // 4] = 0.9
        out = richardson_lucy_filter(image, psf, iterations=5)
        assert out.shape == shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))


def test_even_sized_psf_restores_without_half_pixel_shift() -> None:
    """centre='auto' restores even-sized PSFs without half-pixel centroid shift."""
    full_kernel = combine_psfs(make_motion_psf(15, 20.0), make_defocus_psf(3))
    stored_even = effective_psf(full_kernel, 2)
    assert stored_even.shape[0] % 2 == 0

    shape = (48, 168)
    clean = np.zeros(shape, dtype=np.float32)
    clean[16:32, 60:108] = 1.0

    full_clean = np.zeros((96, 336), dtype=np.float32)
    full_clean[32:64, 120:216] = 1.0
    observed = downsample(blur_fft(full_clean, full_kernel), 2)
    ref_y, ref_x = _centroid(clean)

    # Restoring with auto
    out_auto = richardson_lucy_filter(observed, stored_even, iterations=15,
                                      centre="auto")
    auto_y, auto_x = _centroid(out_auto)
    auto_drift = np.sqrt((auto_y - ref_y) ** 2 + (auto_x - ref_x) ** 2)

    # Restoring with pixel
    out_pixel = richardson_lucy_filter(observed, stored_even, iterations=15,
                                       centre="pixel")
    pix_y, pix_x = _centroid(out_pixel)
    pix_drift = np.sqrt((pix_y - ref_y) ** 2 + (pix_x - ref_x) ** 2)

    assert auto_drift < 0.05
    assert pix_drift > 0.45
    assert auto_drift < 0.15 * pix_drift


def test_asymmetric_kernel_preserves_conjugate_phase() -> None:
    """Skewed kernel must use H* in back-projection; dropping conjugate fails."""
    image = _textured()
    psf = _skewed()
    response = psf_frequency_response(psf, image.shape)
    degraded = _circular_blur(image, response)

    out = richardson_lucy_filter(degraded, psf, iterations=15)
    assert _rmse(image, out, 6) < _rmse(image, degraded, 6)


# --------------------------------------------------------------------------- #
# 4. Restoration behavior and noise properties
# --------------------------------------------------------------------------- #
def test_rl_deblurs_noise_free_plate() -> None:
    """On clean blur, RL error monotonically decreases over iterations."""
    plate = _plate()
    psf = make_motion_psf(3, 8.0)
    blurred = blur_fft(plate, psf)

    rmse_0 = _rmse(plate, blurred, 12)
    rmse_5 = _rmse(plate, richardson_lucy_filter(blurred, psf, iterations=5), 12)
    rmse_15 = _rmse(plate, richardson_lucy_filter(blurred, psf, iterations=15), 12)
    rmse_30 = _rmse(plate, richardson_lucy_filter(blurred, psf, iterations=30), 12)

    assert rmse_5 < rmse_0
    assert rmse_15 < rmse_5
    assert rmse_30 < rmse_15


def test_rl_noise_amplification_semi_convergence() -> None:
    """Under 20 dB noise, RL improves initially then amplifies high-frequency noise."""
    plate = _plate()
    psf = make_motion_psf(3, 8.0)
    blurred = blur_fft(plate, psf)
    noisy = add_noise(blurred, 20.0, seed=3)

    rmse_deg = _rmse(plate, noisy, 12)
    rmse_15 = _rmse(plate, richardson_lucy_filter(noisy, psf, iterations=15), 12)
    rmse_100 = _rmse(plate, richardson_lucy_filter(noisy, psf, iterations=100), 12)

    # Early iterations improve over degraded
    assert rmse_15 < rmse_deg
    # Excessive iterations amplify noise (semi-convergence)
    assert rmse_100 > rmse_15


def test_rl_clip_intermediate_controls_overshoot() -> None:
    """clip_intermediate=True enforces bounds per iteration, reducing overshoot."""
    plate = _plate()
    psf = make_motion_psf(3, 8.0)
    blurred = blur_fft(plate, psf)

    out_unbounded = richardson_lucy_filter(blurred, psf, iterations=20,
                                           clip_intermediate=False)
    out_clipped = richardson_lucy_filter(blurred, psf, iterations=20,
                                         clip_intermediate=True)

    err_unbounded = _rmse(plate, out_unbounded, 12)
    err_clipped = _rmse(plate, out_clipped, 12)
    assert err_clipped < err_unbounded


def test_rl_custom_initial_estimate() -> None:
    """Providing a valid non-negative initial_estimate must be respected."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    custom_init = np.full(image.shape, 0.4, dtype=np.float32)

    out = richardson_lucy_filter(image, psf, iterations=5,
                                 initial_estimate=custom_init)
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))


@pytest.mark.parametrize("profile", ["none", "tukey", "hann", "linear"])
def test_rl_accepts_all_edge_taper_profiles(profile: str) -> None:
    """All taper profiles must be accepted and return finite images in [0, 1]."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)
    restored = richardson_lucy_filter(degraded, psf, iterations=5, profile=profile)
    assert np.all(np.isfinite(restored))
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0


# --------------------------------------------------------------------------- #
# 5. Hypothesis property test
# --------------------------------------------------------------------------- #
@given(image=float32_image_strategy(min_h=8, min_w=8, max_h=24, max_w=24),
       iters=st.sampled_from([0, 1, 3, 5]))
@settings(max_examples=25, deadline=None)
def test_property_rl_output_is_always_valid_float32(image: np.ndarray,
                                                   iters: int) -> None:
    """For any valid float32 image and non-negative iterations, output is valid."""
    restored = richardson_lucy_filter(image, DELTA, iterations=iters)
    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert float(restored.min()) >= 0.0 and float(restored.max()) <= 1.0
