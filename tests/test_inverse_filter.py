"""Tests for Step 4.2 -- frequency-domain inverse filtering.

Every case runs on small deterministic arrays or on a plate rendered by
step 1, so ``data/degraded.npz`` is never needed.  Thresholds are an order of
magnitude looser than the measured value, so a different BLAS rounding cannot
flip them, and each docstring quotes the measurement it pins.

What "exact" means here: with no noise, and a kernel whose response has no
null under the chosen floor, the filter reproduces the reference to the
float32 round-trip of the blur (~1e-6 RMS).  Where a null *does* exist the
output can only be finite, never exact -- that is the ill-conditioning the
docstrings are about, and it is asserted rather than hoped for.
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


# --------------------------------------------------------------------------- #
# 1. Identity
# --------------------------------------------------------------------------- #
def test_delta_psf_returns_the_image_unchanged() -> None:
    """H == 1 everywhere, so the division is a no-op -- bit for bit."""
    image = _textured()

    restored = inverse_filter(image, DELTA, profile="none")

    assert np.array_equal(restored, image)
    assert restored.dtype == np.float32
    assert restored.shape == image.shape


def test_default_arguments_run_the_taper_free_arm() -> None:
    """The default arm is `none`, so it must equal the explicit `none` run."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    assert DEFAULT_INVERSE_TAPER == "none"
    assert np.array_equal(inverse_filter(degraded, psf),
                          inverse_filter(degraded, psf, profile="none"))


# --------------------------------------------------------------------------- #
# 2./3. Noise-free inversion, odd-sided kernels
# --------------------------------------------------------------------------- #
def test_noise_free_blur_is_inverted_exactly() -> None:
    """The blur removes 1.3e-1 RMS; the filter puts back 4.7e-6 RMS."""
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    restored = inverse_filter(degraded, psf, profile="none")

    assert _rmse(image, degraded, 6) > 1e-2                 # it really blurred
    assert _rmse(image, restored, 6) < 1e-4                 # and it came back
    assert _rmse(image, degraded, 6) > 100 * _rmse(image, restored, 6)


def test_odd_motion_psf_restores_a_plate() -> None:
    """A `psf_builder` kernel is pixel-centred, so nothing drifts: 3.3e-5."""
    plate = _plate()
    psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))

    assert psf_centre_offset(psf) == (0.0, 0.0)
    restored = inverse_filter(blur_fft(plate, psf), psf, profile="none")

    assert _rmse(plate, restored, 12) < 1e-3


# --------------------------------------------------------------------------- #
# 4. Even-sided kernels and the half pixel
# --------------------------------------------------------------------------- #
def test_even_sized_psf_is_restored_without_a_half_pixel_shift() -> None:
    """`centre="auto"` lands on 20.3000 px; the pixel rule lands 0.5 px off.

    The kernel's mass sits at (K-1)/2, i.e. -0.5 px from the index ``K // 2``
    that the pixel rule puts on the FFT origin.  Restoring the same degraded
    image with both rules separates the two: auto is exact to 1e-6, the pixel
    rule moves the dot by a full half pixel.
    """
    reference = _blob()
    psf = _even_gaussian()
    response = psf_frequency_response(psf, reference.shape)
    degraded = _circular_blur(reference, response)

    assert psf_centre_offset(psf) == pytest.approx((-0.5, -0.5), abs=1e-3)

    auto = inverse_filter(degraded, psf, eps=1e-3, profile="none")
    pixel = inverse_filter(degraded, psf, eps=1e-3, profile="none",
                           centre="pixel")

    assert _rmse(reference, auto, 6) < 1e-4
    assert abs(_centroid(auto) - _centroid(reference)) < 0.01     # measured 0
    assert abs(_centroid(pixel) - _centroid(reference)) > 0.4     # measured .5
    assert _rmse(reference, pixel, 6) > 1e-2


def test_even_stored_kernel_keeps_the_decimated_plate_aligned() -> None:
    """The dataset's own factor-2 path: full kernel -> blur -> decimate.

    The stored `_psf` is `effective_psf(combined, 2)`, even-sided here, so
    this is the geometry step 4.3 will inherit.  Auto centres the plate on
    the reference to 0.003 px and halves the error of the pixel rule.
    """
    plate = _plate()
    full = combine_psfs(make_motion_psf(15, 20.0), make_defocus_psf(3))
    stored = effective_psf(full, 2)

    assert stored.shape[0] % 2 == 0                     # even, hence ambiguous
    observed = downsample(blur_fft(plate, full), 2)
    reference = downsample(plate, 2)

    auto = inverse_filter(observed, stored, eps=1e-2, profile="none")
    pixel = inverse_filter(observed, stored, eps=1e-2, profile="none",
                           centre="pixel")

    assert _rmse(reference, auto, 12) < 0.5 * _rmse(reference, pixel, 12)
    assert abs(_centroid(auto) - _centroid(reference)) < 0.05
    assert abs(_centroid(pixel) - _centroid(reference)) > 0.1


# --------------------------------------------------------------------------- #
# 5. Complex responses
# --------------------------------------------------------------------------- #
def test_complex_frequency_response_is_not_truncated() -> None:
    """Dividing by the real part only is not a filter -- it is a 2.7e5 error.

    An asymmetric kernel gives a response with |Im H| up to 0.766, so a
    helper that silently cast the spectrum to float would apply a completely
    different operator.  The filter must keep the complex quotient.
    """
    reference = _textured()
    psf = _skewed()
    response = psf_frequency_response(psf, reference.shape)

    assert np.iscomplexobj(response)
    assert float(np.abs(response.imag).max()) > 0.1

    degraded = _circular_blur(reference, response)
    restored = inverse_filter(degraded, psf, profile="none")

    truncated = np.real(np.fft.ifft2(
        np.fft.fft2(degraded.astype(np.float64)) / np.real(response)))

    assert _rmse(reference, restored, 6) < 0.1              # measured 1.4e-2
    assert _rmse(reference, truncated, 6) > 100.0           # measured 2.7e+05


# --------------------------------------------------------------------------- #
# 6. Near-zero responses
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("eps", [DEFAULT_EPS, 1e-6, 1e-3])
def test_near_zero_frequency_response_still_returns_an_image(eps: float) -> None:
    """`|H| == 0` at 79 coefficients: bounded, finite, in range -- not NaN."""
    reference = _textured()
    psf = _bilinear()
    response = psf_frequency_response(psf, reference.shape)
    degraded = _circular_blur(reference, response)

    assert int(near_zero_mask(response, eps=DEFAULT_EPS).sum()) > 0

    restored = inverse_filter(degraded, psf, eps=eps, profile="none")

    assert np.all(np.isfinite(restored))
    assert restored.dtype == np.float32
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_a_noisy_notch_does_not_produce_non_finite_output() -> None:
    """The cost of a small floor is accuracy, never NaN or inf."""
    reference = _textured()
    psf = _gaussian(5, 1.0)
    noisy = add_noise(blur_fft(reference, psf), 20.0, seed=1)

    for eps in (DEFAULT_EPS, 1e-3, 1e-1):
        restored = inverse_filter(noisy, psf, eps=eps, profile="none")
        assert np.all(np.isfinite(restored))
        assert restored.min() >= 0.0 and restored.max() <= 1.0


# --------------------------------------------------------------------------- #
# 7. The denominator floor
# --------------------------------------------------------------------------- #
def test_floor_trades_restoration_against_noise_amplification() -> None:
    """The trade-off, measured in both directions.

    Clean input: the tight floor is the accurate one (4.7e-6 vs 4.0e-2),
    because the loose floor throws away every coefficient below 1e-2.
    At 20 dB BSNR the ordering reverses (9.2e-2 vs 4.4e-1), because the tight
    floor multiplies the noise riding on the notch by 1 / |H|.  Neither value
    is "correct" -- that is why the floor is a parameter.
    """
    reference = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(reference, psf)

    tight_clean = inverse_filter(degraded, psf, eps=1e-12, profile="none")
    loose_clean = inverse_filter(degraded, psf, eps=1e-2, profile="none")
    assert _rmse(reference, tight_clean, 6) < _rmse(reference, loose_clean, 6)

    noisy = add_noise(degraded, 20.0, seed=1)
    tight_noisy = inverse_filter(noisy, psf, eps=1e-12, profile="none")
    loose_noisy = inverse_filter(noisy, psf, eps=1e-1, profile="none")
    assert (_rmse(reference, loose_noisy, 6)
            < 0.5 * _rmse(reference, tight_noisy, 6))


def test_floor_parameter_reaches_the_maths() -> None:
    image = _textured()
    psf = _gaussian(5, 1.0)
    degraded = blur_fft(image, psf)

    assert not np.array_equal(inverse_filter(degraded, psf, eps=1e-12),
                              inverse_filter(degraded, psf, eps=1e-1))


@pytest.mark.parametrize("eps", [0.0, -1e-12, np.nan, np.inf, np.float64(0.0)])
def test_eps_must_be_positive_and_finite(eps) -> None:
    with pytest.raises(ValueError, match="eps must be > 0"):
        inverse_filter(_textured(), DELTA, eps=eps)


@pytest.mark.parametrize("eps", ["1e-3", None, object()])
def test_eps_must_be_a_number(eps) -> None:
    with pytest.raises(ValueError, match="eps must be a number"):
        inverse_filter(_textured(), DELTA, eps=eps)


# --------------------------------------------------------------------------- #
# 8. Invalid inputs
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
def test_inverse_filter_rejects_invalid_input(image, psf, kwargs: dict,
                                              message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inverse_filter(image, psf, **kwargs)


# --------------------------------------------------------------------------- #
# 9./10. Output convention, both taper arms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("profile", ["none", "tukey"])
@pytest.mark.parametrize("eps", [1e-12, 1e-2])
def test_output_keeps_the_image_convention(profile: str, eps: float) -> None:
    plate = _plate()
    psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))
    noisy = add_noise(blur_fft(plate, psf), 20.0, seed=2)

    restored = inverse_filter(noisy, psf, eps=eps, profile=profile)

    assert restored.shape == plate.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0 and restored.max() <= 1.0


def test_taper_choice_only_changes_the_border() -> None:
    """Tapering is a pre-multiplication, so it survives into the estimate.

    With a delta PSF the estimate *is* the (windowed) input: identical on the
    plateau, darkened by the window at the border, and the input itself when
    the taper is off -- to round-off (5e-16 measured), because the identity
    still makes one FFT round-trip and a black (0.0) pixel can come back as
    +3e-17.
    """
    plate = _plate()
    width = int(round(DEFAULT_TAPER_FRACTION * min(plate.shape)))

    plain = inverse_filter(plate, DELTA, profile="none")
    windowed = inverse_filter(plate, DELTA, profile="tukey")

    assert np.allclose(plain, plate, atol=1e-6)
    assert np.allclose(windowed[width:-width, width:-width],
                       plate[width:-width, width:-width], atol=1e-6)
    assert windowed[0, 0] < 0.01 < 0.4 < plate[0, 0]
    assert not np.array_equal(windowed, plain)
    assert np.abs(plain.astype(np.float64) - plate).max() < 1e-6


# --------------------------------------------------------------------------- #
# Property test
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 21: inverse filtering always
# returns a finite float32 image inside [0, 1], whatever the input pair is.
@given(image=float32_image_strategy(min_h=8, min_w=8, max_h=18, max_w=18),
       seed=st.integers(0, 2 ** 31 - 1),
       eps=st.sampled_from([1e-12, 1e-6, 1e-2, 1e-1]))
@settings(max_examples=40, deadline=None)
def test_property_output_is_always_a_valid_image(image: np.ndarray, seed: int,
                                                 eps: float) -> None:
    restored = inverse_filter(image, _random_kernel(seed), eps=eps,
                              profile="none")

    assert restored.shape == image.shape
    assert restored.dtype == np.float32
    assert np.all(np.isfinite(restored))
    assert restored.min() >= 0.0 and restored.max() <= 1.0
