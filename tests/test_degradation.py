"""Tests for Step 2.2 -- FFT blur, block-mean downsampling and BSNR noise.

Covers unit checks (task 7.6) and properties 11-14 (tasks 7.4, 7.5).
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
    measure_bsnr,
    noise_sigma,
)
from license_plate_restoration.psf_builder import make_defocus_psf, make_motion_psf

DELTA = np.array([[1.0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# blur_fft
# --------------------------------------------------------------------------- #
def test_blur_fft_returns_same_shape_and_dtype() -> None:
    image = np.random.default_rng(0).random((40, 60), dtype=np.float32)
    out = blur_fft(image, make_motion_psf(3, 15.0))

    assert out.shape == image.shape
    assert out.dtype == np.float32


def test_blur_fft_delta_psf_is_identity() -> None:
    image = np.random.default_rng(1).random((32, 48), dtype=np.float32)
    out = blur_fft(image, DELTA)

    assert np.max(np.abs(out - image)) <= 1e-5


def test_blur_fft_centred_psf_does_not_shift() -> None:
    # A single bright pixel blurred by a symmetric disc keeps its centre of mass.
    image = np.zeros((41, 41), dtype=np.float32)
    image[20, 20] = 1.0

    out = blur_fft(image, make_defocus_psf(3.0))
    rows, cols = np.mgrid[0:41, 0:41]
    total = out.sum()

    assert abs((out * rows).sum() / total - 20.0) < 0.05
    assert abs((out * cols).sum() / total - 20.0) < 0.05


def test_blur_fft_preserves_mean_for_unit_sum_psf() -> None:
    image = np.random.default_rng(2).random((30, 30), dtype=np.float32)
    out = blur_fft(image, make_motion_psf(4, 33.0))

    # Mean is preserved up to the boundary effect of the reflective padding.
    assert abs(float(out.mean()) - float(image.mean())) < 5e-3


def test_blur_fft_reflect_no_wrap_artefacts() -> None:
    """Energy must not wrap from the right edge to the left edge."""
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 60] = 1.0                       # bright pixel near the right edge
    psf = make_motion_psf(5, 0.0)             # 11 px horizontal smear

    out = blur_fft(image, psf)
    assert np.abs(out[:, :5]).max() < 1e-6, "wrap-around artefact on the left edge"

    # Show that the check is meaningful: circular FFT convolution *does* wrap.
    psf_pad = np.zeros_like(image)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    psf_pad = np.roll(psf_pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    circular = np.real(np.fft.ifft2(np.fft.fft2(image) * np.fft.fft2(psf_pad)))
    assert np.abs(circular[:, :5]).max() > 1e-3


def test_blur_fft_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        blur_fft(np.zeros((10, 10), dtype=np.float32), np.zeros((0, 0), dtype=np.float32))
    with pytest.raises(ValueError):
        blur_fft(np.zeros((10, 10), dtype=np.float32), np.ones((41, 41), dtype=np.float32))


# --------------------------------------------------------------------------- #
# downsample
# --------------------------------------------------------------------------- #
def test_downsample_block_mean_values() -> None:
    image = np.array([[0.4, 0.6], [0.8, 1.0]], dtype=np.float32)
    out = downsample(image, 2)

    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(0.7, abs=1e-6)


def test_downsample_factor1_returns_copy() -> None:
    image = np.random.default_rng(3).random((8, 8), dtype=np.float32)
    out = downsample(image, 1)

    assert np.array_equal(out, image)
    assert out is not image                # a copy, not a view


def test_downsample_odd_shape_is_cropped() -> None:
    image = np.random.default_rng(4).random((7, 9), dtype=np.float32)
    out = downsample(image, 2)

    assert out.shape == (3, 4)
    # block (0, 0) is the mean of image[0:2, 0:2]
    assert out[0, 0] == pytest.approx(float(image[0:2, 0:2].mean()), abs=1e-6)


def test_downsample_rejects_bad_factor() -> None:
    with pytest.raises(ValueError):
        downsample(np.zeros((4, 4), dtype=np.float32), 0)
    with pytest.raises(ValueError):
        downsample(np.zeros((3, 3), dtype=np.float32), 4)


# --------------------------------------------------------------------------- #
# add_noise
# --------------------------------------------------------------------------- #
def test_noise_sigma_formula() -> None:
    image = np.random.default_rng(5).random((64, 64), dtype=np.float32)
    sigma = noise_sigma(image, 25.0)

    expected = np.sqrt(float(np.var(image.astype(np.float64))) / 10.0 ** 2.5)
    assert sigma == pytest.approx(expected, rel=1e-12)


def test_add_noise_range_and_dtype() -> None:
    image = np.random.default_rng(6).random((32, 32), dtype=np.float32)
    out = add_noise(image, 20.0, seed=1)

    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_add_noise_constant_image_is_unchanged() -> None:
    image = np.full((16, 16), 0.3, dtype=np.float32)
    out = add_noise(image, 20.0, seed=0)     # var = 0 -> sigma = 0

    assert np.array_equal(out, image)


def test_add_noise_on_saturated_plate_hits_target() -> None:
    """The saturation-compensated path must hit the target on real plates."""
    from license_plate_restoration.renderer import generate_plates

    plate = generate_plates(1, seed=1)[0][0]
    blurred = blur_fft(plate, make_motion_psf(6, 20.0))

    for target in (40.0, 30.0, 25.0, 20.0):
        measured = measure_bsnr(blurred, add_noise(blurred, target, seed=0))
        assert abs(measured - target) < 1.0


def test_literal_formula_biases_saturated_plates() -> None:
    """Documents *why* the compensation exists (bias without it exceeds 1 dB)."""
    from license_plate_restoration.renderer import generate_plates

    plate = generate_plates(1, seed=1)[0][0]
    blurred = blur_fft(plate, make_motion_psf(6, 20.0))

    naive = add_noise(blurred, 20.0, seed=0, saturation_compensation=False)
    assert measure_bsnr(blurred, naive) > 20.0 + 1.0


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 11: blur_fft with delta PSF is identity
@given(img=float32_image_strategy())
@settings(max_examples=100, deadline=None)
def test_blur_fft_delta_identity(img: np.ndarray) -> None:
    out = blur_fft(img, DELTA)

    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.max(np.abs(out - img)) <= 1e-5


# Feature: license-plate-restoration, Property 12: downsample shape invariants
@given(img=float32_image_strategy(min_h=4, min_w=4))
@settings(max_examples=100, deadline=None)
def test_downsample_factor1_identity(img: np.ndarray) -> None:
    assert np.array_equal(downsample(img, 1), img)


@given(img=float32_image_strategy(even=True))
@settings(max_examples=100, deadline=None)
def test_downsample_factor2_shape(img: np.ndarray) -> None:
    out = downsample(img, 2)

    assert out.shape == (img.shape[0] // 2, img.shape[1] // 2)
    assert out.dtype == np.float32


# Feature: license-plate-restoration, Property 13: add_noise output range and determinism
@given(img=float32_image_strategy(),
       bsnr=st.floats(10, 50),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=60, deadline=None)
def test_add_noise_range_and_determinism(img: np.ndarray, bsnr: float, seed: int) -> None:
    first = add_noise(img, bsnr, seed)
    second = add_noise(img, bsnr, seed)

    assert first.dtype == np.float32
    assert first.shape == img.shape
    assert first.min() >= 0.0 and first.max() <= 1.0
    assert np.array_equal(first, second)


# Feature: license-plate-restoration, Property 14: add_noise BSNR accuracy
@given(img=float32_image_strategy(),
       bsnr=st.floats(10, 50),
       seed=st.integers(0, 2**31 - 1))
@settings(max_examples=60, deadline=None)
def test_add_noise_bsnr_accuracy(img: np.ndarray, bsnr: float, seed: int) -> None:
    noisy = add_noise(img, bsnr, seed)
    measured = measure_bsnr(img, noisy)

    assert abs(measured - bsnr) <= 1.0
