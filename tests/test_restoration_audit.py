"""Tests for phase 4.1.1 -- the technical audit of the restoration foundation.

Three questions were open after phase 4.1, and each one gets its own section:

1. do the numerical helpers handle a **complex** ``H(u, v)`` without losing the
   imaginary part (an inverse filter divides by it);
2. does an **even-sized** stored kernel introduce a measurable spatial
   displacement, and does ``centre="auto"`` remove it;
3. how does the default **edge taper** compare with no taper and with the
   conventional alternatives.

The even-kernel experiments run the real forward model -- ``blur_fft`` then
``downsample``, exactly as ``run_degradation_grid`` builds the dataset -- so
they measure the pipeline rather than a stand-in.  Nothing here regenerates or
modifies the stored kernel.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from license_plate_restoration.degradation import blur_fft, downsample, effective_psf
from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)
from license_plate_restoration.restoration import (
    DEFAULT_EPS,
    DEFAULT_TAPER_FRACTION,
    TAPER_PROFILES,
    clip_unit_interval,
    edge_taper_window,
    near_zero_mask,
    psf_centre_offset,
    psf_frequency_response,
    safe_divide,
    taper_edges,
    validate_image,
    validate_psf,
)


# --------------------------------------------------------------------------- #
# Local helpers
# --------------------------------------------------------------------------- #
def _circular_convolve(image, response: np.ndarray) -> np.ndarray:
    """Apply a frequency response the way the restoration stages will."""
    spectrum = np.fft.fft2(np.asarray(image, dtype=np.float64))
    return np.real(np.fft.ifft2(spectrum * response))


def _centroid(image: np.ndarray) -> tuple[float, float]:
    """Intensity centroid ``(row, col)`` of a non-negative image."""
    rows, cols = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    total = float(image.sum())
    return float((image * rows).sum() / total), float((image * cols).sum() / total)


def _random_psf(seed: int, size: int = 5) -> np.ndarray:
    """Random normalised kernel of the requested side (odd or even)."""
    raw = np.random.default_rng(seed).random((size, size))
    return (raw / raw.sum()).astype(np.float32)


def _stored_kernel(length: int, radius: int, angle: float = 20.0) -> tuple:
    """``(combined, stored)`` kernel pair for one grid point."""
    combined = combine_psfs(make_motion_psf(length, angle),
                            make_defocus_psf(radius))
    return combined, effective_psf(combined, 2)


def _reblur_displacement(combined, factor: int,
                         mode: str) -> tuple[float, float]:
    """Centroid error of a re-blurred reference, in pixels.

    ``observed`` is what the dataset stores: blur the plate, then decimate.
    ``recon`` pushes the decimated *reference* back through the stored kernel.
    With the right centring convention the two land on the same pixel, so the
    returned offset is exactly the displacement the response introduces.
    """
    clean = np.zeros((96, 336), dtype=np.float32)
    clean[24:72, 80:256] = np.random.default_rng(5).random(
        (48, 176), dtype=np.float32)
    observed = downsample(blur_fft(clean, combined), factor)
    reference = downsample(clean, factor)
    stored = effective_psf(combined, factor)

    recon = _circular_convolve(
        reference, psf_frequency_response(stored, observed.shape, centre=mode))
    observed_row, observed_col = _centroid(observed)
    recon_row, recon_col = _centroid(recon)
    return recon_row - observed_row, recon_col - observed_col


def _profile(values: np.ndarray) -> np.ndarray:
    """The 1-D profile of a separable 2-D window, read along its middle row."""
    array = np.asarray(values, dtype=np.float64)
    return array[array.shape[0] // 2]


# --------------------------------------------------------------------------- #
# 1. Complex-valued numerical operations
# --------------------------------------------------------------------------- #
def test_safe_divide_keeps_complex_quotients() -> None:
    """A complex numerator and denominator divide exactly, as complex."""
    numerator = np.array([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j])
    denominator = np.array([0.5 + 0.5j, 2.0 - 1.0j, -1.5 + 0.0j])

    quotient = safe_divide(numerator, denominator)

    assert quotient.dtype == np.complex128
    assert np.array_equal(quotient, numerator / denominator)


def test_safe_divide_preserves_the_phase_of_a_floored_coefficient() -> None:
    """The floor is applied to the modulus, not to the real axis.

    A coefficient below ``eps`` is replaced by a vector of length ``eps`` that
    points the same way, so ``phase(q) == phase(a) - phase(b)`` still holds --
    which is what stops a floor from rotating the restored image.
    """
    phase = 0.7
    numerator = np.array([2.0 + 3.0j])
    denominator = np.array([1e-20 * np.exp(1j * phase)])   # |b| << eps

    quotient = safe_divide(numerator, denominator)

    assert np.angle(quotient[0]) == pytest.approx(np.angle(numerator[0]) - phase,
                                                  abs=1e-12)
    assert abs(quotient[0]) == pytest.approx(abs(numerator[0]) / DEFAULT_EPS)


def test_safe_divide_bounds_a_degenerate_complex_quotient() -> None:
    numerator = np.array([1.0 + 1.0j, 0.0 + 0.0j, -2.0 + 1.0j])
    denominator = np.array([0.0 + 0.0j, 1e-300 * np.exp(2.1j), 1e-30 - 1e-30j])

    quotient = safe_divide(numerator, denominator)

    assert np.all(np.isfinite(quotient))
    assert np.all(np.abs(quotient) <= np.abs(numerator) / DEFAULT_EPS * (1 + 1e-12))
    # A zero coefficient has no phase to keep: the floor is +eps + 0j.
    assert quotient[0] == pytest.approx((1.0 + 1.0j) / DEFAULT_EPS)


def test_safe_divide_never_truncates_a_complex_operand() -> None:
    """The helpers must not fall back to NumPy's real cast.

    ``np.asarray(complex_array, dtype=float64)`` only *warns*
    (``ComplexWarning``) and drops the imaginary part, so a complex spectrum
    would silently become a different operator.  Any warning is an error here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quotient = safe_divide(np.array([1.0 + 2.0j]), np.array([0.5 + 0.5j]))
        mask = near_zero_mask(np.array([1e-20 + 1e-20j]))

    assert quotient[0] == pytest.approx((1.0 + 2.0j) / (0.5 + 0.5j))
    assert quotient[0].imag != 0.0
    assert bool(mask[0])          # |1e-20 + 1e-20j| = 1.4e-20 <= eps


def test_safe_divide_accepts_a_real_numerator_over_a_complex_denominator() -> None:
    quotient = safe_divide(np.array([1.0, 2.0]), np.array([1.0j, 2.0j]))

    assert quotient.dtype == np.complex128
    assert np.array_equal(quotient, np.array([1.0, 2.0]) / np.array([1.0j, 2.0j]))


def test_near_zero_mask_measures_the_modulus_of_a_complex_spectrum() -> None:
    """Near-zero means small *real and imaginary part together*."""
    denominator = np.array([1e-15 + 0.0j,       # modulus 1e-15  -> degenerate
                            1e-7 + 1e-7j,       # modulus 1.4e-7 -> fine
                            1e-13 + 1e-13j,     # modulus 1.4e-13 -> degenerate
                            0.0 + 0.0j])

    mask = near_zero_mask(denominator)

    assert np.array_equal(mask, np.array([True, False, True, True]))
    # Clamping a complex coefficient keeps its direction: the floored
    # denominator points at +45 degrees, so the quotient points at -45.
    floored = safe_divide(np.ones(4), denominator)
    assert np.angle(floored[2]) == pytest.approx(-np.pi / 4, abs=1e-12)
    assert np.angle(floored[3]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("call", [
    lambda: validate_image(np.zeros((3, 3), dtype=np.complex128)),
    lambda: validate_psf(np.full((3, 3), 1.0 / 9.0, dtype=np.complex128)),
    lambda: clip_unit_interval(np.array([0.5 + 0.1j])),
    lambda: taper_edges(np.zeros((6, 6), dtype=np.complex128)),
])
def test_real_only_helpers_reject_complex_input(call) -> None:
    """Images and kernels are real by contract; truncating them would be silent."""
    with pytest.raises(ValueError, match="must be real-valued"):
        call()


def test_inverse_filter_quotient_of_real_data_stays_hermitian() -> None:
    """The check that an inverse filter can use ``safe_divide`` unchanged.

    For a real image and a real kernel both ``G`` and ``H`` are Hermitian, so
    ``G / H`` must be Hermitian too and its inverse transform real.  That only
    holds if the helpers carry the imaginary part through the division.
    """
    image = np.zeros((24, 32), dtype=np.float32)
    image[4:-4, 4:-4] = np.random.default_rng(7).random((16, 24), dtype=np.float32)
    psf = combine_psfs(make_motion_psf(3, 20.0), make_defocus_psf(2))

    spectrum = np.fft.fft2(image.astype(np.float64))
    response = psf_frequency_response(psf, image.shape)
    quotient = safe_divide(spectrum, response)

    assert quotient.dtype == np.complex128
    # H(0, 0) == 1, so the DC coefficient (the *sum*, which fft2 returns) is
    # preserved: a unit DC gain is what keeps the mean brightness intact.
    # 1e-6 leaves room for the kernel's float32 normalisation.
    assert quotient[0, 0].real == pytest.approx(float(image.sum()), rel=1e-6)
    assert quotient[0, 0].imag == pytest.approx(0.0, abs=1e-9)

    restored = np.fft.ifft2(quotient)
    assert np.max(np.abs(restored.imag)) <= 1e-9 * np.max(np.abs(restored))


# --------------------------------------------------------------------------- #
# 2. Even-sized PSF handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length, radius", [(15, 0), (5, 0), (15, 3), (5, 3),
                                            (11, 1), (3, 3)])
def test_psf_centre_offset_follows_the_kernel_parity(length: int,
                                                     radius: int) -> None:
    """Odd kernels are pixel-centred, even ones sit half a pixel off.

    ``effective_psf`` decimates with a 2x2 box, so the stored mass lands on
    ``(K - 1) / 2``: an integer for an odd side, exactly between two samples
    for an even one.
    """
    combined, stored = _stored_kernel(length, radius)
    offset_rows, offset_cols = psf_centre_offset(stored)

    assert stored.shape[0] == length + radius          # documented size rule
    if stored.shape[0] % 2 == 0:
        assert offset_rows == pytest.approx(-0.5, abs=1e-3)
        assert offset_cols == pytest.approx(-0.5, abs=1e-3)
    else:
        assert offset_rows == 0.0 and offset_cols == 0.0


@pytest.mark.parametrize("length, radius", [(15, 3), (5, 3), (11, 1)])
def test_auto_centre_removes_the_stored_kernel_displacement(length: int,
                                                            radius: int) -> None:
    """Controlled experiment against the dataset's own forward model.

    Blurring the plate and decimating (what the dataset does) puts the blurred
    content on the decimated grid.  Re-blurring the decimated *reference* must
    land on the same pixel; with the plain index rule it lands half a pixel
    up/left, which the linear phase ramp of ``centre="auto"`` removes.
    """
    combined, stored = _stored_kernel(length, radius)
    assert stored.shape[0] % 2 == 0                    # the case under audit

    auto_rows, auto_cols = _reblur_displacement(combined, 2, "auto")
    pixel_rows, pixel_cols = _reblur_displacement(combined, 2, "pixel")

    assert abs(auto_rows) < 0.02 and abs(auto_cols) < 0.02
    assert pixel_rows == pytest.approx(-0.5, abs=0.02)
    assert pixel_cols == pytest.approx(-0.5, abs=0.02)


@pytest.mark.parametrize("length, radius", [(15, 0), (15, 3), (5, 3)])
def test_full_resolution_kernels_have_no_displacement(length: int,
                                                      radius: int) -> None:
    """At factor 1 the kernel stays odd, so both conventions must agree."""
    combined = combine_psfs(make_motion_psf(length, 20.0),
                            make_defocus_psf(radius))

    auto_rows, auto_cols = _reblur_displacement(combined, 1, "auto")
    pixel_rows, pixel_cols = _reblur_displacement(combined, 1, "pixel")

    for offset in (auto_rows, auto_cols, pixel_rows, pixel_cols):
        assert abs(offset) < 0.02
    assert np.array_equal(psf_frequency_response(combined, (96, 336)),
                          psf_frequency_response(combined, (96, 336),
                                                 centre="pixel"))


def test_centre_offset_tolerance_separates_the_two_families() -> None:
    """The threshold sits in the empty gap between pixel- and half-pixel-centred.

    Measured over the grid's own ranges: every odd kernel built by
    ``psf_builder`` is within 0.005 px of its middle sample, every even stored
    kernel within 0.0025 px of a full half pixel.  Nothing lands in between,
    which is why ``CENTRE_OFFSET_TOL`` (0.05 px) can snap the first family to
    zero without ever touching the second.
    """
    for length in range(0, 22, 6):
        for radius in (0, 3):
            for angle in (0.0, 8.0, 45.0):
                combined = combine_psfs(make_motion_psf(length, angle),
                                        make_defocus_psf(radius))
                assert max(abs(v) for v in psf_centre_offset(combined)) < 0.05

                stored = effective_psf(combined, 2)
                offset = psf_centre_offset(stored)
                if stored.shape[0] % 2 == 0:
                    assert offset == pytest.approx((-0.5, -0.5), abs=0.05)
                else:
                    assert max(abs(v) for v in offset) < 0.05


def test_centre_correction_is_phase_only() -> None:
    """The ramp changes phase, never magnitude, and keeps the DC gain at 1."""
    _, stored = _stored_kernel(15, 3)
    auto = psf_frequency_response(stored, (48, 168))
    pixel = psf_frequency_response(stored, (48, 168), centre="pixel")

    assert not np.allclose(auto, pixel)                # the ramp really is applied
    assert np.allclose(np.abs(auto), np.abs(pixel), atol=1e-12)
    assert auto.dtype == np.complex128
    assert auto[0, 0].real == pytest.approx(1.0, abs=1e-6)
    assert auto[0, 0].imag == pytest.approx(0.0, abs=1e-9)


def test_response_rejects_an_unknown_centre_mode() -> None:
    with pytest.raises(ValueError, match="centre must be one of"):
        psf_frequency_response(make_motion_psf(3, 0.0), (32, 32), centre="middle")


def test_psf_centre_offset_rejects_an_invalid_psf() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        psf_centre_offset(np.full((5, 5), 0.08))
    assert psf_centre_offset(np.array([[1.0]], dtype=np.float32)) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# 3. Edge tapering evaluation
# --------------------------------------------------------------------------- #
def test_hann_profile_has_no_plateau() -> None:
    """Hann attenuates the whole image, so it cannot be the plate default."""
    window = edge_taper_window((96, 336), profile="hann")

    assert window.min() == 0.0
    assert window.max() < 1.0                          # even the centre is faded
    assert np.count_nonzero(window == 1.0) == 0        # not one untouched pixel
    assert float(np.mean(1.0 - window)) > 0.5          # measured: 75 % loss


def test_linear_profile_keeps_the_plateau_but_jumps_in_slope() -> None:
    """Tukey and the triangle agree inside the plateau, differ where it ends."""
    shape = (96, 96)
    tukey = edge_taper_window(shape)
    linear = edge_taper_window(shape, profile="linear")
    width = int(round(DEFAULT_TAPER_FRACTION * min(shape)))

    assert np.array_equal(tukey[width:-width, width:-width],
                          linear[width:-width, width:-width])
    assert np.array_equal(tukey[width:-width, width:-width],
                          np.ones((96 - 2 * width, 96 - 2 * width),
                                  dtype=np.float32))

    slope_jump = np.abs(np.diff(_profile(linear), 2)).max()
    tukey_jump = np.abs(np.diff(_profile(tukey), 2)).max()
    assert slope_jump == pytest.approx(1.0 / width, abs=1e-6)
    assert tukey_jump < slope_jump                      # smoother ramp


@pytest.mark.parametrize("kwargs, message", [
    ({"profile": "blackman"}, "profile must be one of"),
    ({"profile": "hann", "width": 8}, "width is not used"),
    ({"profile": "hann", "fraction": 0.2}, "fraction is not used"),
    ({"profile": "none", "width": 8}, "width is not used"),
    ({"profile": "none", "fraction": 0.2}, "fraction is not used"),
])
def test_taper_rejects_profile_misuse(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        edge_taper_window((64, 64), **kwargs)
    with pytest.raises(ValueError, match=message):
        taper_edges(np.full((64, 64), 0.5, dtype=np.float32), **kwargs)


def test_none_profile_is_the_identity_window() -> None:
    """The untapered control arm: exactly the image back, no rounding drift."""
    from license_plate_restoration.renderer import generate_plates

    plate, _ = generate_plates(1, seed=7)[0]
    window = edge_taper_window(plate.shape, profile="none")
    tapered = taper_edges(plate, profile="none")

    assert np.array_equal(window, np.ones(plate.shape, dtype=np.float32))
    assert np.array_equal(tapered, plate)


@pytest.mark.parametrize("profile", TAPER_PROFILES)
@pytest.mark.parametrize("shape", [(2, 2), (5, 9), (46, 128), (92, 256)])
def test_taper_profiles_preserve_shape_and_range(profile: str,
                                                 shape: tuple[int, int]) -> None:
    tapered = taper_edges(np.full(shape, 0.7, dtype=np.float32), profile=profile)

    assert tapered.shape == shape
    assert tapered.dtype == np.float32
    assert np.all(np.isfinite(tapered))
    assert tapered.min() >= 0.0 and tapered.max() <= 1.0


def test_default_taper_ramp_leaves_the_glyph_strokes_untouched() -> None:
    """What the ramp costs on a real plate: the frame band, not the characters.

    The plates carry an 8 px grey margin and the default ramp is 9-10 px wide,
    so it does reach the two dark frame rows at the edge of the drawing area --
    but every character stroke sits inside the plateau and is unchanged.
    """
    from license_plate_restoration.renderer import generate_plates

    plate, _ = generate_plates(2, seed=11)[0]
    window = edge_taper_window(plate.shape)
    width = int(round(DEFAULT_TAPER_FRACTION * min(plate.shape)))

    ink = plate < 0.5
    inside = np.zeros_like(ink)
    inside[width:-width, width:-width] = True
    glyph, frame = ink & inside, ink & ~inside

    assert glyph.any() and frame.any()
    assert np.all(window[glyph] == 1.0)                 # strokes untouched
    assert window[frame].min() < 0.9                    # frame band attenuated


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 19: the sub-pixel centre
# correction is phase-only, so it cannot change |H| or the DC gain.
@given(seed=st.integers(0, 2**31 - 1), size=st.integers(3, 8),
       shape=st.tuples(st.integers(9, 24), st.integers(9, 24)))
@settings(max_examples=60, deadline=None)
def test_property_auto_centre_is_phase_only(seed: int, size: int,
                                            shape: tuple[int, int]) -> None:
    kernel = _random_psf(seed, size)

    auto = psf_frequency_response(kernel, shape, centre="auto")
    pixel = psf_frequency_response(kernel, shape, centre="pixel")

    assert np.allclose(np.abs(auto), np.abs(pixel), atol=1e-12)
    assert auto[0, 0].real == pytest.approx(1.0, abs=1e-6)
    assert auto[0, 0].imag == pytest.approx(0.0, abs=1e-9)
