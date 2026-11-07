"""Step 4.1/4.2 -- restoration: validation, response, taper, inverse filter.

This module holds the pieces that *all four* restoration algorithms share plus
the first of them:

* :func:`validate_image` / :func:`validate_psf` / :func:`validate_image_shape`
  -- the integration contract, checked once at the entry point of every
  restoration call;
* :func:`psf_frequency_response` -- pad the PSF, move its centre to the FFT
  origin, transform it;
* :func:`edge_taper_window` / :func:`taper_edges` -- suppress the circular
  wrap-around that a plain FFT deconvolution would fold into the plate;
* :func:`safe_divide` / :func:`near_zero_mask` / :func:`clip_unit_interval`
  -- the small numerical helpers that keep the later stages finite and inside
  the ``[0, 1]`` image convention;
* :func:`inverse_filter` -- step 4.2's frequency-domain inverse filter,
  ``\\hat{F} = G / H`` with a floored denominator, built entirely from the four
  pieces above.

Wiener, constrained least squares and Richardson--Lucy are deliberately *not*
here (steps 4.3-4.5).  They all reduce to "build ``H`` with
:func:`psf_frequency_response`, deconvolve the tapered image, clip the
result", which is exactly the surface this module exposes.

Inherited conventions (do not re-derive, do not change)
-------------------------------------------------------
* Images are grayscale ``float32`` in ``[0, 1]`` -- 0 is black (ink).
* PSF kernels are centred on the middle pixel and sum to 1.  A kernel built by
  ``psf_builder`` is also odd-sized, but a **stored** kernel is not always:
  ``run_degradation_grid`` writes ``effective_psf(combined, factor)``, whose
  size at ``factor == 2`` is ``(L + r, L + r)`` and therefore *even* whenever
  ``L + r`` is even -- 556 of 1050 factor-2 outputs in a run with the grid's
  own parameter ranges.  Those kernels also carry ``~1e-17`` negative lobes
  from the FFT round-off inside ``effective_psf``.  So validation has to
  tolerate an even side and round-off negatives, or step 4 would reject half
  of the low-resolution dataset it is supposed to consume.
* ``<key>_psf`` in ``degraded.npz`` is already resolution-matched to its
  ``<key>_img``.  Consume it **as-is**; at ``downsample_factor == 2`` it is
  already about half as wide as the full-resolution kernel.
* Image sizes vary: ``(96, 336)`` synthetic, ``(92, 256)`` CCPD, halved to
  ``(48, 168)`` / ``(46, 128)`` at factor 2.  No dimension is hard-coded here.
* ``make_motion_psf(length, angle)`` takes a **half**-length: the kernel is
  ``2 * length + 1`` wide, so ``length=15`` is a 31 px smear, and the grid
  maximum ``length=21`` is a 43 px smear.  Verified against the current
  implementation (``psf_builder.make_motion_psf`` samples the line from
  ``-length`` to ``+length``) and pinned by
  ``tests/test_psf_builder.py::test_motion_psf_shape_follows_length``.  The
  hand-over message that describes ``length`` as the full smear extent
  contradicts the code; the half-length reading is the verified one.

Audit findings (phase 4.1.1)
---------------------------
Three properties of the foundation were measured against the real pipeline
before any algorithm was written on top of it.

* **Complex spectra.**  ``H`` is complex, so every numerical helper accepts
  real *or* complex input and keeps the imaginary part; a complex array handed
  to a real-only helper (an image, a PSF, a clipped result) raises instead of
  being truncated by NumPy's ``ComplexWarning`` cast.  ``safe_divide`` floors
  a complex denominator by its modulus and rotates the floor onto the
  denominator's own phase, so an inverse filter keeps
  ``phase(G) - phase(H)``.
* **Even kernels.**  ``effective_psf(combined, 2)`` puts the kernel's centre of
  mass at ``(K - 1) / 2``, i.e. *between* two pixels when ``K`` is even.
  Rolling by ``K // 2`` therefore leaves the mass half a pixel off the FFT
  origin, and re-blurring the reference lands 0.490-0.503 px up/left of the
  stored image (measured across the grid's parameter range).
  :func:`psf_frequency_response` defaults to ``centre="auto"``, which removes
  exactly that offset with a unit-modulus linear phase ramp; no stored kernel
  is modified or regenerated.  Odd kernels are unaffected -- their mass sits on
  ``K // 2`` already, so their response is bit-identical under both modes.
* **Taper.**  Every plate in ``plates.npz`` carries a uniform 8 px grey margin,
  so opposite edges are identical (measured seam jump 0.0000 on all 1050
  images) and the only wrap step is noise (0.118 at 20 dB BSNR).  The ramp at
  the default width leaves the glyph strokes at ``W == 1`` and attenuates only
  the outermost dark frame rows of the drawing area.  Measured consequence: on
  this dataset the untapered arm is the *more* accurate one (the margin already
  makes the wrap continuous, so the ramp only adds ~9e-3 of border model error),
  while a 43 px kernel -- wider than the margin -- is the case the ramp is for.
  ``profile="none"`` is therefore a first-class arm alongside
  ``"tukey"`` (default), ``"hann"`` and ``"linear"``; step 4.2 runs the ablation
  on restored PSNR.  Numbers: ``docs/restoration_foundation.md``, section 5.4.

Failure policy
--------------
Nothing here swallows a numerical failure.  Invalid images and kernels raise
``ValueError`` with the offending shape/range/count in the message; a diverged
restoration is caught by :func:`clip_unit_interval`, which refuses non-finite
input instead of quietly emitting a NaN image; and the only repair that *is*
silent -- clamping a near-zero denominator in :func:`safe_divide` -- is
observable through :func:`near_zero_mask` so a caller can count and report it.

Library rules: NumPy only for arrays, FFT and arithmetic.  No SciPy, no
scikit-image, no deep-learning frame, no OpenCV, no black-box restoration call.
"""

from __future__ import annotations

import numpy as np

# PSF normalisation tolerance, matching ``psf_builder.psf_is_valid``.
PSF_SUM_TOL = 1e-5

# Negative-weight tolerance for a PSF.  ``effective_psf`` convolves through
# ``blur_fft``, so the factor-2 kernels it writes hold ~1e-17 sized negative
# lobes from float round-off; a genuinely negative kernel is many orders of
# magnitude larger than this floor.
PSF_NEGATIVE_TOL = 1e-6

# Slack allowed on the [0, 1] image contract, so a float32 round-trip does not
# trip the range check.
IMAGE_RANGE_TOL = 1e-4

# Default edge-taper width as a fraction of the image's shorter side: 10 % of
# 92 px is 9 px per side, which covers a 43 px-motion kernel's worst ringing
# without touching the glyphs of a plate.
DEFAULT_TAPER_FRACTION = 0.1

# Edge-taper profiles.  ``tukey`` is the raised-cosine ramp with a unity
# plateau (the default); ``hann`` is the full-axis cosine window with no
# plateau, ``linear`` the triangular ramp and ``none`` the identity (no
# windowing at all).  Every profile is a named arm of the taper comparison in
# ``docs/restoration_foundation.md`` -- step 4.2 selects one per algorithm
# instead of hand-rolling the ablation.
TAPER_PROFILES = ("none", "tukey", "hann", "linear")
DEFAULT_TAPER_PROFILE = "tukey"

# Taper arm for :func:`inverse_filter`.  The global default above stays
# ``tukey`` (the safe choice for input without a margin); this one follows the
# measurement on *this* dataset.  Every plate carries a uniform 8 px margin, so
# the circular model is already exact (wrap step 0.0000), and a ramp can then
# only add model error exactly where ``|H|`` is small -- which the inverse
# filter multiplies.  Measured on a rendered plate, clean input: 7x7 motion
# kernel 3.3e-5 RMS untapered vs 0.70 tapered; 51x51 (wider than the margin,
# the case the ramp was meant for) 0.124 vs 0.647 at eps=1e-2.  Section 12.3 of
# ``docs/restoration_foundation.md``.
DEFAULT_INVERSE_TAPER = "none"

# Centre convention for :func:`psf_frequency_response`.  ``auto`` removes the
# sub-pixel gap between the kernel's centre of mass and index ``size // 2``
# (0 px for an odd kernel, -0.5 px for an even stored one); ``pixel`` is the
# plain "index size // 2 is the origin" rule.
CENTRE_MODES = ("auto", "pixel")
DEFAULT_CENTRE_MODE = "auto"

# A centre offset below this many pixels is treated as pixel-centred.  The two
# populations are far apart and do not overlap: measured over 792 kernels,
# every odd kernel built by psf_builder is within 0.0049 px of its middle
# sample (rasterisation asymmetry) and every even factor-2 kernel is within
# 0.0024 px of a full half pixel.  Snapping the small offsets to 0 keeps an odd
# kernel's response bit-identical to ``centre="pixel"``.
CENTRE_OFFSET_TOL = 0.05

# Denominator floor for :func:`safe_divide`.  Small enough to be irrelevant
# for real spectra, large enough to keep ``|a / b| <= |a| / eps`` finite.
# It is also the floor :func:`inverse_filter` uses by default, i.e. the
# textbook inverse filter: the guard against dividing by exactly zero, and
# nothing more.  A noisy input needs a much larger floor (section 12.3 of
# ``docs/restoration_foundation.md``), which is the caller's choice, not a
# silent default.
DEFAULT_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _as_numeric_array(values, name: str) -> np.ndarray:
    """Convert to ``float64``/``complex128`` without truncating a complex input.

    A spectrum is complex, so these helpers must not route through a real
    dtype: ``np.asarray(complex_array, dtype=np.float64)`` only warns
    (``ComplexWarning``) and drops the imaginary part, which silently turns a
    deconvolution into a different -- and wrong -- operator.
    """
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:       # ragged nesting etc.
        raise ValueError(
            f"{name} must be a real or complex numeric array: {exc}"
        ) from exc

    target = np.complex128 if np.iscomplexobj(array) else np.float64
    try:
        return array.astype(target)
    except (TypeError, ValueError) as exc:       # object, string input
        raise ValueError(
            f"{name} must be a real or complex numeric array: {exc}"
        ) from exc


def _as_float_array(values, name: str) -> np.ndarray:
    """Real-only conversion: images, kernels and restored arrays stay real."""
    array = _as_numeric_array(values, name)
    if np.iscomplexobj(array):
        raise ValueError(
            f"{name} must be real-valued, got complex data; use "
            "safe_divide/near_zero_mask for spectra"
        )
    return array


def _count_non_finite(array: np.ndarray) -> int:
    """Number of NaN/inf entries, used to make the error messages specific."""
    return int(np.count_nonzero(~np.isfinite(array)))


def _require_finite(array: np.ndarray, name: str, hint: str = "") -> None:
    """Raise on a NaN/inf instead of letting a broken number flow downstream."""
    non_finite = _count_non_finite(array)
    if non_finite:
        raise ValueError(
            f"{name} has {non_finite} non-finite value(s) (NaN or inf){hint}"
        )


def validate_image(image, name: str = "image") -> np.ndarray:
    """Check the image convention and return a ``float32`` copy.

    An image is a 2-D grayscale array of finite values in ``[0, 1]``
    (0 = black).  Everything downstream assumes that, so the check happens
    once, here, rather than being re-tested inside every algorithm.

    Parameters
    ----------
    image:
        Array-like of shape ``(rows, cols)``.  Integer, ``float32`` and
        ``float64`` input are all accepted.
    name:
        Label used in the error messages, so a caller can say *which* image of
        a comparison failed.

    Returns
    -------
    np.ndarray
        Freshly allocated ``float32`` array, so in-place arithmetic later
        cannot corrupt the caller's data.

    Raises
    ------
    ValueError
        If the array is not 2-D, is empty, holds a NaN/inf, or leaves
        ``[0, 1]`` by more than :data:`IMAGE_RANGE_TOL`.
    """
    array = _as_float_array(image, name)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D grayscale array, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(
            f"{name} is empty (shape {array.shape}); need at least 1x1"
        )

    non_finite = _count_non_finite(array)
    if non_finite:
        raise ValueError(
            f"{name} has {non_finite} non-finite value(s) (NaN or inf)"
        )

    low, high = float(array.min()), float(array.max())
    if low < -IMAGE_RANGE_TOL or high > 1.0 + IMAGE_RANGE_TOL:
        raise ValueError(
            f"{name} must lie in [0, 1] (grayscale float convention), got "
            f"range [{low:.6g}, {high:.6g}]; clip with clip_unit_interval "
            "immediately before validating"
        )

    return np.ascontiguousarray(array, dtype=np.float32)


def validate_psf(psf, name: str = "psf", tol: float = PSF_SUM_TOL,
                 negative_tol: float = PSF_NEGATIVE_TOL,
                 require_odd: bool = True) -> np.ndarray:
    """Check the PSF convention and return a ``float32`` copy.

    A PSF is a 2-D, centred, essentially non-negative kernel summing to 1.
    The kernel is **checked, never renormalised**: a stored ``_psf`` from
    ``degraded.npz`` must be used exactly as it was written, so silently
    rescaling it here would break the "use the stored kernel as-is" contract.
    A kernel that misses the tolerance is an error for the caller to fix.

    Two relaxations exist because the stored dataset needs them (see the module
    docstring): weights down to ``-negative_tol`` are accepted as FFT
    round-off, and ``require_odd=False`` accepts an even side, for which the
    centre is taken at index ``size // 2``.

    Parameters
    ----------
    psf:
        Array-like kernel, normally of shape ``(2L+1, 2L+1)``.
    name:
        Label used in the error messages.
    tol:
        Absolute tolerance on ``sum(psf) == 1``.
    negative_tol:
        How far below 0 a single weight may sit before the kernel is rejected.
    require_odd:
        Require an odd side on both axes, so the kernel has an exact centre
        pixel.  ``True`` for kernels built locally; the factor-2 stored kernels
        need ``False``.

    Returns
    -------
    np.ndarray
        Freshly allocated ``float32`` copy of the kernel.

    Raises
    ------
    ValueError
        If the kernel is not 2-D, is empty, has an even side while
        ``require_odd`` is set, contains a NaN/inf or a weight below
        ``-negative_tol``, or does not sum to 1 within ``tol``.
    """
    array = _as_float_array(psf, name)

    if array.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D kernel, got shape {array.shape}"
        )
    if array.size == 0:
        raise ValueError(
            f"{name} is empty (shape {array.shape}); need at least 1x1"
        )
    if require_odd and (array.shape[0] % 2 == 0 or array.shape[1] % 2 == 0):
        raise ValueError(
            f"{name} must be odd-sized on both axes so it has an exact centre "
            f"pixel and convolution does not shift the image, got shape "
            f"{array.shape}; pass require_odd=False for a stored factor-2 kernel"
        )

    non_finite = _count_non_finite(array)
    if non_finite:
        raise ValueError(
            f"{name} has {non_finite} non-finite value(s) (NaN or inf)"
        )

    negative = int(np.count_nonzero(array < -negative_tol))
    if negative:
        raise ValueError(
            f"{name} has {negative} negative weight(s) below "
            f"{-negative_tol:g}; a PSF is a non-negative energy distribution"
        )

    total = float(array.sum(dtype=np.float64))
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{name} must sum to 1 (energy preserving), got {total:.9g} "
            f"(tolerance {tol:g}); do not rescale stored PSFs, fix the kernel"
        )

    return np.ascontiguousarray(array, dtype=np.float32)


def validate_image_shape(shape, name: str = "image_shape") -> tuple[int, int]:
    """Return ``shape`` as a ``(rows, cols)`` pair of positive integers.

    Used by :func:`psf_frequency_response` and :func:`edge_taper_window`,
    which need a target size but no image content.

    Raises
    ------
    ValueError
        If ``shape`` is not a pair, or holds a non-integral or ``< 1`` entry.
    """
    try:
        rows, cols = shape
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a (rows, cols) pair, got {shape!r}"
        ) from exc

    dims = []
    for value, label in ((rows, "rows"), (cols, "cols")):
        try:
            as_int = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name}.{label} must be a whole number, got {value!r}"
            ) from exc
        if as_int != value:
            raise ValueError(
                f"{name}.{label} must be a whole number, got {value!r}"
            )
        if as_int < 1:
            raise ValueError(f"{name}.{label} must be >= 1, got {as_int}")
        dims.append(as_int)

    return dims[0], dims[1]


# --------------------------------------------------------------------------- #
# PSF -> frequency response
# --------------------------------------------------------------------------- #
def psf_frequency_response(psf, image_shape, tol: float = PSF_SUM_TOL,
                           require_odd: bool = False,
                           centre: str = DEFAULT_CENTRE_MODE) -> np.ndarray:
    """Transfer function of ``psf`` sampled on the DFT grid of ``image_shape``.

    A PSF is a *spatial* description: ``h[y, x]`` says how a point spreads over
    its neighbours.  Deconvolution happens in the frequency domain, so the
    kernel has to be turned into the filter it applies to every DFT coefficient.

    Three steps, in this order, matter:

    1. **Zero-pad** the kernel to the image size ``(rows, cols)``, in the
       top-left corner.  Circulant convolution is defined on an array of the
       image's period, so the kernel must live on that same period.
    2. **Move the centre to the origin.**  The padded kernel currently has its
       centre at ``(kh // 2, kw // 2)``; a circular shift by that amount puts
       the centre at index ``(0, 0)``.  A kernel whose mass sits at index ``d``
       shifts the image by ``+d`` (see the module note in the docs), so
       centring it at the origin is exactly what removes the artificial
       half-kernel shift that a naive pad-and-transform produces.
    3. **Transform** with ``np.fft.fft2``, the same FFT family
       ``degradation.blur_fft`` uses.

    For the padded, re-centred kernel ``h0`` the response is

    .. math:: H(u, v) = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1}
              h_0(m, n)\\, e^{-2\\pi i (u m / M + v n / N)}

    so ``H(0, 0) = sum(h) = 1`` for a normalised kernel (the DC gain, i.e. the
    mean brightness), and ``H`` is real for a symmetric kernel.  Because
    ``h0`` is centred, ``H`` carries no linear phase ramp -- only the phase
    pattern of the blur itself.

    An **even-sided** kernel has no exact centre pixel, and
    ``effective_psf`` centres its mass at ``(K - 1) / 2``, i.e. between two
    samples.  Rolling by ``K // 2`` therefore leaves the mass ``0.5 px`` off
    the origin, which is a real spatial displacement: re-blurring the reference
    with such a response lands 0.490-0.503 px up/left of the stored image
    (measured over the grid's factor-2 cases).  ``centre="auto"`` removes it
    with the unit-modulus ramp

    .. math:: R(u, v) = e^{\\,2\\pi i (\\delta_y u / M + \\delta_x v / N)},
              \\qquad \\delta = \\text{centre of mass} - \\lfloor K / 2 \\rfloor

    which shifts the response's phase and leaves ``|H|`` untouched.  An odd
    kernel is pixel-centred already -- every ``psf_builder`` kernel sits within
    0.005 px of its middle sample, below :data:`CENTRE_OFFSET_TOL` -- so its
    response is bit-identical under both modes; the even stored kernels are a
    full half pixel off, which is the case ``"auto"`` fixes.  Nothing here
    modifies or regenerates the stored kernel.

    Parameters
    ----------
    psf:
        The blur kernel.  Validated with :func:`validate_psf`.
    image_shape:
        ``(rows, cols)`` of the image the response will be applied to.  The
        kernel must fit inside it.
    tol:
        Absolute tolerance on the PSF normalisation, forwarded to
        :func:`validate_psf`.
    require_odd:
        Enforce the odd-sized kernel convention.  Defaults to ``False`` so a
        stored ``<key>_psf`` is accepted as-is; pass ``True`` when working with
        kernels built by ``psf_builder`` and you want the exact-centre
        guarantee.
    centre:
        ``"auto"`` (default) puts the kernel's centre of mass on the FFT
        origin; ``"pixel"`` puts index ``size // 2`` there.  See
        :func:`psf_centre_offset` for the offset ``"auto"`` compensates.

    Returns
    -------
    np.ndarray
        ``complex128`` array of shape ``image_shape`` -- the response at the
        image's own resolution, ready for
        ``real(ifft2(fft2(img) * response))``.

    Raises
    ------
    ValueError
        If the shape is malformed, the kernel fails :func:`validate_psf`, the
        kernel is larger than the image, or ``centre`` is not one of
        :data:`CENTRE_MODES`.
    """
    if centre not in CENTRE_MODES:
        raise ValueError(f"centre must be one of {CENTRE_MODES}, got {centre!r}")

    rows, cols = validate_image_shape(image_shape)
    kernel = validate_psf(psf, tol=tol, require_odd=require_odd)

    kernel_rows, kernel_cols = kernel.shape
    if kernel_rows > rows or kernel_cols > cols:
        raise ValueError(
            f"psf {kernel.shape} does not fit in a {rows}x{cols} image; "
            "restoration runs at the image's own resolution, so the kernel "
            "must be no larger than the image"
        )

    # 1. Zero-pad to the image period, kernel in the top-left corner.
    padded = np.zeros((rows, cols), dtype=np.float64)
    padded[:kernel_rows, :kernel_cols] = kernel

    # 2. Circular shift: centre pixel -> index (0, 0), the FFT origin.
    centred = np.roll(padded, (-(kernel_rows // 2), -(kernel_cols // 2)),
                      axis=(0, 1))

    # 3. Sampled transfer function.
    response = np.fft.fft2(centred)

    # 4. Optional sub-pixel re-centring of the kernel's mass (see docstring).
    if centre == "auto":
        offset_rows, offset_cols = _centre_offset(kernel)
        if offset_rows or offset_cols:
            response = response * _phase_ramp((rows, cols),
                                              offset_rows, offset_cols)
    return response


def _centre_offset(kernel: np.ndarray) -> tuple[float, float]:
    """Centre of mass of ``kernel`` minus the index ``size // 2`` it is put at."""
    rows, cols = kernel.shape
    total = float(kernel.sum())
    row_axis = np.arange(rows, dtype=np.float64)
    col_axis = np.arange(cols, dtype=np.float64)
    row_centre = float((kernel.sum(axis=1) * row_axis).sum() / total)
    col_centre = float((kernel.sum(axis=0) * col_axis).sum() / total)

    offset_rows = row_centre - rows // 2
    offset_cols = col_centre - cols // 2
    # Below the tolerance the gap is rounding error, not a shift: snapping it
    # to 0 keeps an odd kernel bit-identical to ``centre="pixel"``.
    if abs(offset_rows) < CENTRE_OFFSET_TOL:
        offset_rows = 0.0
    if abs(offset_cols) < CENTRE_OFFSET_TOL:
        offset_cols = 0.0
    return offset_rows, offset_cols


def _phase_ramp(shape: tuple[int, int], offset_rows: float,
                offset_cols: float) -> np.ndarray:
    """Unit-modulus linear phase that moves content by ``-offset``."""
    rows, cols = shape
    freq_rows = np.fft.fftfreq(rows)[:, None]
    freq_cols = np.fft.fftfreq(cols)[None, :]
    return np.exp(2j * np.pi * (freq_rows * offset_rows
                                + freq_cols * offset_cols))


def psf_centre_offset(psf, tol: float = PSF_SUM_TOL) -> tuple[float, float]:
    """Sub-pixel offset of a kernel's centre of mass from ``size // 2``.

    Returns ``(offset_rows, offset_cols)`` in pixels.  It is ``(0.0, 0.0)``
    for every kernel whose mass is centred on the middle sample -- all
    ``psf_builder`` kernels, and every odd-sided ``effective_psf`` output --
    and ``(-0.5, -0.5)`` for the even-sided factor-2 kernels, whose mass sits
    between two samples.  This is the quantity ``centre="auto"`` compensates,
    exposed so step 4.2 and step 8 can report the alignment they ran with.

    Raises
    ------
    ValueError
        If the kernel fails :func:`validate_psf`.
    """
    return _centre_offset(validate_psf(psf, tol=tol, require_odd=False))


# --------------------------------------------------------------------------- #
# Edge tapering
# --------------------------------------------------------------------------- #
def _taper_profile(length: int, width: int,
                   profile: str = DEFAULT_TAPER_PROFILE) -> np.ndarray:
    """1-D window profile for one axis, 0 at the border and 1 in the interior.

    With ``d(i) = min(i, length - 1 - i)`` the distance of sample ``i`` from
    the nearer edge, the available profiles are

    .. math::

        w_{\\text{tukey}}(i) &= \\tfrac{1}{2}\\left(1 - \\cos
            \\frac{\\pi\\,\\min(d, W)}{W}\\right)
        \\\\
        w_{\\text{linear}}(i) &= \\min(d / W, 1)
        \\\\
        w_{\\text{hann}}(i) &= \\tfrac{1}{2}\\left(1 - \\cos
            \\frac{2\\pi i}{n - 1}\\right)
        \\\\
        w_{\\text{none}}(i) &= 1

    ``tukey`` is 0 on the outermost sample, reaches exactly 1 at ``d == W`` with
    zero slope (so there is no visible seam at the end of the ramp) and stays 1
    beyond it.  ``linear`` has the same plateau but a slope jump of ``1 / W`` at
    its end.  ``hann`` is the classical full-axis cosine window: it has *no*
    plateau and attenuates the whole image (an odd axis still reaches exactly 1
    at its centre sample, an even one only approaches it).  ``none`` is the
    identity -- the control arm of the comparison, not a taper.

    ``length < 2`` cannot hold a ramp on both sides, so a single-sample axis is
    returned as ``[1.0]`` and left untouched.
    """
    if profile == "none":
        return np.ones(length, dtype=np.float64)
    if length < 2:
        return np.ones(length, dtype=np.float64)

    axis = np.arange(length, dtype=np.float64)
    distance = np.minimum(axis, axis[::-1])

    if profile == "tukey":
        return 0.5 * (1.0 - np.cos(np.pi * np.minimum(distance, width) / width))
    if profile == "linear":
        return np.clip(distance / width, 0.0, 1.0)
    if profile == "hann":
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * axis / (length - 1)))
    raise ValueError(f"profile must be one of {TAPER_PROFILES}, got {profile!r}")


def edge_taper_window(shape, width: int | None = None,
                      fraction: float = DEFAULT_TAPER_FRACTION,
                      profile: str = DEFAULT_TAPER_PROFILE) -> np.ndarray:
    """Separable window that fades an image to 0 at its border.

    The 2-D window is the outer product of two 1-D profiles,

    .. math:: W(y, x) = w_{\\text{rows}}(y)\\; w_{\\text{cols}}(x)

    so with the default ``profile="tukey"`` it is exactly 1 over a rectangular
    interior plateau and falls smoothly to 0 on the outer ``width`` pixels of
    each side.  Being separable and unit inside the plateau, it leaves the
    central part of the plate bit-for-bit unchanged, which is where the
    registration lives.

    Parameters
    ----------
    shape:
        ``(rows, cols)`` of the image the window will multiply.
    width:
        Taper width in pixels, applied on **every** side.  Measured in pixels
        rather than as a fraction of each axis on purpose: a plate is strongly
        anisotropic ``(96, 336)``, so a per-axis fraction would fade 34 px of
        336 horizontally and 9 px of 96 vertically.  Defaults to
        ``round(fraction * min(rows, cols))``, clamped to
        ``[1, min(rows, cols) // 2]`` so a plateau always survives.  Not used
        by ``profile="hann"`` or ``profile="none"``.
    fraction:
        Fraction of the shorter side used when ``width`` is ``None``.  Must be
        in ``(0, 0.5]``.  Not used by ``profile="hann"`` or
        ``profile="none"``.
    profile:
        One of :data:`TAPER_PROFILES`:

        ``"tukey"`` (default)
            raised-cosine ramp with a unity plateau; smooth (zero slope) where
            the ramp meets the plateau, which is why it is the default.
        ``"linear"``
            triangular ramp with the same plateau but a ``1 / W`` slope jump at
            its end; kept for the comparison in the docs.
        ``"hann"``
            classical full-axis cosine window, **no plateau**: measured on a
            ``(96, 336)`` plate it removes 75 % of the mean intensity and
            changes the centre by up to 0.75, so it is unusable for plates and
            exists only as the reference case.
        ``"none"``
            the identity: no windowing at all.  The control arm -- a plate in
            ``plates.npz`` already carries a uniform 8 px margin, so on this
            dataset the circular model is exact to round-off and *any* taper
            can only add error (see the measurements in the docs).  Supplied
            so step 4.2 can run that ablation without a special case.

    Returns
    -------
    np.ndarray
        ``float32`` window of the requested shape, values in ``[0, 1]``, equal
        to 1 on the interior plateau (``tukey`` and ``linear`` only, and
        everywhere for ``none``).

    Raises
    ------
    ValueError
        If the shape is malformed, ``profile`` is unknown, ``fraction`` is
        outside ``(0, 0.5]``, ``width`` is not a whole number ``>= 1``, an
        explicit ``width`` is wider than ``min(rows, cols) // 2`` (it would
        erase the plateau), or ``width``/``fraction`` is passed together with
        ``profile="hann"`` / ``profile="none"``.
    """
    rows, cols = validate_image_shape(shape)
    if profile not in TAPER_PROFILES:
        raise ValueError(f"profile must be one of {TAPER_PROFILES}, got {profile!r}")
    limit = max(min(rows, cols) // 2, 1)

    if profile in ("hann", "none"):
        # Neither profile has a ramp -- 'hann' spans the whole axis and 'none'
        # is the identity -- so width/fraction would be silently ignored.
        # Refuse instead of ignoring them.
        if width is not None:
            raise ValueError(
                f"the {profile!r} profile has no ramp, so width is not used; "
                "use profile='tukey' or profile='linear' for a plateau"
            )
        if abs(float(fraction) - DEFAULT_TAPER_FRACTION) > 1e-12:
            raise ValueError(
                f"the {profile!r} profile has no ramp, so fraction is not "
                "used; use profile='tukey' for a taper with a plateau"
            )
        width = limit
    elif width is None:
        fraction = float(fraction)
        if not 0.0 < fraction <= 0.5:
            raise ValueError(
                f"fraction must be in (0, 0.5], got {fraction!r}"
            )
        width = int(round(fraction * min(rows, cols)))
        width = max(1, min(width, limit))
    else:
        try:
            as_int = int(width)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"width must be a whole number of pixels, got {width!r}"
            ) from exc
        if as_int != width:
            raise ValueError(
                f"width must be a whole number of pixels, got {width!r}"
            )
        if as_int < 1:
            raise ValueError(f"width must be >= 1 px, got {as_int}")
        if as_int > limit:
            raise ValueError(
                f"width {as_int} px leaves no plateau in a {rows}x{cols} image "
                f"(at most {limit} px is usable)"
            )
        width = as_int

    window = np.outer(_taper_profile(rows, width, profile),
                      _taper_profile(cols, width, profile))
    return window.astype(np.float32)


def taper_edges(image, width: int | None = None,
                fraction: float = DEFAULT_TAPER_FRACTION,
                profile: str = DEFAULT_TAPER_PROFILE) -> np.ndarray:
    """Fade an image to 0 at its border, ready for FFT deconvolution.

    A plain ``ifft2(fft2(img) / fft2(psf))`` treats the image as one period of a
    periodic signal: content leaving the right edge re-enters on the left.  A
    bright border next to a dark outside makes that wrap an artificial step,
    and dividing by a near-zero ``H`` turns it into ringing that spreads
    inwards and usually outlives the useful restoration.  The taper removes the
    step by making the border values equal to 0 on both sides of the wrap, so
    the periodic extension is continuous.

    That argument is a *worst case*, not a description of ``plates.npz``: every
    plate there carries a uniform 8 px grey margin, so opposite edges are
    identical and the measured wrap step is 0.0000 -- the window is insurance
    for kernels wider than the margin, and only noise is left to suppress on
    the ordinary cases.  See ``docs/restoration_foundation.md`` section 5.4 for
    the measurement, and ``profile="none"`` for the untapered control.

    It is a *pre-processing* step: :func:`taper_edges` runs before the
    restoration, not after it.

    Parameters
    ----------
    image:
        Grayscale image, validated with :func:`validate_image`.
    width, fraction, profile:
        Forwarded to :func:`edge_taper_window`.

    Returns
    -------
    np.ndarray
        ``float32`` image of the same shape, clipped to ``[0, 1]``.  It is
        exactly equal to the input on the interior plateau and attenuated on
        the tapered border.

    Notes
    -----
    Limitations, all of them real:

    * The tapered border pixels carry almost no signal, so a restored border is
      meaningless.  Score PSNR/SSIM on the interior (the pipeline already
      ignores the 8 px grey margin) and never quote a full-frame metric.
    * Tapering changes the global mean of the image, so it must be applied to
      ``g`` only -- not to the PSF, and not to the reference ``f`` that a metric
      compares against.
    * The window multiplies the image, which is *not* what the forward model
      ``g = h * f + n`` does, so the tapered image is no longer a sample of that
      model at the border.  This is the price of removing the wrap; the interior
      is untouched and the interior is what is scored.
    * A wide taper on a short axis eats into the plate: at ``(46, 168)``
      (factor 2) a 10 % taper is only 5 px per side, while at factor 1 on
      ``(96, 336)`` it is 10 px.  Keep ``width`` well below the glyph height
      (~32 px synthetic, 16 px at factor 2).
    * The default ramp is **not** free: measured on ``plates.npz`` it covers the
      outer 9-10 px, which includes the 8 px grey margin and the two dark
      frame rows of the drawing area (``W`` down to 0.82 there), while every
      glyph stroke stays at ``W == 1``.  On this dataset the untapered arm is
      the *more* accurate one -- the margin already makes the wrap continuous,
      so the ramp only adds its own model error -- and step 4.2 is the place
      that decides per algorithm.  That is why ``profile="none"`` exists and
      ``"tukey"`` remains the default only as the safe choice for inputs
      without a margin.  Numbers: ``docs/restoration_foundation.md``,
      section 5.4.
    """
    checked = validate_image(image)
    window = edge_taper_window(checked.shape, width=width, fraction=fraction,
                               profile=profile)
    tapered = checked.astype(np.float64) * window
    return clip_unit_interval(tapered, name="tapered image")


# --------------------------------------------------------------------------- #
# Shared numerical helpers
# --------------------------------------------------------------------------- #
def _floor_value(eps) -> float:
    """Validate a denominator floor and return it as a plain ``float``.

    One implementation, used by :func:`near_zero_mask` and
    :func:`safe_divide`, so the two can never disagree about what a legal
    floor is -- and so a floor given as ``np.float32`` is compared and
    multiplied as the same number in both places.
    """
    if isinstance(eps, (str, bytes)):
        raise ValueError(f"eps must be a number, got {eps!r}")
    try:
        value = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"eps must be a number, got {eps!r}") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"eps must be > 0 and finite, got {eps!r}")
    return value


def near_zero_mask(denominator, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Boolean mask of the entries :func:`safe_divide` had to clamp.

    Dividing by ``|b| <= eps`` is the one place where a numerical problem is
    masked rather than raised, so the mask is exposed: a caller can count
    ``int(near_zero_mask(H).sum())`` and report how many DFT coefficients were
    degenerate instead of pretending the division was well posed.

    Works on real *and* complex arrays.  A coefficient is degenerate when its
    **modulus** is at or below ``eps``, because ``a / b`` diverges like
    ``1 / |b|`` no matter what phase ``b`` has -- testing the real part, or
    letting NumPy cast the array to ``float64`` first, would both mis-classify
    coefficients (and the cast would also discard the imaginary part).

    Raises
    ------
    ValueError
        If ``eps`` is not a finite number ``> 0`` or the input holds a
        NaN/inf: this helper bounds a division, it does not repair input.

    Notes
    -----
    The floor is validated by :func:`_floor_value`, so ``eps=inf`` is refused
    rather than accepted: an infinite floor would turn every coefficient into
    ``inf * phase`` and, where a phase component is exactly 0, into a NaN --
    a diverged answer that looks like a numerical bug in the caller.
    """
    array = _as_numeric_array(denominator, "denominator")
    floor = _floor_value(eps)
    _require_finite(array, "denominator")
    return np.abs(array) <= floor


def safe_divide(numerator, denominator, eps: float = DEFAULT_EPS) -> np.ndarray:
    """Elementwise ``numerator / denominator`` with the denominator floored.

    Every restoration filter divides by something derived from ``H``, and
    ``H`` has genuine near-zeros (a defocus disc has deep nulls on rings, a
    motion kernel has notches along its zero-stripes).  A literal division
    there produces ``inf``/``NaN`` and the whole image becomes useless.  This
    helper replaces the denominator by

    .. math:: \\tilde{b} = \\frac{\\varepsilon\\, b}{|b|}
              \\quad\\text{for } |b| \\le \\varepsilon

    which is a vector of length ``eps`` pointing in the same direction as
    ``b``.  For a real ``b`` that is the familiar
    ``sign(b) * max(|b|, eps)``; for a complex ``b`` it keeps the phase, which
    matters because a restoration filter is ``G / H``: a floor that dropped the
    phase would apply a different rotation and displace the restored content.
    The quotient is bounded by ``|a| / eps`` either way.

    Parameters
    ----------
    numerator, denominator:
        Broadcastable real or complex arrays.  Harmless for a real numerator
        over a complex denominator (NumPy promotes the result).
    eps:
        Floor applied to ``|denominator|``.  Must be ``> 0``.

    Returns
    -------
    np.ndarray
        Quotient: ``float64`` for real input, ``complex128`` as soon as either
        operand is complex.  The imaginary part of a complex input is never
        discarded.  Clamp how many entries were affected with
        :func:`near_zero_mask`.

    Raises
    ------
    ValueError
        If ``eps`` is not positive or either operand holds a NaN/inf: this
        helper repairs *small* denominators, it does not hide a broken input.
    """
    numerator_array = _as_numeric_array(numerator, "numerator")
    denominator_array = _as_numeric_array(denominator, "denominator")

    hint = "; check the spectrum before dividing"
    _require_finite(numerator_array, "numerator", hint)
    _require_finite(denominator_array, "denominator", hint)

    floor = _floor_value(eps)
    mask = near_zero_mask(denominator_array, eps=floor)
    magnitude = np.abs(denominator_array)
    # b / |b| is the unit vector along b; a zero coefficient has no phase, so
    # the floor falls back to +eps (real) / eps + 0j.
    phase = np.divide(denominator_array, magnitude,
                      out=np.ones_like(denominator_array), where=magnitude > 0)
    floored = np.where(mask, floor * phase, denominator_array)
    return numerator_array / floored


def clip_unit_interval(values, name: str = "values",
                       check_finite: bool = True) -> np.ndarray:
    """Clip to ``[0, 1]`` and cast to ``float32`` -- the one legal place to clip.

    Inverse and Richardson--Lucy restoration overshoot: they can produce
    values below 0 (undershoot near an edge) and above 1 (overshoot on a bright
    stroke).  The image convention is ``[0, 1]``, so the overshoot has to be
    removed exactly once, before any metric or binarizer sees the image --
    ``binarize`` clips internally, and a negative undershoot would otherwise be
    mapped to 0 = ink and fatten the glyphs.

    Clipping is *not* a way to hide a diverged filter.  With
    ``check_finite=True`` (the default) a NaN/inf anywhere raises instead of
    being smeared into the output by ``np.clip``.

    Parameters
    ----------
    values:
        Array-like of any shape.
    name:
        Label used in the error message.
    check_finite:
        Raise on NaN/inf.  Set to ``False`` only for intermediate arrays whose
        finiteness is already guaranteed.

    Returns
    -------
    np.ndarray
        ``float32`` array of the input's shape with every value in ``[0, 1]``.

    Raises
    ------
    ValueError
        If ``check_finite`` is set and the input holds a NaN/inf.
    """
    array = _as_float_array(values, name)
    if check_finite:
        non_finite = _count_non_finite(array)
        if non_finite:
            raise ValueError(
                f"{name} has {non_finite} non-finite value(s) (NaN or inf); "
                "the filter diverged, clip_unit_interval will not mask it"
            )
    return np.clip(array, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Inverse filtering (step 4.2)
# --------------------------------------------------------------------------- #
def inverse_filter(image, psf, eps: float = DEFAULT_EPS,
                   width: int | None = None,
                   fraction: float = DEFAULT_TAPER_FRACTION,
                   profile: str = DEFAULT_INVERSE_TAPER,
                   centre: str = DEFAULT_CENTRE_MODE,
                   require_odd: bool = False,
                   tol: float = PSF_SUM_TOL) -> np.ndarray:
    """Recover a plate from its blur by dividing the spectrum by ``H``.

    The forward model this inverts is ``g = h * f + n`` at the image's own
    resolution (a decimated image is restored at *its* resolution, with the
    ``_psf`` stored beside it -- see the module note).  The convolution
    theorem turns ``*`` into a product,

    .. math:: G(u,v) = H(u,v)\\,F(u,v) + N(u,v), \\qquad
              G = \\mathcal{F}\\{g\\},\\; H = \\mathcal{F}\\{h\\},

    so the unknown ``F`` is recovered by one division per DFT coefficient,

    .. math:: \\hat{F}(u,v) = \\frac{G(u,v)}{H(u,v)}, \\qquad
              \\hat{f} = \\mathcal{F}^{-1}\\{\\hat{F}\\},

    followed by an inverse transform.  That is the whole algorithm.  What
    makes it worth a page of documentation is that the division is ill posed:
    where ``|H|`` is small the same quotient multiplies ``N`` far more than it
    multiplies ``F``, so the noise -- not the picture -- determines the answer.

    Exact recipe, in this order (this *is* the preprocessing, nothing is
    implicit):

    1. the image is validated as grayscale ``float32`` in ``[0, 1]``;
    2. ``H`` is built by :func:`psf_frequency_response` on the image's own
       grid, with ``centre="auto"`` the kernel's centre of mass on the FFT
       origin (no stored kernel is resized, renormalised or regenerated);
    3. the border is faded with :func:`taper_edges` -- or left alone for
       ``profile="none"``, which is the identity;
    4. ``G = fft2(windowed_image)``;
    5. ``\\hat{F} = G / H`` with the denominator floored by ``eps``
       (:func:`safe_divide`, which keeps the complex values and the phase);
    6. ``real(ifft2(\\hat{F}))`` -- the true image is real, and for a real
       image and a real kernel the quotient is Hermitian to round-off, so the
       imaginary part that is dropped here is round-off, not signal;
    7. ``clip_unit_interval``: values back into ``[0, 1]`` and ``float32``.

    That last step is not cosmetic.  A quotient at a floored coefficient can
    exceed 1 by orders of magnitude; without clipping, ``binarize`` (step 5)
    would see an image whose nominal range is meaningless, and a negative
    undershoot near an edge would be read as ink and fatten the glyphs.

    Parameters
    ----------
    image:
        The degraded plate, ``(rows, cols)`` in ``[0, 1]``.  Validated by
        :func:`validate_image`; a copy is taken, so the caller's array is not
        modified.
    psf:
        The blur kernel that produced ``image``, consumed **as-is** (the
        stored ``<key>_psf``).  Validated by :func:`validate_psf`; an even
        side is accepted because half of the factor-2 stored kernels have one,
        and ``centre="auto"`` handles the half-pixel convention that comes
        with it.
    eps:
        Denominator floor: the smallest ``|H|`` the division is allowed to
        use.  For ``|H| <= eps`` the denominator is replaced by
        ``eps * H / |H|``, i.e. the same direction with a bounded magnitude, so
        the quotient at that coefficient is bounded by ``|G| / eps`` instead
        of diverging.  The default :data:`DEFAULT_EPS` (``1e-12``) makes this
        the textbook inverse filter -- a guard against dividing by an exact
        zero, nothing more.  **Raising it is the only knob this filter has**:
        with ``eps`` at 1e-2 the gain at a notch is capped at 100, the noise
        that rides on the notch is capped with it, and the price is that the
        detail the notch destroyed is capped too.  There is no single correct
        value: it trades resolution of the nulled frequencies against noise
        amplification, which is exactly the quantity a Wiener filter sets from
        the measured BSNR instead of guessing.  Measured trade-off in
        ``docs/restoration_foundation.md`` section 12.3.
    width, fraction, profile:
        Forwarded to :func:`taper_edges`, so the arms of the edge-taper
        comparison are selectable per call.  ``profile="none"`` runs the
        filter on the raw image and is the default
        (:data:`DEFAULT_INVERSE_TAPER`) *because it was measured*: on this
        dataset the 8 px margin already makes the circular model exact, so the
        ramp can only add model error where ``|H|`` is small -- and the
        inverse filter amplifies precisely that.  Measured on a rendered
        plate, clean input, 7x7 motion kernel: 3.3e-5 RMS untapered against
        0.70 tapered.  ``profile="tukey"`` remains one keyword away, and it is
        the right choice for an input without a margin; the global
        :data:`DEFAULT_TAPER_PROFILE` was not changed by this phase.
    centre:
        ``"auto"`` (default) or ``"pixel"``, forwarded to
        :func:`psf_frequency_response`.  See :func:`psf_centre_offset` for the
        offset ``"auto"`` compensates.
    require_odd:
        Forwarded to :func:`validate_psf` for the kernel.  ``False`` (default)
        accepts the even-sided stored kernels; ``True`` enforces the
        ``psf_builder`` contract and is what a hand-built kernel should use.
    tol:
        Absolute tolerance on ``sum(psf) == 1``.

    Returns
    -------
    np.ndarray
        ``float32`` estimate of the sharp plate, same shape as ``image``,
        every value finite and inside ``[0, 1]``.  It is the estimate of the
        *windowed* plate: with a taper the border carries the window, so only
        the interior plateau is comparable with the reference.

    Raises
    ------
    ValueError
        If the image breaks the grayscale convention, the kernel fails
        :func:`validate_psf` or is larger than the image, ``eps`` is not a
        finite number ``> 0``, ``centre`` or ``profile`` is unknown, or the
        taper parameters are inconsistent with the profile.  A diverged
        quotient is not masked: :func:`safe_divide` raises on a non-finite
        input and :func:`clip_unit_interval` raises on a non-finite output, so
        a NaN/inf can never reach a caller disguised as a plausible image.

    Notes
    -----
    Limitations, all of them inherent to the filter and none hidden by the
    interface:

    * ``H`` has genuine zeros and near-zeros (measured ``|H|`` minimum
      ``2.1e-6`` for a 7x7 motion kernel on a plate), so the filter is only
      meaningful at frequencies where ``|H|`` is comfortably above the noise.
      A larger ``eps`` buys a stable image by giving up the nulled detail.
    * The floor does not remove the noise amplification *around* the notches;
      it only bounds it.  At 20 dB BSNR the best a floor can do is modestly
      beat the blurred input (7x7: 8.9e-2 restored against 1.1e-1 blurred), and
      the position of the optimum moves with the noise level -- which is exactly
      why step 4.3 sets its ``K`` from the measured BSNR instead of a constant.
    * The estimate cannot be better than the model: ``H`` must belong to the
      image (the stored ``_psf`` is resolution-matched), and a decimation the
      model does not describe is not recoverable.
    * Clipping is lossy.  Saturated pixels hide how far the quotient overshot,
      so report ``eps`` alongside any restored image you quote.

    Examples
    --------
    >>> import numpy as np
    >>> from license_plate_restoration.restoration import inverse_filter
    >>> image = np.full((16, 16), 0.5, dtype=np.float32)
    >>> delta = np.array([[1.0]], dtype=np.float32)
    >>> np.allclose(inverse_filter(image, delta), image, atol=1e-6)
    True
    """
    degraded = validate_image(image, name="image")
    response = psf_frequency_response(psf, degraded.shape, tol=tol,
                                      require_odd=require_odd, centre=centre)

    windowed = taper_edges(degraded, width=width, fraction=fraction,
                           profile=profile)
    spectrum = np.fft.fft2(windowed.astype(np.float64))
    estimate = safe_divide(spectrum, response, eps=eps)
    restored = np.real(np.fft.ifft2(estimate))
    return clip_unit_interval(restored, name="restored image")


# --------------------------------------------------------------------------- #
# Manual check: python -m license_plate_restoration.restoration
# --------------------------------------------------------------------------- #
def _demo() -> None:  # pragma: no cover
    """Print the numbers the docs quote: response, centring, taper, inverse."""
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

    def centroid(image: np.ndarray) -> float:
        rows = np.arange(image.shape[0], dtype=np.float64)[:, None]
        mass = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
        return float((mass * rows).sum() / mass.sum())

    def rmse(estimate, reference, margin: int = 12) -> float:
        error = (np.asarray(estimate, np.float64)
                 - np.asarray(reference, np.float64))
        return float(np.sqrt(np.mean(error[margin:-margin,
                                           margin:-margin] ** 2)))

    plate, text = generate_plates(1, seed=7)[0]
    psf = make_motion_psf(3, 8.0)
    response = psf_frequency_response(psf, plate.shape)

    # A zero border at least as wide as the kernel keeps the circular result
    # free of wrap, so it must reproduce the reflect-padded blur_fft exactly.
    margin = psf.shape[0]
    padded = np.zeros(plate.shape, dtype=np.float64)
    padded[margin:-margin, margin:-margin] = plate[margin:-margin, margin:-margin]
    circular = np.real(np.fft.ifft2(np.fft.fft2(padded) * response))
    reference = blur_fft(padded.astype(np.float32), psf)

    tapered = taper_edges(plate)
    window = edge_taper_window(plate.shape)
    width = int(round(DEFAULT_TAPER_FRACTION * min(plate.shape)))
    plateau_identical = np.array_equal(tapered[width:-width, width:-width],
                                       plate[width:-width, width:-width])

    print(f"plate {text}  shape {plate.shape}  psf {psf.shape}")
    print(f"H(0,0) = {response[0, 0].real:.6f}  (DC gain, must be 1)")
    print(f"|H| min = {np.abs(response).min():.3e}  -> why a floor is needed")
    print(f"max |response - blur_fft| = "
          f"{np.max(np.abs(circular - reference)):.3e}")
    print(f"taper width {width} px per side; "
          f"{int(np.count_nonzero(window == 1.0))} untouched pixels")
    print(f"corner before/after {plate[0, 0]:.4f} / {tapered[0, 0]:.4f}; "
          f"plateau identical = {plateau_identical}")

    # Complex arithmetic: a real image and a real kernel give a Hermitian
    # quotient, so the imaginary part must cancel instead of being dropped.
    quotient = safe_divide(np.fft.fft2(plate.astype(np.float64)), response)
    restored = np.fft.ifft2(quotient)
    print(f"complex quotient {quotient.dtype}; "
          f"max |imag(ifft2)| = {np.abs(restored.imag).max():.3e}")
    floored = safe_divide(np.array([1.0 + 2.0j]),
                          np.array([1e-20 * np.exp(0.7j)]))
    print(f"floored coefficient phase {np.angle(floored[0]):+.6f} rad, "
          f"expected {np.angle(1.0 + 2.0j) - 0.7:+.6f}")

    # Even-sized stored kernels: displacement against the dataset's own model.
    print()
    print(f"{'kernel':>8} {'parity':>6} {'offset':>8} {'auto':>9} {'pixel':>9}")
    for length, radius in ((15, 0), (15, 3), (5, 3), (11, 1)):
        combined = combine_psfs(make_motion_psf(length, 20.0),
                                make_defocus_psf(radius))
        stored = effective_psf(combined, 2)
        clean = np.zeros((96, 336), dtype=np.float32)
        clean[24:72, 80:256] = np.random.default_rng(0).random(
            (48, 176), dtype=np.float32)
        observed = downsample(blur_fft(clean, combined), 2)
        decimated = downsample(clean, 2)
        shifts = []
        for mode in CENTRE_MODES:
            kernel_response = psf_frequency_response(stored, observed.shape,
                                                     centre=mode)
            recon = np.real(np.fft.ifft2(
                np.fft.fft2(decimated.astype(np.float64)) * kernel_response))
            shifts.append(centroid(recon) - centroid(observed))
        parity = "even" if stored.shape[0] % 2 == 0 else "odd"
        print(f"{str(stored.shape):>8} {parity:>6} "
              f"{psf_centre_offset(stored)[0]:>+8.3f} "
              f"{shifts[0]:>+9.4f} {shifts[1]:>+9.4f}")

    # Taper profiles: what the ramp costs and how smoothly it ends.
    print()
    print(f"{'window':>8} {'mean loss':>10} {'centre':>8} {'slope jump':>11}")
    for profile in TAPER_PROFILES:
        candidate = edge_taper_window(plate.shape, profile=profile)
        middle_row = np.asarray(candidate, dtype=np.float64)[plate.shape[0] // 2]
        print(f"{profile:>8} {100.0 * float(np.mean(1.0 - candidate)):>9.2f}% "
              f"{candidate[plate.shape[0] // 2, plate.shape[1] // 2]:>8.4f} "
              f"{np.abs(np.diff(middle_row, 2)).max():>11.4f}")

    # Inverse filtering: what the floor and the taper arm actually cost.
    print()
    print(f"{'kernel':>7} {'noise':>6} {'taper':>5} {'eps':>7} "
          f"{'rmse':>9} {'floored':>8}")
    for length, radius in ((3, 0), (21, 4)):
        kernel = combine_psfs(make_motion_psf(length, 8.0),
                             make_defocus_psf(radius))
        blurred = blur_fft(plate, kernel)
        floored_by_eps = {eps: int(np.count_nonzero(
            near_zero_mask(psf_frequency_response(kernel, plate.shape), eps)))
            for eps in (DEFAULT_EPS, 1e-2, 1e-1)}
        for label, degraded in (("clean", blurred),
                                ("20 dB", add_noise(blurred, 20.0, seed=3))):
            for taper, eps in (("none", DEFAULT_EPS), ("none", 1e-2),
                               ("tukey", DEFAULT_EPS), ("tukey", 1e-2),
                               ("none", 1e-1), ("tukey", 1e-1)):
                estimate = inverse_filter(degraded, kernel, eps=eps,
                                          profile=taper)
                print(f"{str(kernel.shape):>7} {label:>6} {taper:>5} "
                      f"{eps:>7.0e} {rmse(estimate, plate):>9.3e} "
                      f"{floored_by_eps[eps]:>8d}")

    # The floor is a real knob: at 20 dB BSNR it has an interior optimum.
    for length, radius in ((3, 0), (21, 4)):
        kernel = combine_psfs(make_motion_psf(length, 8.0),
                              make_defocus_psf(radius))
        noisy = add_noise(blur_fft(plate, kernel), 20.0, seed=3)
        sweep = " ".join(
            f"{eps:g}:{rmse(inverse_filter(noisy, kernel, eps=eps),
                            plate):.3e}"
            for eps in (1e-2, 1e-1, 3e-1, 1.0))
        print(f"floor sweep {str(kernel.shape):>7} at 20 dB BSNR -> {sweep}")

    # The even stored kernel: does the filter land on the reference?
    full = combine_psfs(make_motion_psf(15, 20.0), make_defocus_psf(3))
    stored = effective_psf(full, 2)
    observed = downsample(blur_fft(plate, full), 2)
    reference = downsample(plate, 2)
    print(f"factor 2: stored {stored.shape}, offset "
          f"{psf_centre_offset(stored)[0]:+.3f} px")
    for mode in CENTRE_MODES:
        estimate = inverse_filter(observed, stored, eps=1e-2, centre=mode)
        drift = centroid(estimate) - centroid(reference)
        print(f"  centre={mode:>5} centroid drift {drift:+.4f} px   "
              f"rmse {rmse(estimate, reference):.3f}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
