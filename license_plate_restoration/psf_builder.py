"""Step 2.1 -- analytic PSF construction (motion blur and defocus blur).

A PSF (point spread function) is the image of a single point light source; it
*is* the blur kernel ``h`` of the forward model ``g = D(h * f) + n``.  Both
builders return ``float32`` kernels that are odd-sized (so they have an exact
centre pixel), centred (no image shift after convolution) and normalised to sum
to 1 (no change of mean brightness).

Everything here is plain NumPy: no ``scipy.signal``, no ``cv2.getStructuring``
-style helpers, no Gaussian shortcuts.

Angle convention: degrees, counter-clockwise from the positive x-axis.  Since
image rows grow *downwards*, a CCW rotation in the plane corresponds to
``(dx, dy) = (cos t, -sin t)`` in array coordinates.
"""

from __future__ import annotations

import numpy as np

# Sub-pixel sampling step along the motion line.  0.01 gives <= 0.5 px error in
# the anti-aliased line profile, at negligible cost for kernel sizes <= 43.
_LINE_STEP = 0.01


def _normalise(kernel: np.ndarray) -> np.ndarray:
    """Scale ``kernel`` so it sums to exactly 1 (float64 maths, float32 out)."""
    total = kernel.sum(dtype=np.float64)
    if total <= 0:                              # degenerate: fall back to delta
        delta = np.zeros(kernel.shape, dtype=np.float32)
        delta[kernel.shape[0] // 2, kernel.shape[1] // 2] = 1.0
        return delta
    return (kernel / total).astype(np.float32)


def make_motion_psf(length: int, angle: float) -> np.ndarray:
    """Build an anti-aliased linear (camera-shake) motion PSF.

    The kernel models a point that travels ``2 * length`` pixels along a
    straight line at ``angle`` degrees CCW from the x-axis, so a plate edge is
    smeared over ``length`` pixels on either side of its true position.

    ``length`` is a *half*-length, which is the naming used by the project
    brief and by ``design.md``.  ``length=15`` therefore smears over 31 pixels
    ``(2*15 + 1)``, not 15; quote the smeared width, not ``length``, in the
    report.

    Algorithm: walk the line in steps of ``_LINE_STEP`` pixels and deposit each
    sub-sample's weight into the four nearest integer pixels using bilinear
    weights.  This anti-aliases the line (a single-pixel Bresenham line would
    quantise the angle to 45 degree steps and bias the blur direction).

    Parameters
    ----------
    length:
        Half-length of the smear in pixels.  ``length == 0`` returns the 1x1
        delta PSF ``[[1.0]]`` (no blur).
    angle:
        Direction of the smear in degrees, counter-clockwise from the +x axis.

    Returns
    -------
    np.ndarray
        ``float32`` kernel of shape ``(2*length+1, 2*length+1)``, odd-sized,
        centred on the middle pixel, summing to 1 within 1e-5.
    """
    length = int(length)
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    if length == 0:
        return np.array([[1.0]], dtype=np.float32)

    size = 2 * length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    centre = length                             # index of the middle pixel

    # Direction of travel in array coordinates (y grows downwards).
    theta = np.deg2rad(float(angle))
    dx, dy = np.cos(theta), -np.sin(theta)

    # Sample the whole line, endpoints included.
    samples = np.arange(-length, length + _LINE_STEP, _LINE_STEP)
    weight = _LINE_STEP / (2.0 * length)        # normalised after accumulation

    xs = centre + samples * dx
    ys = centre + samples * dy

    # Bilinear splat: split each weight over the four surrounding pixels,
    # proportional to the overlap with that sub-pixel position.
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = xs - x0
    fy = ys - y0

    for ox, oy, w in (
        (0, 0, (1.0 - fx) * (1.0 - fy)),
        (1, 0, fx * (1.0 - fy)),
        (0, 1, (1.0 - fx) * fy),
        (1, 1, fx * fy),
    ):
        px, py = x0 + ox, y0 + oy
        inside = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        np.add.at(kernel, (py[inside], px[inside]), (w * weight)[inside])

    return _normalise(kernel)


def make_defocus_psf(radius: float) -> np.ndarray:
    """Build an anti-aliased disc PSF (out-of-focus lens).

    A defocused point spreads into a uniform disc of light of the given radius.

    Algorithm: for every pixel, take its distance ``d`` from the centre.  A
    pixel fully inside the disc gets weight 1, a pixel fully outside gets 0, and
    a pixel straddling the boundary gets a linear ramp
    ``radius + 0.5 - d`` which approximates the fraction of its area covered
    (this is the anti-aliasing term that keeps the kernel smooth as ``radius``
    varies continuously).

    Parameters
    ----------
    radius:
        Disc radius in pixels.  ``radius == 0`` returns the 1x1 delta PSF.
        Non-integer radii are allowed and give a softer boundary.

    Returns
    -------
    np.ndarray
        ``float32`` kernel of shape ``(2*ceil(radius)+1, 2*ceil(radius)+1)``,
        odd-sized, centred, summing to 1 within 1e-5.
    """
    radius = float(radius)
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if radius == 0:
        return np.array([[1.0]], dtype=np.float32)

    half = int(np.ceil(radius))
    size = 2 * half + 1

    # Distance of every pixel centre from the kernel centre.
    axis = np.arange(size, dtype=np.float64) - half
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    dist = np.hypot(xx, yy)

    # Fraction of each pixel's area inside the circle (linear approximation).
    kernel = np.clip(radius + 0.5 - dist, 0.0, 1.0)

    return _normalise(kernel)


def combine_psfs(motion_psf: np.ndarray, defocus_psf: np.ndarray) -> np.ndarray:
    """Convolve two PSFs into one combined kernel, still summing to 1.

    Convolution of the two spread functions corresponds to applying both blurs
    in sequence.  The convolution is done by direct accumulation (flip and
    shifted-add) rather than through ``blur_fft``: the kernels are tiny, so the
    exact linear convolution is clearer and avoids the reflect-padding that
    ``blur_fft`` performs.

    Returns
    -------
    np.ndarray
        ``float32`` kernel of shape ``(m + d - 1, m + d - 1)``.
    """
    motion = np.asarray(motion_psf, dtype=np.float64)
    defocus = np.asarray(defocus_psf, dtype=np.float64)

    out_h = motion.shape[0] + defocus.shape[0] - 1
    out_w = motion.shape[1] + defocus.shape[1] - 1
    combined = np.zeros((out_h, out_w), dtype=np.float64)

    # Direct 2-D convolution: flip the second kernel and accumulate shifted,
    # scaled copies of it.  Two nested loops over a <= 43 px kernel are cheap
    # and keep the operation transparent.
    flipped = defocus[::-1, ::-1]
    for i in range(motion.shape[0]):
        for j in range(motion.shape[1]):
            w = motion[i, j]
            if w:
                combined[i:i + flipped.shape[0], j:j + flipped.shape[1]] += w * flipped

    return _normalise(combined)


def psf_is_valid(psf: np.ndarray, tol: float = 1e-5) -> bool:
    """Sanity check used by the tests: odd-sized, finite, centred sum of 1.

    "Centred" is checked through the first moment: for a symmetric kernel the
    centre of mass sits on the middle pixel, so an unshifted PSF must satisfy
    ``sum(i * h[i]) == centre * sum(h)`` in both axes.
    """
    psf = np.asarray(psf, dtype=np.float64)
    if psf.ndim != 2 or psf.shape[0] % 2 == 0 or psf.shape[1] % 2 == 0:
        return False
    if not np.all(np.isfinite(psf)) or np.any(psf < 0):
        return False
    if abs(psf.sum() - 1.0) > tol:
        return False

    rows = np.arange(psf.shape[0])[:, None]
    cols = np.arange(psf.shape[1])[None, :]
    centre_r = (psf * rows).sum() / psf.sum()
    centre_c = (psf * cols).sum() / psf.sum()
    return abs(centre_r - psf.shape[0] // 2) < 0.5 and abs(centre_c - psf.shape[1] // 2) < 0.5
