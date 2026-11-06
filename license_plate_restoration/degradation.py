"""Steps 2.2 and 2.3 -- the degradation stage of the forward model.

``g = D( h * f ) + n``

* ``h * f``  -> :func:`blur_fft`   (FFT convolution with reflect-padded edges)
* ``D(.)``   -> :func:`downsample` (block-mean decimation, from scratch)
* ``+ n``    -> :func:`add_noise`  (Gaussian noise at a target BSNR)
* the loop  -> :func:`run_degradation_grid` (parameter sweep + logging)

All of it is written with NumPy only: no ``scipy.signal``/``scipy.ndimage``,
no ``cv2.filter2D``/``cv2.GaussianBlur``/``cv2.resize``.

Images are grayscale ``float32`` in ``[0, 1]``.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)


# --------------------------------------------------------------------------- #
# Fourier-domain convolution
# --------------------------------------------------------------------------- #
def blur_fft(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve ``img`` with ``psf`` in the Fourier domain, without wraparound.

    Plain ``fft2`` convolution is *circular*: content leaving the right edge
    re-enters on the left.  Two steps remove that artefact:

    1. Reflect-pad the image by ``max(psf.shape) // 2`` pixels on every side, so
       the kernel never sees the opposite edge.
    2. Embed the PSF in an array of the padded size with its centre at pixel
       ``(0, 0)`` (the origin of a circular convolution), which is exactly what
       the FFT expects for a shift-invariant filter.

    After the inverse FFT the padding is cropped away again, so the output has
    the input's shape and the image is not shifted (the kernel is centred).

    Parameters
    ----------
    img:
        2-D grayscale image, ``float32`` (other dtypes are cast).
    psf:
        2-D blur kernel, typically odd-sized and summing to 1.  A
        ``[[1.0]]`` delta leaves the image unchanged.

    Returns
    -------
    np.ndarray
        ``float32`` array with the same shape as ``img``.  Values are not
        clipped, so a PSF summing to 1 keeps the intensity scale.
    """
    image = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(psf, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"img must be 2-D, got shape {image.shape}")
    if kernel.ndim != 2 or kernel.size == 0:
        raise ValueError(f"psf must be a non-empty 2-D kernel, got {kernel.shape}")

    rows, cols = image.shape
    pad = max(kernel.shape) // 2
    # np.pad(mode="reflect") cannot pad more than (dim - 1) pixels.
    pad = min(pad, min(rows, cols) - 1) if min(rows, cols) > 1 else 0

    padded_shape = (rows + 2 * pad, cols + 2 * pad)
    if kernel.shape[0] > padded_shape[0] or kernel.shape[1] > padded_shape[1]:
        raise ValueError(
            f"psf {kernel.shape} is too large for a {image.shape} image "
            f"(padded to {padded_shape})"
        )

    # 1. Reflect-pad the image.
    image_pad = np.pad(image, pad, mode="reflect") if pad > 0 else image.copy()

    # 2. Place the PSF centre at the origin (top-left corner).
    psf_pad = np.zeros(padded_shape, dtype=np.float64)
    kh, kw = kernel.shape
    psf_pad[:kh, :kw] = kernel
    psf_pad = np.roll(psf_pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))

    # 3. Multiply in the frequency domain, invert, crop back.
    spectrum = np.fft.fft2(image_pad) * np.fft.fft2(psf_pad)
    blurred = np.real(np.fft.ifft2(spectrum))
    cropped = blurred[pad:pad + rows, pad:pad + cols]

    return cropped.astype(np.float32)


# --------------------------------------------------------------------------- #
# Block-mean downsampling
# --------------------------------------------------------------------------- #
def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    """Decimate ``img`` by averaging non-overlapping ``factor x factor`` blocks.

    Block averaging (rather than picking every ``factor``-th pixel) is the
    low-pass first, decimate second behaviour that the forward model ``D(.)``
    describes, so it does not alias character strokes into moire patterns.

    Parameters
    ----------
    img:
        2-D grayscale image.
    factor:
        Integer decimation factor.  ``1`` returns an unchanged copy.

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(H // factor, W // factor)``.  Rows and
        columns that do not fill a whole block are dropped (documented
        behaviour for non-divisible sizes).

    Notes
    -----
    The averaging is a ``reshape`` + ``mean`` over the two block axes: no
    library call performs the downsampling.
    """
    image = np.asarray(img, dtype=np.float32)
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return image.copy()

    rows, cols = image.shape
    out_rows, out_cols = rows // factor, cols // factor
    if out_rows == 0 or out_cols == 0:
        raise ValueError(
            f"factor {factor} is larger than the image {image.shape}"
        )

    # Crop to a whole number of blocks, then average each (factor x factor) cell.
    cropped = image[:out_rows * factor, :out_cols * factor]
    blocks = cropped.reshape(out_rows, factor, out_cols, factor)
    return blocks.mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


# --------------------------------------------------------------------------- #
# Noise injection at a target BSNR
# --------------------------------------------------------------------------- #
def noise_sigma(img: np.ndarray, bsnr_db: float) -> float:
    """Return the noise standard deviation for a target BSNR.

    ``BSNR = 10 * log10(var(signal) / sigma^2)``  =>  ``sigma^2 = var / 10^(BSNR/10)``

    The variance is taken over the *whole* blurred image, which is the usual
    definition of BSNR for a single realisation.
    """
    variance = float(np.var(np.asarray(img, dtype=np.float64)))
    return float(np.sqrt(variance / (10.0 ** (float(bsnr_db) / 10.0))))


def add_noise(img: np.ndarray, bsnr_db: float, seed: int,
              saturation_compensation: bool = True) -> np.ndarray:
    """Add white Gaussian noise at the requested BSNR.

    Parameters
    ----------
    img:
        Blurred 2-D image, ``float32`` in ``[0, 1]``.
    bsnr_db:
        Blurred signal-to-noise ratio in dB.  Typical values: 40 (clean) down
        to 20 (very noisy).
    seed:
        Seed of the single ``numpy.random.default_rng`` used here, so the same
        ``(img, bsnr_db, seed)`` always gives the same noise.
    saturation_compensation:
        Correct the noise amplitude for the clipping at 0 and 1 (default).
        See the notes below.

    Returns
    -------
    np.ndarray
        ``float32`` array of the input's shape, clipped to ``[0, 1]`` (clipping
        happens last, as required by the forward model).

    Notes
    -----
    The starting standard deviation is the one required by the forward model::

        sigma = sqrt(var(img) / 10 ** (bsnr_db / 10))

    A plate image is about 30 % pure white, so the positive tail of the noise
    is truncated when the result is clipped to ``[0, 1]``.  That truncation
    removes noise energy: with the raw ``sigma`` the *measured* BSNR of a
    synthetic plate comes out about +1 dB above the target (worse at low BSNR),
    which would fail the \"measured BSNR within 1 dB\" requirement.

    ``saturation_compensation=True`` therefore re-scales the single seeded noise
    realisation so that the variance of the *realised* (post-clipping) noise
    ``clip(img + n) - img`` matches ``var(img) / 10**(bsnr/10)``.  The fixed-point
    loop starts from the formula above and only touches the amplitude, so the
    noise pattern and the reproducibility of a given ``seed`` are unchanged.
    Set the flag to ``False`` to get the literal formula with no correction.
    """
    image = np.asarray(img, dtype=np.float32)
    sigma = noise_sigma(image, bsnr_db)
    target_var = float(sigma) ** 2

    # One seeded standard-normal realisation; only its amplitude is adjusted.
    normal = np.random.default_rng(seed).standard_normal(image.shape).astype(np.float32)
    if target_var <= 0.0:                    # constant image -> nothing to add
        return image.copy()

    signal = image.astype(np.float64)
    amplitude = float(sigma)
    if saturation_compensation:
        for _ in range(10):
            resid = np.clip(signal + amplitude * normal, 0.0, 1.0) - signal
            realised = float(np.var(resid))
            if realised <= 0.0:
                break
            ratio = target_var / realised
            if abs(ratio - 1.0) <= 1e-3:      # within ~0.004 dB
                break
            amplitude *= float(np.sqrt(ratio))

    noisy = np.clip(signal + amplitude * normal, 0.0, 1.0)
    return noisy.astype(np.float32)


def measure_bsnr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure the achieved BSNR in dB given the clean and noisy images.

    ``10 * log10(var(clean) / var(noisy - clean))`` -- used by the tests to
    confirm that :func:`add_noise` hits its target.
    """
    clean = np.asarray(clean, dtype=np.float64)
    residual = np.asarray(noisy, dtype=np.float64) - clean
    signal_var = float(np.var(clean))
    noise_var = float(np.var(residual))
    if noise_var <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(signal_var / noise_var))


# --------------------------------------------------------------------------- #
# Resolution-matched kernels
# --------------------------------------------------------------------------- #
def effective_psf(psf: np.ndarray, factor: int) -> np.ndarray:
    """Return the blur kernel as seen by an image decimated by ``factor``.

    Block-mean decimation is itself a filter (a ``factor x factor`` box followed
    by picking every ``factor``-th sample).  Filtering with ``psf`` and then
    decimating is equivalent to filtering with ``psf * box`` and decimating, so
    the kernel that belongs to the *low-resolution* image is
    ``decimate(psf * box)``.  It is renormalised to sum to 1.

    Deconvolution of a decimated image must use this kernel, not the
    full-resolution one, otherwise the blur is modelled twice as wide as it is.

    Parameters
    ----------
    psf:
        Full-resolution kernel.
    factor:
        Decimation factor used on the image (1 = unchanged).

    Returns
    -------
    np.ndarray
        ``float32`` kernel summing to 1.  Its centre is aligned to the nearest
        low-resolution pixel (a half-pixel quantisation is possible).
    """
    psf = np.asarray(psf, dtype=np.float32)
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1 or psf.size == 1:
        # No decimation, or a 1x1 delta PSF: decimating a delta is a delta.
        return psf.copy()

    box = np.full((factor, factor), 1.0 / (factor * factor), dtype=np.float32)
    smoothed = blur_fft(psf, box)              # h * box, same shape
    reduced = downsample(smoothed, factor)     # pick every factor-th sample
    total = float(reduced.sum())
    if total <= 0:
        return np.array([[1.0]], dtype=np.float32)
    return (reduced / total).astype(np.float32)


# --------------------------------------------------------------------------- #
# Degradation grid loop
# --------------------------------------------------------------------------- #
BSNR_LEVELS = (40.0, 30.0, 25.0, 20.0)
DOWNSAMPLE_FACTORS = (1, 2)
BLUR_LENGTH_RANGE = (0, 21)      # inclusive
ANGLE_RANGE = (0.0, 180.0)       # half-open [0, 180)
DEFOCUS_RADIUS_RANGE = (0, 5)    # inclusive
GRID_COLUMNS = ("plate_id", "blur_length", "angle", "defocus_radius",
                "bsnr_db", "downsample_factor", "seed", "key", "true_text")


def _plate_batches(plates_npz: str | Path, sources: tuple[str, ...] | None,
                   limit: int | None) -> list[tuple[str, str, np.ndarray]]:
    """Yield ``(plate_id, true_text, image)`` from a ``plates.npz`` file."""
    prefixes = {"synthetic": "synthetic", "base": "ccpd_base", "blur": "ccpd_blur"}
    wanted = sources or tuple(prefixes)

    plates: list[tuple[str, str, np.ndarray]] = []
    with np.load(plates_npz) as data:
        for source in wanted:
            images = data[f"{source}_imgs"]
            texts = data[f"{source}_text"]
            for i, (image, text) in enumerate(zip(images, texts)):
                plates.append((f"{prefixes[source]}_{i:04d}", str(text), image))
                if limit is not None and len(plates) >= limit:
                    return plates
    return plates


def _grid_key(plate_id: str, length: int, angle: float, radius: int,
              bsnr_db: float, factor: int) -> str:
    """Reproducible short identifier for one degraded output."""
    plain = f"{plate_id}|L{length}|A{angle:.4f}|R{radius}|N{bsnr_db}|X{factor}"
    digest = hashlib.sha1(plain.encode("utf-8")).hexdigest()[:10]
    return f"{plate_id}_{digest}"


def run_degradation_grid(plates_npz: str, out_dir: str, seed: int = 0,
                         *, limit: int | None = None,
                         sources: tuple[str, ...] | None = None) -> None:
    """Degrade every clean plate across the parameter grid and log everything.

    For each plate one blur condition is drawn -- ``blur_length`` in ``[0, 21]``
    px, ``angle`` in ``[0, 180)`` degrees, ``defocus_radius`` in ``[0, 5]`` px --
    from a single seeded generator, and the plate is then degraded at every
    ``bsnr_db`` in ``{40, 30, 25, 20}`` and ``downsample_factor`` in ``{1, 2}``
    (8 outputs per plate, following the forward model ``g = D(h*f) + n``).

    Parameters
    ----------
    plates_npz:
        Path of the clean-plate archive written by ``dataset_writer``.
    out_dir:
        Destination directory; receives ``degraded.npz`` and ``degraded.csv``.
    seed:
        Seed of the generator that draws blur parameters *and* per-output noise
        seeds, so a run is fully reproducible.
    limit:
        Optional cap on the number of plates (handy for smoke runs).
    sources:
        Optional subset of ``("synthetic", "base", "blur")``.

    Returns
    -------
    None

    Notes
    -----
    ``degraded.npz`` holds, for every output key ``k``:

    * ``<k>_img`` -- the degraded plate, ``float32``, clipped to ``[0, 1]``;
    * ``<k>_psf`` -- the combined PSF **at the resolution of that image**
      (see :func:`effective_psf`).  Consume it as-is: at
      ``downsample_factor == 2`` it is about half as wide as the
      full-resolution kernel, so re-deriving the kernel from the logged
      ``blur_length`` / ``angle`` / ``defocus_radius`` would model the blur
      twice as wide as it really is.

    ``degraded.csv`` logs ``plate_id``, ``blur_length``, ``angle``,
    ``defocus_radius``, ``bsnr_db``, ``downsample_factor``, ``seed`` plus the
    ``key`` used in the archive and the plate's ``true_text``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plates = _plate_batches(plates_npz, sources, limit)
    rng = np.random.default_rng(seed)

    payload: dict[str, np.ndarray] = {}
    rows: list[tuple] = []

    for plate_id, text, clean in plates:
        # One blur condition per plate: motion (length, angle) + defocus disc.
        length = int(rng.integers(BLUR_LENGTH_RANGE[0], BLUR_LENGTH_RANGE[1] + 1))
        angle = float(rng.uniform(*ANGLE_RANGE))
        radius = int(rng.integers(DEFOCUS_RADIUS_RANGE[0], DEFOCUS_RADIUS_RANGE[1] + 1))

        combined = combine_psfs(make_motion_psf(length, angle),
                                make_defocus_psf(radius))
        blurred = blur_fft(np.asarray(clean, dtype=np.float32), combined)

        for bsnr_db in BSNR_LEVELS:
            for factor in DOWNSAMPLE_FACTORS:
                # h * f, then D(.)
                decimated = downsample(blurred, factor) if factor > 1 else blurred

                # + n, with an independent seeded noise draw per output
                noise_seed = int(rng.integers(0, 2 ** 31 - 1))
                degraded = add_noise(decimated, bsnr_db, noise_seed)

                key = _grid_key(plate_id, length, angle, radius, bsnr_db, factor)
                payload[f"{key}_img"] = np.clip(degraded, 0.0, 1.0).astype(np.float32)
                payload[f"{key}_psf"] = effective_psf(combined, factor)
                rows.append((plate_id, length, angle, radius, bsnr_db, factor,
                             noise_seed, key, text))

    np.savez_compressed(out / "degraded.npz", **payload)

    with open(out / "degraded.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GRID_COLUMNS)
        writer.writerows(rows)


def _demo_grid() -> None:  # pragma: no cover
    """Run the grid on the current dataset: python -m ...degradation --grid."""
    from pathlib import Path as _Path

    data_dir = _Path(__file__).with_name("data")
    run_degradation_grid(str(data_dir / "plates.npz"), str(data_dir / "degraded"))
    print(f"wrote {data_dir / 'degraded' / 'degraded.npz'}")


# --------------------------------------------------------------------------- #
# Manual visual check: python -m license_plate_restoration.degradation
# --------------------------------------------------------------------------- #
def _demo() -> None:  # pragma: no cover
    """Render the 'motion blur 15 px, 8 degrees, 30 dB' case from the brief."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from license_plate_restoration.renderer import generate_plates

    clean, text = generate_plates(1, seed=7)[0]
    psf = make_motion_psf(15, 8.0)
    blurred = blur_fft(clean, psf)
    noisy = add_noise(blurred, 30.0, seed=0)
    half = downsample(noisy, 2)

    fig, axes = plt.subplots(4, 1, figsize=(7, 6))
    for ax, img, title in zip(
        axes,
        (clean, blurred, noisy, half),
        (f"clean  {text}",
         f"motion blur: length 15 px, angle 8 deg  (kernel {psf.shape[0]}x{psf.shape[1]})",
         "after 30 dB BSNR noise",
         f"after downsample x2  {half.shape}"),
    ):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out = Path(__file__).with_name("data") / "preview_degradation_demo.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    print(f"bsnr target 30 dB -> measured {measure_bsnr(blurred, noisy):.2f} dB")


if __name__ == "__main__":  # pragma: no cover
    if "--grid" in sys.argv:
        _demo_grid()
    else:
        _demo()
