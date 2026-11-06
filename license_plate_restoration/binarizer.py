"""Step 5 -- binarization and character segmentation.

* :func:`otsu_threshold` -- from-scratch Otsu criterion over a 256-bin histogram
* :func:`binarize`        -- black text on white, polarity chosen from class means
* :func:`erode` / :func:`dilate` / :func:`opening` / :func:`closing` -- from
  scratch morphology (no ``cv2.erode``, no ``cv2.morphologyEx``, no
  ``scipy.ndimage``, no ``skimage``)
* :func:`segment_characters` -- connected components (OpenCV, allowed) plus the
  character-shape rules and a left-to-right sort

Conventions
-----------
Binary images are ``uint8`` with values in ``{0, 255}`` where **0 is
foreground (ink)** and **255 is background**.  That matches the grayscale
convention used everywhere else in the pipeline (0 = black = ink), so
``binarize`` needs no inversion downstream.

Border rule and polarity
-----------------------
The morphology below implements the definitions in the requirements literally:
with ink coded as 0, an output pixel is ink only if *all* kernel-covered input
pixels are ink (:func:`erode`), or if *any* of them is ink (:func:`dilate`).
Out-of-image pixels are neutral, so the reduction runs over the in-image part of
the window only -- that is exactly how OpenCV handles its default border.

Note the polarity when cross-checking against OpenCV: OpenCV's morphology treats
*white* (255) as the object, so ``erode`` here corresponds to ``cv2.erode`` on
the polarity-inverted image::

    erode(img, k)  == 255 - cv2.erode(255 - img, k)
    dilate(img, k) == 255 - cv2.dilate(255 - img, k)

which is the morphological duality used by the tests.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:                                        # OpenCV: connected components only
    import cv2
except ImportError:                         # pragma: no cover
    cv2 = None

BINS = 256
FOREGROUND = np.uint8(0)
BACKGROUND = np.uint8(255)

# Frozen team default for the ink polarity.  Requirement 8.3's literal rule
# (lower class mean = ink) is correct for the synthetic plates but inverts
# every CCPD plate, which carries white text on a dark field.  Measured on 10
# ccpd_base plates: "dark" leaves 0-2 characters, "auto" leaves 7-8.
DEFAULT_POLARITY = "auto"


# --------------------------------------------------------------------------- #
# Otsu thresholding
# --------------------------------------------------------------------------- #
def _quantise(img: np.ndarray) -> np.ndarray:
    """Map a ``[0, 1]`` float image onto integer bins ``0..255``."""
    scaled = np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * (BINS - 1)
    return scaled.astype(np.int32)           # truncation, as in the design


def _histogram(bins: np.ndarray) -> np.ndarray:
    """Count the 256 bins by hand (``np.add.at`` scatter, no ``np.histogram``)."""
    hist = np.zeros(BINS, dtype=np.float64)
    np.add.at(hist, bins.ravel(), 1.0)
    return hist


def otsu_threshold(img: np.ndarray) -> float:
    """Otsu threshold: the cut that maximises between-class variance.

    For every possible cut ``t`` the pixels are split into ``[0..t]`` and
    ``[t+1..255]``; the criterion is

    .. math:: \\sigma_b^2(t) = \\omega_0(t)\\,\\omega_1(t)\\,
              \\left(\\mu_0(t) - \\mu_1(t)\\right)^2

    where ``w`` are the class probabilities and ``mu`` the class means.  The
    split point is chosen by ``argmax`` over the 256 candidates, computed from
    manually accumulated cumulative sums.

    Parameters
    ----------
    img:
        2-D grayscale image, ``float32`` in ``[0, 1]``.

    Returns
    -------
    float
        Threshold ``t / 255`` in ``[0, 1]``.  Pixels ``<= t`` form class 0
        (the darker class), matching ``cv2.threshold``'s convention where
        ``src > t`` is foreground.
    """
    bins = _quantise(img)
    hist = _histogram(bins)
    total = hist.sum()
    if total <= 0:                            # empty image
        return 0.0

    probabilities = hist / total
    levels = np.arange(BINS, dtype=np.float64)

    # Cumulative class probability w0(t) and cumulative first moment mu(t).
    weight_below = np.cumsum(probabilities)
    moment_below = np.cumsum(probabilities * levels)
    weight_above = 1.0 - weight_below
    total_mean = moment_below[-1]

    # Guard the divisions where a class is empty.
    mean_below = np.divide(moment_below, weight_below,
                           out=np.zeros(BINS), where=weight_below > 0)
    mean_above = np.divide(total_mean - moment_below, weight_above,
                           out=np.zeros(BINS), where=weight_above > 0)

    between = weight_below * weight_above * (mean_below - mean_above) ** 2
    between[(weight_below <= 0) | (weight_above <= 0)] = 0.0

    return float(np.argmax(between)) / (BINS - 1)


def binarize(img: np.ndarray,
             polarity: str = DEFAULT_POLARITY,
             margin: int = 8) -> np.ndarray:
    """Threshold a plate into black text (0) on white background (255).

    Otsu gives the cut, but it cannot say which side is the text: the synthetic
    plates are dark text on a white plate, while CCPD plates are *white* text on
    a dark plate.  The polarity therefore has to be chosen explicitly, and
    requirement 8.3's literal rule is wrong for half the dataset.

    Parameters
    ----------
    img:
        2-D grayscale image, ``float32`` in ``[0, 1]``.
    polarity:
        ``"auto"`` (:data:`DEFAULT_POLARITY`, the team decision) makes the
        *smaller* class of the plate interior the ink.  Text always covers less
        area than the plate background, so this handles both dark text on light
        plates (synthetic) and light text on dark plates (CCPD).
        ``"dark"`` implements the literal requirement 8.3 rule (lower class
        mean is the ink) and is what you want for dark text on a light
        background.  It inverts CCPD plates and is kept only for the report's
        literal-rule comparison.
        ``"light"`` is its mirror image.
    margin:
        Width of the constant border added by the pipeline, ignored when the
        ``"auto"`` polarity is estimated.  The margin is a large, flat region
        (about 22 % of a synthetic plate) and would otherwise bias the area
        comparison towards the wrong class.

    Returns
    -------
    np.ndarray
        ``uint8`` array with values in ``{0, 255}``, same shape as ``img``.

    Raises
    ------
    ValueError
        If ``polarity`` is not one of the three accepted strings.

    Notes
    -----
    Otsu is a two-class split, so a plate with three intensity levels (black
    text, white plate, mid-grey margin) puts the margin in the darker class: the
    margin comes out as ink and forms a ring around the plate.  That ring is the
    largest connected component, so :func:`segment_characters` discards it.
    """
    if polarity not in ("auto", "dark", "light"):
        raise ValueError(
            f"polarity must be 'auto', 'dark' or 'light', got {polarity!r}")

    image = np.asarray(img, dtype=np.float32)
    bins = _quantise(image)
    cut = int(round(otsu_threshold(image) * (BINS - 1)))

    below = bins <= cut
    above = ~below

    if not below.any() or not above.any():
        # A single flat class: there is no ink to extract.
        return np.full(image.shape, BACKGROUND, dtype=np.uint8)

    if polarity == "auto":
        # Decide on the plate interior: the border/margin is a flat region that
        # is large enough to flip a whole-image area comparison.
        rows, cols = bins.shape
        margin = max(0, int(margin))
        if rows > 2 * margin and cols > 2 * margin:
            core = bins[margin:rows - margin, margin:cols - margin]
        else:
            core = bins
        below_area = int((core <= cut).sum())
        above_area = int(core.size - below_area)
        ink_is_below = below_area < above_area
    else:
        # Requirement 8.3: the class with the lower mean is the ink.
        lower_is_below = bins[below].mean() < bins[above].mean()
        ink_is_below = lower_is_below if polarity == "dark" else not lower_is_below

    foreground = below if ink_is_below else above

    return np.where(foreground, FOREGROUND, BACKGROUND).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Morphology from scratch
# --------------------------------------------------------------------------- #
def _check_kernel(kernel: np.ndarray) -> np.ndarray:
    mask = np.asarray(kernel) != 0
    if mask.ndim != 2 or not mask.any():
        raise ValueError("kernel must be a non-empty 2-D structuring element")
    return mask


def _morphology(img: np.ndarray, kernel: np.ndarray, op: str) -> np.ndarray:
    """Shared sliding-window erosion / dilation on the foreground mask."""
    image = np.asarray(img)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    mask = _check_kernel(kernel)
    kh, kw = mask.shape

    # True where the pixel is ink (0), so the reductions are plain boolean ops.
    foreground = image == 0

    # Pad with the neutral element so out-of-image positions never change the
    # result: erosion treats the outside as ink (it can only help "all"),
    # dilation treats it as background (it can only help "none").
    pad_value = op == "erode"
    padded = np.pad(
        foreground,
        ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2)),
        mode="constant",
        constant_values=pad_value,
    )

    # (H, W, kh, kw) read-only sliding window over the padded image.
    windows = sliding_window_view(padded, (kh, kw))

    if op == "erode":
        # Keep ink only if every *active* kernel position is ink; inactive
        # positions are forced True so they are neutral for ``all``.
        hit = np.all(windows | ~mask, axis=(2, 3))
    else:
        # Mark ink if any *active* kernel position is ink.
        hit = np.any(windows & mask, axis=(2, 3))

    return np.where(hit, FOREGROUND, BACKGROUND).astype(np.uint8)


def erode(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Morphological erosion: keep a pixel only if *all* covered pixels are ink.

    Shrinks foreground regions (removes specks, thins strokes).

    Parameters
    ----------
    img:
        Binary ``uint8`` image, 0 = ink.
    kernel:
        Structuring element; non-zero entries are the active ones.

    Returns
    -------
    np.ndarray
        Binary ``uint8`` image of the same shape.
    """
    return _morphology(img, kernel, "erode")


def dilate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Morphological dilation: mark a pixel if *any* covered pixel is ink.

    Grows foreground regions (fills gaps, joins broken strokes).

    Parameters
    ----------
    img:
        Binary ``uint8`` image, 0 = ink.
    kernel:
        Structuring element; non-zero entries are the active ones.

    Returns
    -------
    np.ndarray
        Binary ``uint8`` image of the same shape.
    """
    return _morphology(img, kernel, "dilate")


def opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Erosion followed by dilation: removes specks, keeps large shapes."""
    return dilate(erode(img, kernel), kernel)


def closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Dilation followed by erosion: fills pinholes, joins broken strokes."""
    return erode(dilate(img, kernel), kernel)


def square_kernel(size: int = 3) -> np.ndarray:
    """Convenience ``size x size`` box structuring element of ones."""
    size = int(size)
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return np.ones((size, size), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Character segmentation
# --------------------------------------------------------------------------- #
def character_boxes(
    binary_plate: np.ndarray,
    min_height_ratio: float = 0.20,
    max_height_ratio: float = 0.90,
    min_aspect: float = 1.0,
    max_aspect: float = 6.0,
    min_area: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Bounding boxes ``(x, y, w, h)`` of the character components, left to right.

    This is :func:`segment_characters` without the cropping, so callers (and
    tests) can inspect the positions that define the left-to-right order.
    """
    if cv2 is None:                          # pragma: no cover
        raise ImportError("OpenCV is required for connected components")

    plate = np.asarray(binary_plate)
    if plate.dtype != np.uint8:
        plate = np.clip(plate, 0, 255).astype(np.uint8)
    if plate.ndim != 2:
        raise ValueError(f"binary_plate must be 2-D, got shape {plate.shape}")

    ink = (plate == 0).astype(np.uint8)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:                           # only the background label exists
        return []

    height, width = plate.shape
    components = list(range(1, count))

    # 2. Reject the plate border / background blob first.  That is normally the
    #    largest component, but only when it really spans the plate: on a
    #    light-on-dark plate the largest ink blob can be a character, and
    #    discarding it would silently drop a glyph.
    areas = {label: int(stats[label, 4]) for label in components}
    largest = max(areas, key=lambda label: areas[label])
    lx, ly, lw, lh, _ = (int(v) for v in stats[largest])
    border_like = (
        lw >= 0.9 * width or lh >= 0.9 * height
        or lx <= 1 or ly <= 1 or lx + lw >= width - 1 or ly + lh >= height - 1
    )

    kept: list[tuple[int, int, int, int]] = []
    for label in components:
        if label == largest and border_like:
            continue
        # stats row layout: left, top, width, height, area (stable across versions)
        x, y, w, h, area = (int(v) for v in stats[label])

        if area < min_area:                                   # too small
            continue
        if not (min_height_ratio * height <= h <= max_height_ratio * height):
            continue
        aspect = h / max(w, 1)
        if not (min_aspect <= aspect <= max_aspect):           # not char-like
            continue

        kept.append((x, y, w, h))

    kept.sort(key=lambda box: (box[0], box[1]))               # left to right
    return kept


def segment_characters(
    binary_plate: np.ndarray,
    min_height_ratio: float = 0.20,
    max_height_ratio: float = 0.90,
    min_aspect: float = 1.0,
    max_aspect: float = 6.0,
    min_area: int = 50,
) -> list[np.ndarray]:
    """Extract character-sized connected components, left to right.

    Steps (requirement 10):

    1. Label the ink pixels with ``cv2.connectedComponentsWithStats``
       (8-connectivity).
    2. Drop the largest component when it is the plate border or the background
       blob, i.e. when its bounding box spans (or touches the edge of) the
       plate.  A large character is kept.
    3. Keep components whose height is between ``min_height_ratio`` and
       ``max_height_ratio`` of the plate height, whose aspect ratio
       ``height / width`` lies in ``[min_aspect, max_aspect]`` and whose area
       is at least ``min_area`` pixels.
    4. Return tight bounding-box crops sorted by x.

    Parameters
    ----------
    binary_plate:
        Binary ``uint8`` plate, 0 = ink.
    min_height_ratio, max_height_ratio, min_aspect, max_aspect, min_area:
        Shape filters, defaulting to the values in the requirements.

    Returns
    -------
    list[np.ndarray]
        ``uint8`` crops ordered left to right.  The list may be empty or hold
        more than ten items; no exception is raised for unexpected counts.

    Notes
    -----
    The height filters are relative to the height of the image passed in, which
    includes the 8 px margin.  A synthetic plate (96 px tall) needs glyphs of at
    least 19 px, so a 2x decimated plate sits close to that limit by design.
    """
    plate = np.asarray(binary_plate)
    if plate.dtype != np.uint8:
        plate = np.clip(plate, 0, 255).astype(np.uint8)

    boxes = character_boxes(plate, min_height_ratio, max_height_ratio,
                            min_aspect, max_aspect, min_area)
    return [plate[y:y + h, x:x + w].copy() for x, y, w, h in boxes]


# --------------------------------------------------------------------------- #
# Manual visual check: python -m license_plate_restoration.binarizer
# --------------------------------------------------------------------------- #
def _demo() -> None:  # pragma: no cover
    """Binarize, clean up and segment one synthetic and one CCPD plate."""
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from license_plate_restoration.renderer import generate_plates

    kernel = square_kernel(3)
    samples = [("synthetic", *generate_plates(1, seed=5)[0])]

    ccpd_npz = Path(__file__).with_name("data") / "plates.npz"
    if ccpd_npz.is_file():
        with np.load(ccpd_npz) as data:
            images, texts = data["base_imgs"], data["base_text"]
        samples.append(("ccpd_base", images[0], str(texts[0])))

    fig, axes = plt.subplots(len(samples) * 3, 1, figsize=(8, 2.2 * len(samples) * 3))
    axes = np.atleast_1d(axes).ravel()
    row = 0
    for source, image, text in samples:
        binary = binarize(image)
        cleaned = opening(closing(binary, kernel), kernel)
        crops = segment_characters(binary)

        axes[row].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row].set_title(f"{source}: {text}", fontsize=9)
        axes[row + 1].imshow(binary, cmap="gray", vmin=0, vmax=255)
        axes[row + 1].set_title(
            f"otsu t={otsu_threshold(image):.3f}  ->  {len(crops)} characters", fontsize=9)
        # character crops side by side
        if crops:
            width = sum(c.shape[1] for c in crops) + 4 * len(crops)
            strip = np.full((max(c.shape[0] for c in crops), width), 255, np.uint8)
            x = 0
            for crop in crops:
                strip[:crop.shape[0], x:x + crop.shape[1]] = crop
                x += crop.shape[1] + 4
            axes[row + 2].imshow(strip, cmap="gray", vmin=0, vmax=255)
        axes[row + 2].set_title("character crops (left to right)", fontsize=9)
        for axis in axes[row:row + 3]:
            axis.set_xticks([])
            axis.set_yticks([])
        row += 3

    fig.tight_layout()
    out = Path(__file__).with_name("data") / "preview_binarization.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    _demo()
