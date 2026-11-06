"""Step 1.2 -- CCPD plate loading and rectification.

CCPD filenames encode everything needed to crop the plate (the official CCPD
README documents the layout)::

    <area>-<tilt>-<bbox>-<4 vertices>-<7 char indices>-<brightness>-<blur>jpg

Example::

    00245450191571-90_90-276&480_370&520-364&524_274&519_275&486_365&491-0_0_30_29_10_30_26-80-16.jpg
                    ^bbox          ^vertices (x&y pairs)       ^char indices

The seven indices are ``[province, alphabet, ad1, ad2, ad3, ad4, ad5]``.  The
province index points at a *Chinese* character and is dropped, so the returned
ground truth is always the six Latin characters ``[alphabet, ad1..ad5]``.

OpenCV is used only for image I/O (``imread``) and perspective warping
(``getPerspectiveTransform`` + ``warpPerspective``).  The 8 px grey margin is
built with NumPy.

Output convention: ``float32`` array of shape ``(92, 256)`` in ``[0, 1]``
(straightened 240x76 plate + 8 px margin on each side).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# CCPD character tables (copied from the official CCPD toolkit)
# --------------------------------------------------------------------------- #
# 24 letters (no I, no O) followed by "O"; index 0 of this table is the first
# letter *after* the province character.
ALPHABETS: list[str] = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]
# 24 letters, then 10 digits, then "O" -> 35 entries.
ADS: list[str] = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") + ["O"]

# --------------------------------------------------------------------------- #
# Geometry and fixed constants
# --------------------------------------------------------------------------- #
PLATE_W, PLATE_H = 240, 76           # straightened plate size (pixels)
MARGIN = 8                           # grey margin width (pixels)
OUT_W, OUT_H = PLATE_W + 2 * MARGIN, PLATE_H + 2 * MARGIN   # 256 x 92
GREY = 128 / 255.0                   # same margin value as the synthetic plates

# CCPD vertices are listed bottom-right -> bottom-left -> top-left -> top-right,
# so the matching destinations are the corners of the target rectangle in that
# same order.  ``-1`` keeps the 1-pixel border visible after warping.
_DST_CORNERS = np.array(
    [[PLATE_W - 1, PLATE_H - 1],      # bottom-right
     [0, PLATE_H - 1],                # bottom-left
     [0, 0],                          # top-left
     [PLATE_W - 1, 0]],               # top-right
    dtype=np.float32,
)


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def parse_ccpd_filename(filename: str) -> tuple[np.ndarray, str] | None:
    """Decode the four plate corners and the ground-truth text from a name.

    Parameters
    ----------
    filename:
        CCPD file name or full path; only the stem is inspected.

    Returns
    -------
    tuple[np.ndarray, str] | None
        ``(corners, text)`` where ``corners`` is a ``(4, 2)`` float32 array of
        ``(x, y)`` pairs (bottom-right, bottom-left, top-left, top-right) and
        ``text`` is the six-character ground truth.  ``None`` if the name is
        malformed or an index falls outside the character tables.
    """
    fields = Path(filename).stem.split("-")
    if len(fields) < 5:                      # need at least vertex + index fields
        return None

    try:
        pairs = [vertex.split("&") for vertex in fields[3].split("_")]
        corners = np.array([[int(x), int(y)] for x, y in pairs], dtype=np.float32)
        indices = [int(i) for i in fields[4].split("_")]
    except ValueError:                       # non-numeric garbage
        return None

    if corners.shape != (4, 2) or len(indices) != 7:
        return None
    if not (0 <= indices[1] < len(ALPHABETS)):
        return None
    if any(not (0 <= i < len(ADS)) for i in indices[2:]):
        return None

    # Drop index 0 (Chinese province character), decode the remaining six.
    text = ALPHABETS[indices[1]] + "".join(ADS[i] for i in indices[2:])
    return corners, text


# --------------------------------------------------------------------------- #
# Single-plate loading
# --------------------------------------------------------------------------- #
def load_ccpd_plate(image_path: str) -> tuple[np.ndarray, str] | None:
    """Load one CCPD image, straighten the plate and attach the margin.

    Parameters
    ----------
    image_path:
        Path to a CCPD ``.jpg`` whose name follows the CCPD convention.

    Returns
    -------
    tuple[np.ndarray, str] | None
        ``(plate, text)`` with ``plate`` float32 ``(92, 256)`` in ``[0, 1]`` and
        ``text`` the six-character ground truth, or ``None`` if the file is
        unreadable or its name cannot be parsed.  Never raises for bad input.
    """
    parsed = parse_ccpd_filename(image_path)
    if parsed is None:
        return None
    corners, text = parsed

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:                        # missing file / not an image
        return None

    try:
        matrix = cv2.getPerspectiveTransform(corners, _DST_CORNERS)
        plate = cv2.warpPerspective(image, matrix, (PLATE_W, PLATE_H),
                                    flags=cv2.INTER_CUBIC)
    except cv2.error:                        # degenerate (collinear) corners
        return None

    # Grey margin written with NumPy (no cv2.copyMakeBorder needed).
    out = np.full((OUT_H, OUT_W), GREY, dtype=np.float32)
    out[MARGIN:MARGIN + PLATE_H, MARGIN:MARGIN + PLATE_W] = (
        plate.astype(np.float32) / 255.0
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32), text


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_ccpd(folder: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load ``n`` plates from a CCPD folder in a reproducible random order.

    Parameters
    ----------
    folder:
        Directory such as ``CCPD2019/ccpd_base``.
    n:
        Number of plates wanted.
    seed:
        Seed for the single ``numpy.random.default_rng`` that shuffles the file
        list, so the same ``(folder, n, seed)`` always yields the same plates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        ``(images, texts, filenames)`` with ``images`` of shape ``(k, 92, 256)``
        float32, ``texts`` of shape ``(k,)`` and ``len(filenames) == k``.  ``k``
        equals ``n`` unless the folder ran out of parseable files; malformed
        files are skipped silently.
    """
    if n <= 0:
        return (np.empty((0, OUT_H, OUT_W), dtype=np.float32),
                np.empty((0,), dtype="<U32"), [])

    files = sorted(Path(folder).glob("*.jpg"))
    order = np.random.default_rng(seed).permutation(len(files))

    images: list[np.ndarray] = []
    texts: list[str] = []
    names: list[str] = []

    for idx in order:
        path = files[int(idx)]
        loaded = load_ccpd_plate(str(path))
        if loaded is None:                   # malformed name or unreadable file
            continue
        images.append(loaded[0])
        texts.append(loaded[1])
        names.append(path.name)
        if len(images) == n:
            break

    if not images:                           # nothing parseable in the folder
        return (np.empty((0, OUT_H, OUT_W), dtype=np.float32),
                np.empty((0,), dtype="<U32"), [])

    return (np.stack(images).astype(np.float32),
            np.array(texts, dtype="<U32"),
            names)


# --------------------------------------------------------------------------- #
# Manual visual check: python -m license_plate_restoration.ccpd_loader
# --------------------------------------------------------------------------- #
def _write_preview(root: str = "CCPD2019", n: int = 8, seed: int = 0) -> None:
    """Save a labelled grid of straightened CCPD plates (plots only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imgs, texts, names = sample_ccpd(str(Path(root) / "ccpd_base"), n, seed)
    fig, axes = plt.subplots(len(imgs), 1, figsize=(7, 1.3 * len(imgs)))
    for ax, img, text in zip(np.atleast_1d(axes), imgs, texts):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(str(text), fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    out = Path(__file__).with_name("data") / "preview_ccpd_warp.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}  ({len(imgs)} plates, shape {imgs.shape})")


if __name__ == "__main__":  # pragma: no cover
    _write_preview()
