"""Step 1.1 -- synthetic Indian license-plate renderer.

Produces the sharp plate ``f`` of the forward model: black bold text on a
white plate with a black border, surrounded by an 8 px grey margin.

Library policy (requirement 1.3): PIL is used *only* to rasterise the text
glyphs.  Every other pixel -- margin, white background, black border -- is
written with NumPy slicing.

Geometry::

    +----------------------------------+  <- 336 x 96 float32, margin = 128/255
    |  ##############################  |     (8 px grey margin on all sides)
    |  #                            #  |
    |  #      KA 05 MN 1234         #  |     inner plate = 320 x 80, white
    |  #                            #  |     border = 2 px black, inside it
    |  ##############################  |
    +----------------------------------+
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Geometry and fixed constants
# --------------------------------------------------------------------------- #
PLATE_W, PLATE_H = 320, 80           # inner plate area (pixels)
MARGIN = 8                           # grey border width (pixels)
OUT_W, OUT_H = PLATE_W + 2 * MARGIN, PLATE_H + 2 * MARGIN   # 336 x 96
GREY = 128 / 255.0                   # margin fill value, ~0.502
BORDER_PX = 2                        # black frame drawn inside the plate

# Real Indian state / union-territory registration codes.
STATE_CODES = [
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA", "KL",
    "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "AN", "CH", "DN", "DD", "DL", "LD", "PY",
]

# `AA 00 AA 0000` -- state code, RTO district, series, serial number.
PLATE_PATTERN = re.compile(r"^([A-Z]{2}) ([0-9]{2}) ([A-Z]{2}) ([0-9]{4})$")

# Series letters: I and O are skipped because Indian plates never use them
# (they are confused with 1 and 0).  The result still matches ``[A-Z]{2}``.
SERIAL_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

# Bold TrueType fonts, in order of preference.  The first existing one wins and
# is used for the whole dataset so that rendering stays reproducible.
_FONT_CANDIDATES = (
    "C:/Windows/Fonts/arialbd.ttf",                                  # Windows
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",        # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",          # Linux
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",             # macOS
)

_START_SIZE = int(PLATE_H * 0.62)    # first font size tried, then shrunk
_MIN_SIZE = 8
_TEXT_PAD = BORDER_PX + 3            # clearance kept between text and border


# --------------------------------------------------------------------------- #
# Font helpers (PIL is used for glyph rasterisation only)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _font_path() -> str | None:
    """Return the path of the first available bold font, or ``None``."""
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


@lru_cache(maxsize=64)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load the bold font at ``size`` pixels (cached: deterministic + fast)."""
    path = _font_path()
    if path is not None:
        return ImageFont.truetype(path, size)
    # Last resort so the module still runs; Pillow >= 10 accepts a size.
    return ImageFont.load_default(size=size)


def _text_extent(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    """Bounding box ``(left, top, right, bottom)`` of ``text`` drawn at (0, 0)."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    return probe.textbbox((0, 0), text, font=font)


def _fit_font(text: str) -> ImageFont.FreeTypeFont:
    """Largest font size (<= ``_START_SIZE``) whose text fits the plate.

    Shrinking one pixel at a time keeps the code branch-free and fully
    deterministic -- no randomness, so ``render_plate`` is reproducible.
    """
    max_w = PLATE_W - 2 * _TEXT_PAD
    max_h = PLATE_H - 2 * _TEXT_PAD
    for size in range(_START_SIZE, _MIN_SIZE - 1, -1):
        font = _load_font(size)
        left, top, right, bottom = _text_extent(text, font)
        if (right - left) <= max_w and (bottom - top) <= max_h:
            return font
    return _load_font(_MIN_SIZE)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def render_plate(text: str, seed: int = 0) -> np.ndarray:
    """Render one synthetic Indian plate.

    Parameters
    ----------
    text:
        Registration number in the form ``AA 00 AA 0000`` (e.g. ``KA 05 MN 1234``).
        The two-letter prefix must be a real ``STATE_CODES`` entry.
    seed:
        Kept for interface compatibility and future use; the current layout is
        fixed, so the seed does not change the pixels.  This is what makes the
        output deterministic for a given ``text`` (requirement 1.4).

    Returns
    -------
    np.ndarray
        ``float32`` array of shape ``(96, 336)`` with values in ``[0, 1]``:
        320x80 white plate with a 2 px black frame and centred black bold text,
        surrounded by an 8 px grey margin.

    Raises
    ------
    ValueError
        If ``text`` does not match ``^[A-Z]{2} [0-9]{2} [A-Z]{2} [0-9]{4}$``
        or its state code is not a valid Indian state code.
    """
    match = PLATE_PATTERN.match(text)
    if match is None:
        raise ValueError(
            f"invalid plate text {text!r}: expected the format "
            f"'AA 00 AA 0000' (state code, 2 digits, 2 letters, 4 digits)"
        )
    state_code = match.group(1)
    if state_code not in STATE_CODES:
        raise ValueError(
            f"invalid state code {state_code!r} in plate text {text!r}: "
            f"not a real Indian state / UT registration code"
        )

    # 1. Grey canvas everywhere (the 8 px margin).
    plate = np.full((OUT_H, OUT_W), GREY, dtype=np.float32)

    # 2. Inner 320x80 plate area, white.
    r0, c0 = MARGIN, MARGIN
    plate[r0:r0 + PLATE_H, c0:c0 + PLATE_W] = 1.0

    # 3. 2 px black frame drawn just inside the white area.
    r1, c1 = r0 + PLATE_H, c0 + PLATE_W
    plate[r0:r0 + BORDER_PX, c0:c1] = 0.0          # top edge
    plate[r1 - BORDER_PX:r1, c0:c1] = 0.0          # bottom edge
    plate[r0:r1, c0:c0 + BORDER_PX] = 0.0          # left edge
    plate[r0:r1, c1 - BORDER_PX:c1] = 0.0          # right edge

    # 4. Rasterise the text with PIL on a transparent-white layer.
    font = _fit_font(text)
    canvas = Image.new("L", (PLATE_W, PLATE_H), 255)      # white background
    left, top, right, bottom = _text_extent(text, font)
    # Centre the glyph box (offset by the bbox origin so descenders stay centred).
    x = (PLATE_W - (right - left)) // 2 - left
    y = (PLATE_H - (bottom - top)) // 2 - top
    ImageDraw.Draw(canvas).text((x, y), text, font=font, fill=0)

    glyphs = np.asarray(canvas, dtype=np.float32) / 255.0

    # 5. Composite with ``minimum``: glyph pixels (0.0) darken the white plate,
    #    background pixels (1.0) leave it untouched, and the frame survives
    #    because ``min(0.0, 1.0) == 0.0``.
    inner = plate[r0:r1, c0:c1]
    plate[r0:r1, c0:c1] = np.minimum(inner, glyphs)

    return np.ascontiguousarray(plate, dtype=np.float32)


def generate_plates(n: int, seed: int = 0) -> list[tuple[np.ndarray, str]]:
    """Generate ``n`` plates with random but valid Indian registration numbers.

    All texts match ``^[A-Z]{2} [0-9]{2} [A-Z]{2} [0-9]{4}$`` with a real state
    code, and are unique within one call.

    Parameters
    ----------
    n:
        Number of plates.  ``n <= 0`` yields an empty list.
    seed:
        Seed of the single ``numpy.random.default_rng`` used to draw state
        codes, district numbers, series letters and serial numbers.

    Returns
    -------
    list[tuple[np.ndarray, str]]
        ``(image, text)`` pairs, image shape ``(96, 336)``, dtype ``float32``.
    """
    if n <= 0:
        return []

    rng = np.random.default_rng(seed)
    plates: list[tuple[np.ndarray, str]] = []
    seen: set[str] = set()

    while len(plates) < n:
        state = STATE_CODES[int(rng.integers(len(STATE_CODES)))]
        district = int(rng.integers(0, 100))                 # RTO: 00-99
        series = (SERIAL_LETTERS[int(rng.integers(len(SERIAL_LETTERS)))]
                  + SERIAL_LETTERS[int(rng.integers(len(SERIAL_LETTERS)))])
        serial = int(rng.integers(0, 10000))                 # 0000-9999
        text = f"{state} {district:02d} {series} {serial:04d}"

        if text in seen:        # keep the dataset duplicate-free
            continue
        seen.add(text)
        plates.append((render_plate(text, seed), text))

    return plates


# --------------------------------------------------------------------------- #
# Manual visual check: python -m license_plate_restoration.renderer
# --------------------------------------------------------------------------- #
def _write_preview(path: str, n: int = 6, seed: int = 42) -> None:
    """Save a labelled grid of generated plates (Matplotlib: plots only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = generate_plates(n, seed)
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.35 * n))
    for ax, (img, text) in zip(np.atleast_1d(axes), samples):
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(text, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}  ({len(samples)} plates, shape {samples[0][0].shape})")


if __name__ == "__main__":  # pragma: no cover
    _write_preview(str(Path(__file__).with_name("data") / "preview_renderer.png"))
