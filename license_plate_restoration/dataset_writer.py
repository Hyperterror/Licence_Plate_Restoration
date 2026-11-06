"""Step 1.3 -- persist every clean plate and its ground truth in one place.

Artifacts written into ``out_dir``::

    plates.npz                    synthetic_imgs / synthetic_text
                                  base_imgs / base_text
                                  blur_imgs / blur_text
    plates.csv                    id, source, true_text, filename
    preview_synthetic.png         grid of 20 synthetic plates
    preview_ccpd_base.png         grid of 20 ccpd_base plates
    preview_ccpd_blur.png         grid of 20 ccpd_blur plates

All image arrays are stored as ``float32`` in ``[0, 1]``.  Matplotlib is used
only to draw the preview grids, and the stored arrays are never modified.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SOURCES = ("synthetic", "ccpd_base", "ccpd_blur")
CSV_COLUMNS = ("id", "source", "true_text", "filename")
PREVIEW_GRID = (5, 4)                     # 5 rows x 4 columns = 20 plates
GREY = 128 / 255.0                        # margin value, used for padding only


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_batch(images: np.ndarray, name: str) -> np.ndarray:
    """Return ``images`` as a contiguous float32 batch clipped to ``[0, 1]``."""
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim != 3:
        raise ValueError(f"{name} must be a batch of 2-D images, got shape {batch.shape}")
    return np.clip(batch, 0.0, 1.0)


def _as_text(text: np.ndarray, count: int, name: str) -> np.ndarray:
    """Return ``text`` as a length-``count`` string array."""
    labels = np.asarray(text).astype(str).reshape(-1)
    if labels.size != count:
        raise ValueError(f"{name}: {labels.size} labels for {count} images")
    return labels


def _grid_panel(image: np.ndarray) -> np.ndarray:
    """Tightly fit ``image`` into an aspect-matched canvas for readable grids.

    Synthetic plates are 336x96 (3.5:1) and CCPD plates 256x92, but every cell
    of the preview grid has the same aspect ratio, so short plates are padded
    with mid-grey (not stretched, which would distort the glyphs).  A copy is
    returned: the caller's array is never touched.
    """
    rows, cols = image.shape
    target = 3.5                            # widest plate aspect in the dataset
    pad_total = int(round(cols / target)) - rows
    if pad_total <= 0:
        return np.array(image, dtype=np.float32, copy=True)

    top = pad_total // 2
    canvas = np.full((rows + pad_total, cols), GREY, dtype=np.float32)
    canvas[top:top + rows, :] = image
    return canvas


def _write_preview(path: Path, images: np.ndarray, texts: np.ndarray,
                   title: str, n: int = 20) -> None:
    """Write a ``5x4`` grid of the first ``n`` plates, titled with their text."""
    import matplotlib
    matplotlib.use("Agg")                   # headless backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(*PREVIEW_GRID, figsize=(12, 8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")

    for ax, image, text in zip(axes, images[:n], texts[:n]):
        ax.imshow(_grid_panel(image), cmap="gray", vmin=0.0, vmax=1.0,
                  interpolation="nearest")
        ax.set_title(str(text), fontsize=9)
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """Write ``rows`` with the canonical ``id, source, true_text, filename``."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def save_dataset(
    synthetic_imgs: np.ndarray,
    synthetic_text: np.ndarray,
    base_imgs: np.ndarray,
    base_text: np.ndarray,
    blur_imgs: np.ndarray,
    blur_text: np.ndarray,
    out_dir: str,
    *,
    base_filenames: list[str] | None = None,
    blur_filenames: list[str] | None = None,
    n_preview: int = 20,
) -> None:
    """Save all clean plates, their labels, a metadata table and preview grids.

    Parameters
    ----------
    synthetic_imgs, base_imgs, blur_imgs:
        Image batches of shape ``(N, H, W)``; anything not ``float32`` in
        ``[0, 1]`` is cast and clipped before saving.
    synthetic_text, base_text, blur_text:
        Ground-truth strings, one per image.
    out_dir:
        Destination directory (created if missing); receives ``plates.npz``,
        ``plates.csv`` and the ``preview_*.png`` files.
    base_filenames, blur_filenames:
        Optional original CCPD filenames, stored in the ``filename`` column.
        When omitted, generated identifiers are written instead.
    n_preview:
        Plates per preview grid (the grid holds at most 20).

    Returns
    -------
    None
        Matches the interface used by the other pipeline stages.

    Notes
    -----
    The six NPZ keys are fixed by the team interface: ``synthetic_imgs``,
    ``synthetic_text``, ``base_imgs``, ``base_text``, ``blur_imgs``,
    ``blur_text``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    batches = {
        "synthetic": (_as_batch(synthetic_imgs, "synthetic_imgs"),
                      _as_text(synthetic_text, len(synthetic_imgs), "synthetic_text")),
        "ccpd_base": (_as_batch(base_imgs, "base_imgs"),
                      _as_text(base_text, len(base_imgs), "base_text")),
        "ccpd_blur": (_as_batch(blur_imgs, "blur_imgs"),
                      _as_text(blur_text, len(blur_imgs), "blur_text")),
    }

    # --- arrays ---------------------------------------------------------- #
    np.savez_compressed(
        out / "plates.npz",
        synthetic_imgs=batches["synthetic"][0],
        synthetic_text=batches["synthetic"][1],
        base_imgs=batches["ccpd_base"][0],
        base_text=batches["ccpd_base"][1],
        blur_imgs=batches["ccpd_blur"][0],
        blur_text=batches["ccpd_blur"][1],
    )

    # --- metadata table --------------------------------------------------- #
    original_names = {
        "ccpd_base": base_filenames,
        "ccpd_blur": blur_filenames,
    }
    rows: list[tuple[str, str, str, str]] = []
    for source in SOURCES:
        images, texts = batches[source]
        names = original_names.get(source)
        for i, text in enumerate(texts):
            plate_id = f"{source}_{i:04d}"
            if names is not None and i < len(names):
                filename = str(names[i])
            else:
                filename = f"{plate_id}.png"     # generated identifier
            rows.append((plate_id, source, str(text), filename))
    _write_csv(out / "plates.csv", rows)

    # --- preview grids ---------------------------------------------------- #
    titles = {
        "synthetic": "Synthetic Indian plates",
        "ccpd_base": "CCPD base plates",
        "ccpd_blur": "CCPD blur plates",
    }
    for source in SOURCES:
        images, texts = batches[source]
        if len(images) == 0:                     # nothing to preview
            continue
        _write_preview(out / f"preview_{source}.png", images, texts,
                       titles[source], n=min(n_preview, len(images)))


# --------------------------------------------------------------------------- #
# Build the real dataset: python -m license_plate_restoration.dataset_writer
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # pragma: no cover
    from license_plate_restoration.ccpd_loader import sample_ccpd
    from license_plate_restoration.renderer import generate_plates

    repo = Path(__file__).resolve().parents[1]
    out = Path(__file__).with_name("data")

    synthetic = generate_plates(600, seed=0)
    synth_imgs = np.stack([img for img, _ in synthetic])
    synth_text = np.array([text for _, text in synthetic])

    base_imgs, base_text, base_names = sample_ccpd(str(repo / "CCPD2019" / "ccpd_base"), 400, seed=0)
    blur_imgs, blur_text, blur_names = sample_ccpd(str(repo / "CCPD2019" / "ccpd_blur"), 50, seed=0)

    save_dataset(synth_imgs, synth_text, base_imgs, base_text, blur_imgs, blur_text,
                 str(out), base_filenames=base_names, blur_filenames=blur_names)
    print(f"synthetic {synth_imgs.shape}  base {base_imgs.shape}  blur {blur_imgs.shape}")
    print(f"wrote {out / 'plates.npz'} and {out / 'plates.csv'}")
