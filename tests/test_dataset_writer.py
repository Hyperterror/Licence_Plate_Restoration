"""Tests for Step 1.3 -- dataset persistence.

Covers unit checks (task 4.3), Property 8 (round-trip) and Property 9
(previews must not mutate the stored arrays).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import float32_image_batch_strategy
from hypothesis import given, settings

from license_plate_restoration.dataset_writer import (
    CSV_COLUMNS,
    save_dataset,
)

NPZ_KEYS = ("synthetic_imgs", "synthetic_text",
            "base_imgs", "base_text",
            "blur_imgs", "blur_text")


def _make_dataset(n_syn: int = 3, n_base: int = 2, n_blur: int = 1):
    """Return the six arrays ``save_dataset`` expects, with distinct values."""
    rng = np.random.default_rng(0)
    synth = rng.random((n_syn, 96, 336), dtype=np.float32)
    base = rng.random((n_base, 92, 256), dtype=np.float32)
    blur = rng.random((n_blur, 92, 256), dtype=np.float32)
    return (
        synth, np.array([f"KA 0{i} MN 123{i}" for i in range(n_syn)]),
        base, np.array([f"AB{i}C{i}D"[:6] for i in range(n_base)]),
        blur, np.array([f"XY{i}Z{i}W"[:6] for i in range(n_blur)]),
    )


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_npz_keys_present(tmp_path: Path) -> None:
    save_dataset(*_make_dataset(), str(tmp_path))

    with np.load(tmp_path / "plates.npz") as data:
        assert set(data.files) == set(NPZ_KEYS)
        assert data["synthetic_imgs"].shape == (3, 96, 336)
        assert data["base_imgs"].shape == (2, 92, 256)
        assert data["blur_imgs"].shape == (1, 92, 256)
        for key in NPZ_KEYS[:1] + NPZ_KEYS[2:3] + NPZ_KEYS[4:5]:
            assert data[key].dtype == np.float32


def test_csv_columns(tmp_path: Path) -> None:
    import pandas as pd

    save_dataset(*_make_dataset(), str(tmp_path))
    frame = pd.read_csv(tmp_path / "plates.csv")

    assert tuple(frame.columns) == CSV_COLUMNS
    assert len(frame) == 3 + 2 + 1
    assert set(frame["source"]) == {"synthetic", "ccpd_base", "ccpd_blur"}
    assert frame["id"].is_unique
    # texts survive the round trip unchanged
    assert frame.loc[frame["source"] == "synthetic", "true_text"].tolist() == [
        "KA 00 MN 1230", "KA 01 MN 1231", "KA 02 MN 1232"]


def test_csv_keeps_original_ccpd_filenames(tmp_path: Path) -> None:
    import pandas as pd

    synth, synth_text, base, base_text, blur, blur_text = _make_dataset()
    save_dataset(synth, synth_text, base, base_text, blur, blur_text, str(tmp_path),
                 base_filenames=["a.jpg", "b.jpg"], blur_filenames=["c.jpg"])

    frame = pd.read_csv(tmp_path / "plates.csv").set_index("id")
    assert frame.loc["ccpd_base_0001", "filename"] == "b.jpg"
    assert frame.loc["ccpd_blur_0000", "filename"] == "c.jpg"


def test_preview_files_created(tmp_path: Path) -> None:
    save_dataset(*_make_dataset(), str(tmp_path))

    for source in ("synthetic", "ccpd_base", "ccpd_blur"):
        path = tmp_path / f"preview_{source}.png"
        assert path.is_file() and path.stat().st_size > 0


def test_preview_skipped_for_empty_source(tmp_path: Path) -> None:
    empty = np.empty((0, 92, 256), dtype=np.float32)
    save_dataset(*_make_dataset(n_blur=0)[:4], empty, np.empty((0,), dtype="<U32"),
                 str(tmp_path))

    assert (tmp_path / "preview_synthetic.png").is_file()
    assert not (tmp_path / "preview_ccpd_blur.png").exists()


def test_out_of_range_values_are_clipped(tmp_path: Path) -> None:
    synth = np.full((1, 96, 336), -0.5, dtype=np.float32)
    synth[0, ::2, :] = 1.5                  # values outside [0, 1] on purpose
    save_dataset(synth, np.array(["KA 00 MN 1234"]),
                 np.zeros((0, 92, 256), np.float32), np.empty((0,), dtype="<U32"),
                 np.zeros((0, 92, 256), np.float32), np.empty((0,), dtype="<U32"),
                 str(tmp_path))

    with np.load(tmp_path / "plates.npz") as data:
        stored = data["synthetic_imgs"]
    assert stored.dtype == np.float32
    assert stored.min() >= 0.0 and stored.max() <= 1.0


# --------------------------------------------------------------------------- #
# Property tests
# --------------------------------------------------------------------------- #
# Feature: license-plate-restoration, Property 8: dataset round-trip preserves float32 values in [0,1]
@given(batch=float32_image_batch_strategy())
@settings(max_examples=20, deadline=None)
def test_dataset_roundtrip(batch: np.ndarray) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        n = len(batch)
        texts = np.array([f"KA 0{i} MN 123{i}" for i in range(n)])
        empty = np.empty((0, 4, 4), dtype=np.float32)
        save_dataset(batch, texts, empty, np.empty((0,), dtype="<U32"),
                     empty, np.empty((0,), dtype="<U32"), tmp)

        with np.load(Path(tmp) / "plates.npz") as data:
            stored = data["synthetic_imgs"]

    assert stored.dtype == np.float32
    assert stored.shape == batch.shape
    assert stored.min() >= 0.0 and stored.max() <= 1.0
    assert np.allclose(stored, batch, atol=0.0)      # lossless for float32


# Feature: license-plate-restoration, Property 9: preview does not mutate plate arrays
@given(batch=float32_image_batch_strategy())
@settings(max_examples=10, deadline=None)
def test_preview_does_not_mutate(batch: np.ndarray) -> None:
    import tempfile

    before = [row.copy() for row in batch]
    with tempfile.TemporaryDirectory() as tmp:
        n = len(batch)
        texts = np.array([f"KA 0{i} MN 123{i}" for i in range(n)])
        empty = np.empty((0, 4, 4), dtype=np.float32)
        save_dataset(batch, texts, empty, np.empty((0,), dtype="<U32"),
                     empty, np.empty((0,), dtype="<U32"), tmp)

    for original, after in zip(before, batch):
        assert np.array_equal(original, after)


def test_save_dataset_rejects_mismatched_labels(tmp_path: Path) -> None:
    synth, synth_text, base, base_text, blur, blur_text = _make_dataset()
    with pytest.raises(ValueError, match="labels"):
        save_dataset(synth, synth_text[:-1], base, base_text, blur, blur_text,
                     str(tmp_path))
