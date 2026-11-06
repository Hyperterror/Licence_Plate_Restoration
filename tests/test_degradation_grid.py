"""Tests for Step 2.3 -- the degradation grid loop.

Covers the logging/normalisation requirements (tasks 8.2, 8.3) and the grid's
reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from license_plate_restoration.dataset_writer import save_dataset
from license_plate_restoration.degradation import (
    BSNR_LEVELS,
    DOWNSAMPLE_FACTORS,
    GRID_COLUMNS,
    effective_psf,
    run_degradation_grid,
)
from license_plate_restoration.psf_builder import make_defocus_psf, make_motion_psf

N_PLATES = 3


def _write_plates(tmp_path: Path) -> Path:
    """Save a tiny clean-plate archive the grid can consume."""
    rng = np.random.default_rng(0)
    synth = rng.random((N_PLATES, 96, 336), dtype=np.float32)
    texts = np.array([f"KA 0{i} MN 123{i}" for i in range(N_PLATES)])
    empty_imgs = np.empty((0, 92, 256), dtype=np.float32)
    empty_text = np.empty((0,), dtype="<U32")
    save_dataset(synth, texts, empty_imgs, empty_text, empty_imgs, empty_text,
                 str(tmp_path))
    return tmp_path / "plates.npz"


# --------------------------------------------------------------------------- #
# effective_psf
# --------------------------------------------------------------------------- #
def test_effective_psf_factor1_is_copy() -> None:
    psf = make_motion_psf(3, 10.0)
    out = effective_psf(psf, 1)

    assert np.array_equal(out, psf)
    assert out is not psf


def test_effective_psf_factor2_shrinks_and_normalises() -> None:
    psf = make_motion_psf(6, 0.0)            # 13 x 13
    out = effective_psf(psf, 2)

    assert out.shape == (6, 6)               # 13 // 2
    assert out.dtype == np.float32
    assert abs(float(out.sum()) - 1.0) < 1e-5
    # half the smear width in low-resolution pixels
    assert out.shape[1] < psf.shape[1]


def test_effective_psf_factor1_of_defocus_is_normalised() -> None:
    psf = make_defocus_psf(2.5)
    assert abs(float(effective_psf(psf, 1).sum()) - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Grid loop
# --------------------------------------------------------------------------- #
def test_grid_writes_outputs_and_log(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    out = tmp_path / "degraded"

    run_degradation_grid(str(plates), str(out), seed=0)

    assert (out / "degraded.npz").is_file()
    log = pd.read_csv(out / "degraded.csv")
    assert set(GRID_COLUMNS).issubset(log.columns)
    assert len(log) == N_PLATES * len(BSNR_LEVELS) * len(DOWNSAMPLE_FACTORS)
    assert log["key"].is_unique
    assert set(log["bsnr_db"]) == set(BSNR_LEVELS)
    assert set(log["downsample_factor"]) == set(DOWNSAMPLE_FACTORS)
    assert log["plate_id"].nunique() == N_PLATES
    assert log["true_text"].str.match(r"^[A-Z]{2} [0-9]{2} [A-Z]{2} [0-9]{4}$").all()


def test_grid_log_has_all_seven_required_fields(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    run_degradation_grid(str(plates), str(tmp_path / "degraded"), seed=2)

    log = pd.read_csv(tmp_path / "degraded" / "degraded.csv")
    required = ("plate_id", "blur_length", "angle", "defocus_radius",
                "bsnr_db", "downsample_factor", "seed")
    for column in required:
        assert column in log.columns
        assert log[column].notna().all()
    assert log["blur_length"].between(0, 21).all()
    assert (log["angle"] >= 0.0).all() and (log["angle"] < 180.0).all()
    assert log["defocus_radius"].between(0, 5).all()


def test_grid_images_and_psfs_are_consistent(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    out = tmp_path / "degraded"
    run_degradation_grid(str(plates), str(out), seed=3)

    log = pd.read_csv(out / "degraded.csv")
    native_psf_size: dict[str, int] = {}

    with np.load(out / "degraded.npz") as data:
        for _, row in log.iterrows():
            key = row["key"]
            image = data[f"{key}_img"]
            psf = data[f"{key}_psf"]

            assert image.dtype == np.float32
            assert image.min() >= 0.0 and image.max() <= 1.0
            expected = (96, 336) if row["downsample_factor"] == 1 else (48, 168)
            assert image.shape == expected

            assert psf.dtype == np.float32
            assert abs(float(psf.sum()) - 1.0) < 1e-5

            # The stored kernel must be expressed in the image's own pixels, so
            # the factor-2 kernel is about half the width of the native one.
            if row["downsample_factor"] == 1:
                native_psf_size[row["plate_id"]] = psf.shape[0]
            else:
                native = native_psf_size[row["plate_id"]]
                assert abs(psf.shape[0] - native / 2) <= 1


def test_grid_keeps_psf_per_blur_condition(tmp_path: Path) -> None:
    """The 8 outputs of one plate share one blur condition, hence one PSF shape."""
    plates = _write_plates(tmp_path)
    out = tmp_path / "degraded"
    run_degradation_grid(str(plates), str(out), seed=4)

    log = pd.read_csv(out / "degraded.csv")
    for _, group in log.groupby("plate_id"):
        assert group["blur_length"].nunique() == 1
        assert group["angle"].nunique() == 1
        assert group["defocus_radius"].nunique() == 1


def test_grid_is_reproducible(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    first, second = tmp_path / "a", tmp_path / "b"

    run_degradation_grid(str(plates), str(first), seed=7)
    run_degradation_grid(str(plates), str(second), seed=7)

    log_a = pd.read_csv(first / "degraded.csv")
    log_b = pd.read_csv(second / "degraded.csv")
    pd.testing.assert_frame_equal(log_a, log_b)

    with np.load(first / "degraded.npz") as a, np.load(second / "degraded.npz") as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            assert np.array_equal(a[key], b[key]), key


def test_grid_limit(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    out = tmp_path / "degraded"
    run_degradation_grid(str(plates), str(out), seed=0, limit=1)

    log = pd.read_csv(out / "degraded.csv")
    assert log["plate_id"].nunique() == 1
    assert len(log) == len(BSNR_LEVELS) * len(DOWNSAMPLE_FACTORS)


def test_grid_rejects_unknown_source(tmp_path: Path) -> None:
    plates = _write_plates(tmp_path)
    with pytest.raises(KeyError):
        run_degradation_grid(str(plates), str(tmp_path / "x"), sources=("nope",))
