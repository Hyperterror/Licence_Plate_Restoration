"""End-to-end integration tests and source-level library guards.

Task 14: wire steps 1 -> 2 -> 5 together on a handful of plates and confirm the
handover artifacts exist and are consistent.  The guard tests additionally
enforce the project's library rules by inspecting the implementation source.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from license_plate_restoration.binarizer import binarize, closing, opening, segment_characters, square_kernel
from license_plate_restoration.dataset_writer import save_dataset
from license_plate_restoration.degradation import GRID_COLUMNS, run_degradation_grid
from license_plate_restoration.renderer import generate_plates

PACKAGE = Path(__file__).resolve().parents[1] / "license_plate_restoration"
MODULES = sorted(PACKAGE.glob("*.py"))

PREVIEWS = ("preview_synthetic.png", "preview_ccpd_base.png", "preview_ccpd_blur.png")


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def test_full_pipeline_smoke(tmp_path: Path) -> None:
    # Step 1.1 -> 1.3
    plates = generate_plates(5, seed=42)
    images = np.stack([img for img, _ in plates])
    texts = np.array([text for _, text in plates])

    empty_imgs = np.empty((0, 92, 256), dtype=np.float32)
    empty_text = np.empty((0,), dtype="<U32")
    save_dataset(images, texts, empty_imgs, empty_text, empty_imgs, empty_text,
                 str(tmp_path))

    assert (tmp_path / "plates.npz").is_file()
    assert (tmp_path / "plates.csv").is_file()

    # Step 2.3
    degraded_dir = tmp_path / "degraded"
    run_degradation_grid(str(tmp_path / "plates.npz"), str(degraded_dir), seed=0)

    assert (degraded_dir / "degraded.npz").is_file()
    log = pd.read_csv(degraded_dir / "degraded.csv")
    assert len(log) >= 1
    assert set(GRID_COLUMNS).issubset(log.columns)
    assert log["plate_id"].nunique() == 5
    assert len(log) == 5 * 4 * 2                     # 4 BSNR levels x 2 factors


def test_pipeline_degraded_then_binarized(tmp_path: Path) -> None:
    """A degraded plate must survive binarization + morphology + segmentation."""
    from license_plate_restoration.degradation import add_noise, blur_fft
    from license_plate_restoration.psf_builder import make_motion_psf

    clean, text = generate_plates(1, seed=4)[0]
    degraded = add_noise(blur_fft(clean, make_motion_psf(1, 5.0)), 40.0, seed=0)

    binary = binarize(degraded)
    kernel = square_kernel(3)
    cleaned = opening(closing(binary, kernel), kernel)
    crops = segment_characters(binary)

    assert binary.shape == clean.shape
    assert set(np.unique(binary)) == {0, 255}
    assert set(np.unique(cleaned)) <= {0, 255}
    assert len(crops) == sum(character.isalnum() for character in text)


def test_preview_files_created(tmp_path: Path) -> None:
    plates = generate_plates(2, seed=1)
    images = np.stack([img for img, _ in plates])
    texts = np.array([text for _, text in plates])

    base = np.stack([img for img, _ in generate_plates(2, seed=2)])
    blur = np.stack([img for img, _ in generate_plates(2, seed=3)])
    save_dataset(images, texts, base, texts, blur, texts, str(tmp_path))

    for name in PREVIEWS:
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0, name


def test_degraded_log_has_all_fields(tmp_path: Path) -> None:
    plates = generate_plates(2, seed=5)
    images = np.stack([img for img, _ in plates])
    texts = np.array([text for _, text in plates])
    empty_imgs = np.empty((0, 92, 256), dtype=np.float32)
    empty_text = np.empty((0,), dtype="<U32")
    save_dataset(images, texts, empty_imgs, empty_text, empty_imgs, empty_text,
                 str(tmp_path))

    run_degradation_grid(str(tmp_path / "plates.npz"), str(tmp_path / "d"), seed=1)
    log = pd.read_csv(tmp_path / "d" / "degraded.csv")

    for column in ("plate_id", "blur_length", "angle", "defocus_radius",
                   "bsnr_db", "downsample_factor", "seed"):
        assert column in log.columns
        assert log[column].notna().all()


# --------------------------------------------------------------------------- #
# Library guards (enforced on the implementation source, not on tests)
# --------------------------------------------------------------------------- #
# The proposal allows OpenCV for image I/O, rotation and connected components.
ALLOWED_CV2 = {
    "imread", "IMREAD_GRAYSCALE",              # image I/O
    "getPerspectiveTransform", "warpPerspective", "INTER_CUBIC",   # straightening
    "getRotationMatrix2D", "warpAffine", "INTER_LINEAR",          # rotation (step 3)
    "ROTATE_90_CLOCKWISE", "ROTATE_180", "ROTATE_90_COUNTERCLOCKWISE",
    "connectedComponentsWithStats",            # connected components
    "error",                                   # exception type
}
# Calls that must never appear in the implementation.
FORBIDDEN_NUMPY = {"histogram", "convolve", "correlate", "percentile"}
FORBIDDEN_ROOTS = {"scipy", "skimage", "sklearn", "tensorflow", "torch", "keras"}


def _iter_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            base = node.func.value
            root = getattr(base, "id", getattr(base, "attr", None))
            yield root, node.func.attr


def test_no_forbidden_library_calls() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root, attr in _iter_calls(tree):
            assert not (root == "np" and attr in FORBIDDEN_NUMPY), \
                f"{path.name} uses np.{attr}"
            assert not (root == "numpy" and attr in FORBIDDEN_NUMPY), \
                f"{path.name} uses numpy.{attr}"


def test_no_forbidden_imports() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (roots & FORBIDDEN_ROOTS), f"{path.name} imports {roots & FORBIDDEN_ROOTS}"


def test_only_allowed_opencv_functions() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "cv2":
                assert node.attr in ALLOWED_CV2, \
                    f"{path.name} uses cv2.{node.attr}, which is outside the allowed set"


def test_morphology_and_thresholding_are_home_grown() -> None:
    """The from-scratch primitives must be defined in the package itself."""
    from license_plate_restoration import binarizer, degradation, psf_builder

    for module, names in (
        (binarizer, ("otsu_threshold", "binarize", "erode", "dilate", "opening", "closing")),
        (degradation, ("blur_fft", "downsample", "add_noise")),
        (psf_builder, ("make_motion_psf", "make_defocus_psf")),
    ):
        for name in names:
            assert callable(getattr(module, name)), f"{module.__name__}.{name} missing"
