"""Stage 8 Benchmarking and Restoration Evaluation Utilities.

Provides mathematically verified metrics (RMSE, PSNR, SSIM on unpadded interior),
paired degradation generation, parameter resolution for Deployable vs. Oracle runs,
readability proxy evaluation via Otsu segmentation, failure classification,
and hardware profiling metadata.
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from license_plate_restoration.binarizer import binarize, segment_characters
from license_plate_restoration.degradation import (
    add_noise,
    blur_fft,
    downsample,
    effective_psf,
    measure_bsnr,
)
from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)
from license_plate_restoration.restoration import (
    clip_unit_interval,
    cls_filter,
    inverse_filter,
    richardson_lucy_filter,
    wiener_filter,
    wiener_k_from_bsnr,
)


# --------------------------------------------------------------------------- #
# 1. Mathematical Metrics on Unpadded Interior Region
# --------------------------------------------------------------------------- #
def crop_interior(image: np.ndarray, margin: int = 12) -> np.ndarray:
    """Extract unpadded interior region Omega excluding periodic boundary band."""
    if margin <= 0:
        return image
    h, w = image.shape[:2]
    if 2 * margin >= h or 2 * margin >= w:
        raise ValueError(f"Margin {margin} too large for image shape {image.shape}")
    return image[margin:-margin, margin:-margin]


def compute_interior_rmse(estimate: np.ndarray, reference: np.ndarray,
                           margin: int = 12) -> float:
    """Calculate Root Mean Squared Error (RMSE) over the unpadded interior.

    Parameters
    ----------
    estimate:
        Restored image estimate, [0, 1] range.
    reference:
        Clean ground-truth reference image, [0, 1] range.
    margin:
        Boundary margin to exclude (12 px for full-res, 6 px for decimated).

    Returns
    -------
    float
        RMSE value, lower is better (0.0 is perfect reconstruction).
    """
    est = crop_interior(estimate, margin).astype(np.float64)
    ref = crop_interior(reference, margin).astype(np.float64)
    err = est - ref
    return float(np.sqrt(np.mean(err ** 2)))


def compute_interior_psnr(estimate: np.ndarray, reference: np.ndarray,
                           margin: int = 12) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in dB over unpadded interior.

    Uses dynamic range L = 1.0. Clamped with RMSE floor 1e-12 to prevent
    infinite PSNR on exact reconstructions.
    """
    rmse = compute_interior_rmse(estimate, reference, margin)
    clamped_rmse = max(rmse, 1e-12)
    return float(-20.0 * np.log10(clamped_rmse))


def _gaussian_kernel_1d(win_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """Generate 1-D Gaussian weighting window normalized to unit sum."""
    pad = win_size // 2
    coords = np.arange(win_size, dtype=np.float64) - pad
    g1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g1d /= np.sum(g1d)
    return g1d


def _convolve2d_separable_symmetric(img: np.ndarray, g1d: np.ndarray) -> np.ndarray:
    """Separable 2D convolution with symmetric boundary conditions (matches scipy/skimage)."""
    win_size = len(g1d)
    pad = win_size // 2

    # In scipy.ndimage, mode='reflect' corresponds to numpy mode='symmetric'
    padded = np.pad(img, pad, mode="symmetric")
    # convolve along axis 1 (rows)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, g1d, mode="valid"), axis=1, arr=padded)
    # convolve along axis 0 (columns)
    out = np.apply_along_axis(lambda m: np.convolve(m, g1d, mode="valid"), axis=0, arr=tmp)
    return out


def compute_interior_ssim(estimate: np.ndarray, reference: np.ndarray,
                           margin: int = 12, win_size: int = 11,
                           sigma: float = 1.5, K1: float = 0.01,
                           K2: float = 0.03, data_range: float = 1.0) -> float:
    """Calculate Structural Similarity Index (SSIM) over unpadded interior.

    Follows Wang et al. (2004) structural similarity formulation with
    11x11 Gaussian window (sigma=1.5), C1=(0.01*L)^2, C2=(0.03*L)^2.
    Matches skimage.metrics.structural_similarity to within 1e-6.
    """
    im1 = crop_interior(estimate, margin).astype(np.float64)
    im2 = crop_interior(reference, margin).astype(np.float64)

    h, w = im1.shape
    if h < win_size or w < win_size:
        raise ValueError(f"Cropped interior shape {im1.shape} smaller than win_size {win_size}")

    g1d = _gaussian_kernel_1d(win_size, sigma)

    ux = _convolve2d_separable_symmetric(im1, g1d)
    uy = _convolve2d_separable_symmetric(im2, g1d)

    uxx = _convolve2d_separable_symmetric(im1 * im1, g1d)
    uyy = _convolve2d_separable_symmetric(im2 * im2, g1d)
    uxy = _convolve2d_separable_symmetric(im1 * im2, g1d)

    vx = np.maximum(uxx - ux * ux, 0.0)
    vy = np.maximum(uyy - uy * uy, 0.0)
    vxy = uxy - ux * uy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    A1 = 2.0 * ux * uy + C1
    A2 = 2.0 * vxy + C2
    B1 = ux ** 2 + uy ** 2 + C1
    B2 = vx + vy + C2

    ssim_map = (A1 * A2) / (B1 * B2)
    pad = win_size // 2
    cropped_map = ssim_map[pad:-pad, pad:-pad]
    return float(np.mean(cropped_map))


def compute_all_metrics(estimate: np.ndarray, reference: np.ndarray,
                        margin: int = 12) -> dict[str, float]:
    """Compute RMSE, PSNR, and SSIM in a single call over interior Omega."""
    est = clip_unit_interval(estimate)
    ref = clip_unit_interval(reference)
    rmse = compute_interior_rmse(est, ref, margin)
    psnr = compute_interior_psnr(est, ref, margin)
    ssim = compute_interior_ssim(est, ref, margin)
    return {"rmse": rmse, "psnr": psnr, "ssim": ssim}


# --------------------------------------------------------------------------- #
# 2. Kernel and PSF Generation
# --------------------------------------------------------------------------- #
def make_benchmark_psf(kernel_name: str, factor: int = 1) -> np.ndarray:
    """Build standardized benchmark blur PSF.

    Parameters
    ----------
    kernel_name:
        "7x7" / "small" (motion length=3, angle=8.0 deg, defocus radius=0)
        "51x51" / "large" (motion length=21, angle=8.0 deg, defocus radius=4)
    factor:
        Decimation factor (1 for full res, 2 for decimated). If > 1,
        effective_psf is applied.
    """
    k = kernel_name.lower().strip()
    if k in ("7x7", "small"):
        base_psf = combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))
    elif k in ("51x51", "large"):
        base_psf = combine_psfs(make_motion_psf(21, 8.0), make_defocus_psf(4))
    else:
        raise ValueError(f"Unknown benchmark kernel name: {kernel_name}")

    if factor > 1:
        return effective_psf(base_psf, factor)
    return base_psf


# --------------------------------------------------------------------------- #
# 3. Paired Degradation Generator
# --------------------------------------------------------------------------- #
def generate_paired_degradation(
    clean_plate: np.ndarray,
    psf: np.ndarray,
    noise_type: str,
    noise_param: float | int | None,
    factor: int = 1,
    seed: int = 0,
    s_scale: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Generate degraded observation g and matching PSF for a paired trial.

    Parameters
    ----------
    clean_plate:
        Clean input image (float32, [0, 1]).
    psf:
        Full-resolution PSF before decimation.
    noise_type:
        "clean", "gaussian", or "poisson".
    noise_param:
        None for clean; float (BSNR dB) for gaussian; int (photons) for poisson.
    factor:
        Spatial downsampling factor (1 or 2).
    seed:
        Deterministic trial seed.
    s_scale:
        Headroom scale for Poisson observations (default 0.70).

    Returns
    -------
    tuple of (degraded_observation, effective_psf, degradation_metadata)
    """
    clean_f32 = np.asarray(clean_plate, dtype=np.float32)

    # 1. Blur via FFT
    blurred = blur_fft(clean_f32, psf)

    # 2. Downsample if decimated
    if factor > 1:
        decimated = downsample(blurred, factor)
        eff_psf = effective_psf(psf, factor)
        clean_eval = downsample(clean_f32, factor)
    else:
        decimated = blurred
        eff_psf = psf
        clean_eval = clean_f32

    meta: dict[str, Any] = {
        "noise_type": noise_type,
        "noise_param": noise_param,
        "factor": factor,
        "seed": seed,
        "s_scale": s_scale if noise_type == "poisson" else 1.0,
    }

    # 3. Apply noise
    if noise_type == "clean":
        obs = decimated.astype(np.float32)
        meta["measured_bsnr"] = float("inf")
    elif noise_type == "gaussian":
        bsnr_target = float(noise_param)
        obs = add_noise(decimated, bsnr_target, seed=seed)
        meta["target_bsnr"] = bsnr_target
        meta["measured_bsnr"] = measure_bsnr(decimated, obs)
    elif noise_type == "poisson":
        nominal_photons = int(noise_param)
        effective_photons = int(round(nominal_photons * s_scale))
        rng = np.random.default_rng(seed)
        counts = rng.poisson(nominal_photons * s_scale * np.clip(decimated, 0.0, 1.0))
        obs = (counts / nominal_photons).astype(np.float32)

        max_val = float(obs.max())
        min_val = float(obs.min())
        if max_val > 1.0 or min_val < 0.0:
            raise RuntimeError(
                f"Poisson observation exceeded [0, 1] range: min={min_val:.6f}, "
                f"max={max_val:.6f} at nominal_photons={nominal_photons}, seed={seed}."
            )
        meta["nominal_photons"] = nominal_photons
        meta["effective_photons"] = effective_photons
        meta["max_observation"] = max_val
        meta["min_observation"] = min_val
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return obs, eff_psf, meta


# --------------------------------------------------------------------------- #
# 4. Single Restoration Execution
# --------------------------------------------------------------------------- #
def run_single_restoration(
    method: str,
    degraded_obs: np.ndarray,
    psf: np.ndarray,
    noise_type: str,
    param_val: float | int | None = None,
    s_scale: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Execute one restoration algorithm and return (raw, clipped, runtime_ms).

    Parameters
    ----------
    method:
        "inverse", "wiener", "cls", or "richardson_lucy".
    degraded_obs:
        Degraded image observation in [0, 1].
    psf:
        Resolution-matched PSF.
    noise_type:
        "clean", "gaussian", or "poisson".
    param_val:
        Hyperparameter (eps for inverse, K for wiener, lambda for cls,
        iters for RL).
    s_scale:
        Headroom scale for Poisson observations (default 0.70).
    """
    m = method.lower().strip()
    t0 = time.perf_counter()

    if m == "inverse":
        eps = float(param_val) if param_val is not None else 1e-12
        raw = inverse_filter(degraded_obs, psf, eps=eps)
    elif m == "wiener":
        k_val = float(param_val) if param_val is not None else 1e-3
        raw = wiener_filter(degraded_obs, psf, K=k_val)
    elif m == "cls":
        lam = float(param_val) if param_val is not None else 1e-3
        raw = cls_filter(degraded_obs, psf, regularization=lam)
    elif m in ("richardson_lucy", "rl"):
        iters = int(param_val) if param_val is not None else 20
        raw = richardson_lucy_filter(degraded_obs, psf, iterations=iters)
    else:
        raise ValueError(f"Unknown restoration method: {method}")

    t1 = time.perf_counter()
    runtime_ms = float((t1 - t0) * 1000.0)

    # Post-restoration rescaling for Poisson observations
    if noise_type == "poisson":
        rescaled_raw = raw / s_scale
        clipped = clip_unit_interval(rescaled_raw)
        return rescaled_raw, clipped, runtime_ms

    clipped = clip_unit_interval(raw)
    return raw, clipped, runtime_ms


# --------------------------------------------------------------------------- #
# 5. Readability Proxy & Character Segmentation
# --------------------------------------------------------------------------- #
def evaluate_readability_proxy(image: np.ndarray,
                               expected_text: str | None = None) -> dict[str, Any]:
    """Assess character segmentability using Step 5 binarizer.

    Returns crop count, expected alphanumeric glyph count, and segmentation status.
    """
    binary = binarize(image)
    crops = segment_characters(binary)
    crop_count = len(crops)

    expected_count = None
    status = "valid"
    if expected_text is not None:
        expected_count = sum(c.isalnum() for c in expected_text)
        if crop_count == 0:
            status = "failed_empty"
        elif crop_count < expected_count:
            status = "under_segmented_merging"
        elif crop_count > expected_count:
            status = "over_segmented_fragmentation"
        else:
            status = "exact_match"
    else:
        if crop_count == 0:
            status = "failed_empty"

    return {
        "segmented_crops": crop_count,
        "expected_chars": expected_count,
        "segmentation_status": status,
    }


# --------------------------------------------------------------------------- #
# 6. Failure Classification
# --------------------------------------------------------------------------- #
def classify_failure(
    raw_restored: np.ndarray,
    clipped_restored: np.ndarray,
    reference: np.ndarray,
    degraded_rmse: float,
    margin: int = 12,
    segmented_crops: int | None = None,
    expected_chars: int | None = None,
) -> list[str]:
    """Identify and tag failure modes for an individual restoration result."""
    failures: list[str] = []

    # 1. NaN or Inf
    if not np.isfinite(raw_restored).all():
        failures.append("nan_or_infinity")
        return failures

    # 2. Excessive Output Clipping (more than 15% outside [0, 1])
    out_of_bounds = np.sum((raw_restored < 0.0) | (raw_restored > 1.0))
    pct_out = float(100.0 * out_of_bounds / raw_restored.size)
    if pct_out > 15.0:
        failures.append(f"excessive_clipping_{pct_out:.1f}pct")

    # 3. Severe Noise Amplification (restored RMSE > 1.25x degraded RMSE)
    restored_rmse = compute_interior_rmse(clipped_restored, reference, margin)
    if degraded_rmse > 0 and restored_rmse > 1.25 * degraded_rmse:
        failures.append("severe_noise_amplification")

    # 4. Ringing / High-Frequency Explosion
    ref_interior = crop_interior(reference, margin)
    rest_interior = crop_interior(clipped_restored, margin)
    ref_tv = float(np.mean(np.abs(np.diff(ref_interior, axis=0))) +
                   np.mean(np.abs(np.diff(ref_interior, axis=1))))
    rest_tv = float(np.mean(np.abs(np.diff(rest_interior, axis=0))) +
                    np.mean(np.abs(np.diff(rest_interior, axis=1))))

    if ref_tv > 1e-6:
        tv_ratio = rest_tv / ref_tv
        if tv_ratio > 2.0:
            failures.append("ringing_or_gradient_explosion")
        elif tv_ratio < 0.40:
            failures.append("oversmoothing")

    # 5. Segmentation readability anomalies
    if segmented_crops is not None and expected_chars is not None:
        if segmented_crops == 0:
            failures.append("failed_segmentation")
        elif segmented_crops < expected_chars - 1:
            failures.append("character_merging")
        elif segmented_crops > expected_chars + 2:
            failures.append("character_fragmentation")

    return failures


# --------------------------------------------------------------------------- #
# 7. Statistical Aggregation and Paired Tests
# --------------------------------------------------------------------------- #
def aggregate_metric_records(values: list[float]) -> dict[str, float]:
    """Compute mean, std, min, max across trials for a single metric."""
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def compute_paired_statistics(vals_a: list[float],
                              vals_b: list[float]) -> dict[str, float]:
    """Compute paired differences (A - B), mean difference, and paired t-stat."""
    res = compute_paired_ttest_detailed(vals_a, vals_b)
    return {
        "mean_diff": res["mean_diff"],
        "std_diff": res["std_diff"],
        "t_stat": res["t_stat"],
        "p_val": res["p_val"] if res["p_val"] is not None else 1.0,
    }


def compute_paired_ttest_detailed(
    vals_a: list[float] | np.ndarray,
    vals_b: list[float] | np.ndarray,
    name_a: str = "method_a",
    name_b: str = "method_b",
) -> dict[str, Any]:
    """Compute exact two-sided paired Student's t-test (method A - method B).

    Parameters
    ----------
    vals_a:
        Metrics for method A across paired trials.
    vals_b:
        Metrics for method B across the same paired trials.
    name_a:
        Identifier for method A.
    name_b:
        Identifier for method B.

    Returns
    -------
    dict[str, Any]
        Detailed paired comparison statistics, including df, 95% CI, and exact two-sided p-value.
    """
    a = np.asarray(vals_a, dtype=np.float64)
    b = np.asarray(vals_b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return {
            "comparison": f"{name_a} - {name_b}",
            "method_a": name_a,
            "method_b": name_b,
            "sample_count": int(a.size),
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "sem": 0.0,
            "t_stat": 0.0,
            "df": max(0, int(a.size - 1)),
            "p_val": 1.0,
            "ci_95": [0.0, 0.0],
        }

    diff = a - b
    n = int(a.size)
    df = n - 1
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    sem = float(std_d / np.sqrt(n))

    # Zero-variance handling
    if std_d < 1e-12:
        if abs(mean_d) < 1e-12:
            return {
                "comparison": f"{name_a} - {name_b}",
                "method_a": name_a,
                "method_b": name_b,
                "sample_count": n,
                "mean_diff": 0.0,
                "std_diff": 0.0,
                "sem": 0.0,
                "t_stat": 0.0,
                "df": df,
                "p_val": 1.0,
                "ci_95": [0.0, 0.0],
            }
        else:
            t_stat = float("inf") if mean_d > 0 else float("-inf")
            return {
                "comparison": f"{name_a} - {name_b}",
                "method_a": name_a,
                "method_b": name_b,
                "sample_count": n,
                "mean_diff": mean_d,
                "std_diff": 0.0,
                "sem": 0.0,
                "t_stat": t_stat,
                "df": df,
                "p_val": 0.0,
                "ci_95": [mean_d, mean_d],
            }

    # Genuine two-sided paired Student t-test using SciPy
    try:
        from scipy import stats
        res = stats.ttest_rel(a, b)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci_95 = [float(mean_d - t_crit * sem), float(mean_d + t_crit * sem)]
    except ImportError:
        # If SciPy is unavailable, calculate t_stat but mark inferential p-value unavailable
        t_stat = float(mean_d / sem)
        p_val = None
        ci_95 = None

    return {
        "comparison": f"{name_a} - {name_b}",
        "method_a": name_a,
        "method_b": name_b,
        "sample_count": n,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "sem": sem,
        "t_stat": t_stat,
        "df": df,
        "p_val": p_val,
        "ci_95": ci_95,
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction to control family-wise error rate.

    Parameters
    ----------
    p_values:
        List of unadjusted p-values for a family of hypotheses.

    Returns
    -------
    list[float]
        Monotonically adjusted p-values bounded by [0.0, 1.0].
    """
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    adjusted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for k in range(m):
        multiplier = m - k
        val = min(1.0, sorted_p[k] * multiplier)
        running_max = max(running_max, val)
        adjusted[k] = min(1.0, running_max)

    out = np.empty(m, dtype=np.float64)
    out[order] = adjusted
    return [float(x) for x in out]


ALGORITHM_PAIRWISE_ORDER = [
    ("inverse", "wiener"),
    ("inverse", "cls"),
    ("inverse", "richardson_lucy"),
    ("wiener", "cls"),
    ("wiener", "richardson_lucy"),
    ("cls", "richardson_lucy"),
]


def compute_pairwise_algorithm_comparisons(
    metric_series: dict[str, dict[str, list[float]]],
    methods: list[str],
) -> dict[str, Any]:
    """Compute paired Student t-tests between all applicable deployable restoration methods.

    Parameters
    ----------
    metric_series:
        Dictionary holding per-trial metrics (e.g. "wiener_deployable": {"psnr": [...]})
    methods:
        List of methods evaluated in the cell (e.g. ["inverse", "wiener", "cls", "richardson_lucy"])

    Returns
    -------
    dict[str, Any]
        Dictionary of paired comparisons with unadjusted and Holm-Bonferroni adjusted p-values.
    """
    results: dict[str, Any] = {}
    pair_keys = []
    p_vals = []

    for m_a, m_b in ALGORITHM_PAIRWISE_ORDER:
        key_a = f"{m_a}_deployable"
        key_b = f"{m_b}_deployable"
        if key_a in metric_series and key_b in metric_series:
            vals_a = metric_series[key_a]["psnr"]
            vals_b = metric_series[key_b]["psnr"]
            comp_key = f"{m_a}_vs_{m_b}"
            stats_dict = compute_paired_ttest_detailed(vals_a, vals_b, name_a=key_a, name_b=key_b)
            results[comp_key] = stats_dict
            pair_keys.append(comp_key)
            if stats_dict["p_val"] is not None:
                p_vals.append(stats_dict["p_val"])

    # Apply Holm-Bonferroni multiple-comparison correction across the family of tests
    if p_vals and len(p_vals) == len(pair_keys):
        adj_p = holm_bonferroni_correction(p_vals)
        for k, p_adj in zip(pair_keys, adj_p):
            results[k]["p_val_adjusted_holm"] = p_adj
            results[k]["significant_holm_05"] = bool(p_adj < 0.05)
            results[k]["correction_method"] = "Holm-Bonferroni step-down"

    return results


# --------------------------------------------------------------------------- #
# 8. Hardware and Environment Metadata
# --------------------------------------------------------------------------- #
def collect_hardware_metadata() -> dict[str, Any]:
    """Collect host environment and hardware specifications."""
    meta = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    try:
        import scipy
        meta["scipy_version"] = scipy.__version__
    except ImportError:
        meta["scipy_version"] = None

    try:
        import skimage
        meta["skimage_version"] = skimage.__version__
    except ImportError:
        meta["skimage_version"] = None

    try:
        import cv2
        meta["opencv_version"] = cv2.__version__
    except ImportError:
        meta["opencv_version"] = None

    return meta
