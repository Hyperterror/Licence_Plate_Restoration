"""Unit and regression tests for Stage 8 Benchmarking utilities."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.stage8_utils import (
    aggregate_metric_records,
    classify_failure,
    compute_all_metrics,
    compute_interior_psnr,
    compute_interior_rmse,
    compute_interior_ssim,
    compute_paired_statistics,
    compute_paired_ttest_detailed,
    compute_pairwise_algorithm_comparisons,
    crop_interior,
    evaluate_readability_proxy,
    generate_paired_degradation,
    holm_bonferroni_correction,
    make_benchmark_psf,
    run_single_restoration,
)
from license_plate_restoration.renderer import generate_plates


# --------------------------------------------------------------------------- #
# 1. Metric Verification Tests
# --------------------------------------------------------------------------- #
def test_interior_rmse_identity_and_constant_shift() -> None:
    ref = np.zeros((96, 336), dtype=np.float32)
    # Perfect match
    assert compute_interior_rmse(ref, ref, margin=12) == 0.0

    # Constant shift of 0.05
    est = ref + 0.05
    assert np.isclose(compute_interior_rmse(est, ref, margin=12), 0.05, atol=1e-6)


def test_interior_margin_exclusion() -> None:
    ref = np.zeros((96, 336), dtype=np.float32)
    est = ref.copy()
    # Add severe corruption only in the margin region (e.g. top 5 rows)
    est[:10, :] = 1.0
    est[:, :10] = 1.0

    # Interior RMSE must be exactly 0.0 because margin=12 excludes the corrupted border
    assert compute_interior_rmse(est, ref, margin=12) == 0.0


def test_interior_psnr_values() -> None:
    ref = np.zeros((96, 336), dtype=np.float32)
    # Exact match clamped to 240 dB
    assert compute_interior_psnr(ref, ref, margin=12) == 240.0

    # RMSE = 0.1 -> PSNR = -20 * log10(0.1) = 20.0 dB
    assert np.isclose(compute_interior_psnr(ref + 0.1, ref, margin=12), 20.0, atol=1e-4)

    # RMSE = 0.01 -> PSNR = -20 * log10(0.01) = 40.0 dB
    assert np.isclose(compute_interior_psnr(ref + 0.01, ref, margin=12), 40.0, atol=1e-4)


def test_interior_ssim_identity_and_cross_check() -> None:
    from skimage.metrics import structural_similarity

    rng = np.random.default_rng(123)
    plate = rng.uniform(0.1, 0.9, (96, 336)).astype(np.float32)

    # Identity must be 1.0
    s_ident = compute_interior_ssim(plate, plate, margin=12)
    assert np.isclose(s_ident, 1.0, atol=1e-5)

    # Slightly noisy image
    noisy = np.clip(plate + rng.normal(0, 0.02, plate.shape), 0.0, 1.0).astype(np.float32)
    s_calc = compute_interior_ssim(noisy, plate, margin=12)
    assert 0.0 < s_calc < 1.0

    # Cross-check against skimage on the unpadded interior
    ref_int = crop_interior(plate, margin=12)
    noisy_int = crop_interior(noisy, margin=12)
    s_skimage = structural_similarity(
        noisy_int, ref_int, data_range=1.0, win_size=11, gaussian_weights=True, sigma=1.5
    )
    assert np.isclose(s_calc, s_skimage, atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. Kernel and Degradation Tests
# --------------------------------------------------------------------------- #
def test_make_benchmark_psf() -> None:
    psf_small = make_benchmark_psf("7x7", factor=1)
    assert psf_small.shape == (7, 7)
    assert np.isclose(psf_small.sum(), 1.0, atol=1e-5)

    psf_large = make_benchmark_psf("51x51", factor=1)
    assert psf_large.shape == (51, 51)
    assert np.isclose(psf_large.sum(), 1.0, atol=1e-5)

    psf_small_dec = make_benchmark_psf("7x7", factor=2)
    assert psf_small_dec.shape == (3, 3)
    assert np.isclose(psf_small_dec.sum(), 1.0, atol=1e-5)

    psf_large_dec = make_benchmark_psf("51x51", factor=2)
    assert psf_large_dec.shape == (25, 25)
    assert np.isclose(psf_large_dec.sum(), 1.0, atol=1e-5)


def test_paired_degradation_determinism_and_gaussian_bsnr() -> None:
    plate = generate_plates(1, seed=42)[0][0]
    psf = make_benchmark_psf("7x7", factor=1)

    obs1, eff_psf1, meta1 = generate_paired_degradation(plate, psf, "gaussian", 30.0, seed=101)
    obs2, eff_psf2, meta2 = generate_paired_degradation(plate, psf, "gaussian", 30.0, seed=101)

    # Identical seed produces bitwise identical observation
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(eff_psf1, eff_psf2)

    # Measured BSNR is close to target (within 1.5 dB)
    assert abs(meta1["measured_bsnr"] - 30.0) <= 1.5


def test_poisson_degradation_headroom_no_clipping() -> None:
    plate = generate_plates(1, seed=7)[0][0]
    psf = make_benchmark_psf("7x7", factor=1)

    # Under nominal 1000 photons with headroom s=0.70, observations stay in [0, 1]
    obs, eff_psf, meta = generate_paired_degradation(
        plate, psf, "poisson", 1000, seed=201, s_scale=0.70
    )
    assert 0.0 <= obs.min() <= obs.max() <= 1.0
    assert meta["effective_photons"] == 700


# --------------------------------------------------------------------------- #
# 3. Restoration and Readability Proxy Tests
# --------------------------------------------------------------------------- #
def test_restoration_methods_execute_and_produce_valid_outputs() -> None:
    plate = generate_plates(1, seed=5)[0][0]
    psf = make_benchmark_psf("7x7", factor=1)
    obs, eff_psf, _ = generate_paired_degradation(plate, psf, "gaussian", 30.0, seed=105)

    for method in ("inverse", "wiener", "cls", "richardson_lucy"):
        raw, clip, ms = run_single_restoration(method, obs, eff_psf, "gaussian")
        assert raw.shape == plate.shape
        assert clip.shape == plate.shape
        assert 0.0 <= clip.min() <= clip.max() <= 1.0
        assert ms > 0.0


def test_readability_proxy_on_clean_plate() -> None:
    plate, text = generate_plates(1, seed=9)[0]
    res = evaluate_readability_proxy(plate, text)
    expected_alnum = sum(c.isalnum() for c in text)
    assert res["expected_chars"] == expected_alnum
    # Clean synthetic plates segment cleanly into exact character count
    assert res["segmented_crops"] == expected_alnum
    assert res["segmentation_status"] == "exact_match"


def test_classify_failure_detects_severe_noise_and_nan() -> None:
    ref = np.ones((96, 336), dtype=np.float32) * 0.5
    nan_arr = ref.copy()
    nan_arr[10, 10] = np.nan

    fails_nan = classify_failure(nan_arr, nan_arr, ref, degraded_rmse=0.05)
    assert "nan_or_infinity" in fails_nan

    # Raw array with extreme overshoot outside [0, 1]
    exploded = ref.copy() + 2.0
    clipped = np.clip(exploded, 0.0, 1.0)
    fails_clip = classify_failure(exploded, clipped, ref, degraded_rmse=0.01)
    assert any("excessive_clipping" in f for f in fails_clip)
    assert "severe_noise_amplification" in fails_clip


# --------------------------------------------------------------------------- #
# 4. Statistical Utility Tests
# --------------------------------------------------------------------------- #
def test_paired_statistics() -> None:
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    b = [0.8, 1.9, 2.7, 3.8, 4.9]
    stats = compute_paired_statistics(a, b)
    assert stats["mean_diff"] > 0
    assert 0.0 <= stats["p_val"] <= 1.0

    agg = aggregate_metric_records(a)
    assert agg["mean"] == 3.0
    assert agg["min"] == 1.0
    assert agg["max"] == 5.0


def test_paired_ttest_detailed_direction() -> None:
    vals_a = [10.0, 11.5, 12.0, 9.5, 10.5]
    vals_b = [8.0, 9.0, 10.0, 8.0, 9.0]
    res_ab = compute_paired_ttest_detailed(vals_a, vals_b, name_a="method_a", name_b="method_b")

    assert res_ab["comparison"] == "method_a - method_b"
    assert res_ab["mean_diff"] > 0.0
    assert res_ab["t_stat"] > 0.0
    assert res_ab["df"] == 4
    assert 0.0 <= res_ab["p_val"] <= 1.0

    # Opposite direction: B - A
    res_ba = compute_paired_ttest_detailed(vals_b, vals_a, name_a="method_b", name_b="method_a")
    assert res_ba["comparison"] == "method_b - method_a"
    assert np.isclose(res_ba["mean_diff"], -res_ab["mean_diff"])
    assert np.isclose(res_ba["t_stat"], -res_ab["t_stat"])
    assert np.isclose(res_ba["p_val"], res_ab["p_val"])


def test_paired_ttest_detailed_sample_count_and_ci() -> None:
    rng = np.random.default_rng(42)
    a = rng.normal(30.0, 2.0, 20).tolist()
    b = (np.array(a) - 3.5 + rng.normal(0, 0.5, 20)).tolist()

    res = compute_paired_ttest_detailed(a, b, name_a="rl", name_b="wiener")
    assert res["sample_count"] == 20
    assert res["df"] == 19
    assert res["ci_95"] is not None
    ci_low, ci_high = res["ci_95"]
    assert ci_low < res["mean_diff"] < ci_high
    assert res["p_val"] < 1e-10


def test_paired_ttest_detailed_zero_variance() -> None:
    # Identical arrays -> zero variance, zero mean diff
    a = [5.0, 5.0, 5.0, 5.0]
    res_zero = compute_paired_ttest_detailed(a, a, name_a="a", name_b="a")
    assert res_zero["std_diff"] == 0.0
    assert res_zero["mean_diff"] == 0.0
    assert res_zero["t_stat"] == 0.0
    assert res_zero["p_val"] == 1.0
    assert res_zero["ci_95"] == [0.0, 0.0]

    # Constant shift -> zero variance, non-zero mean diff
    b = [3.0, 3.0, 3.0, 3.0]
    res_shift = compute_paired_ttest_detailed(a, b, name_a="a", name_b="b")
    assert res_shift["std_diff"] == 0.0
    assert res_shift["mean_diff"] == 2.0
    assert res_shift["t_stat"] == float("inf")
    assert res_shift["p_val"] == 0.0
    assert res_shift["ci_95"] == [2.0, 2.0]


def test_holm_bonferroni_correction() -> None:
    # Empty list
    assert holm_bonferroni_correction([]) == []

    # Known family of p-values
    p_raw = [0.01, 0.04, 0.03]
    # Sorted: p(1)=0.01 (m=3 -> 0.03)
    #         p(2)=0.03 (m=2 -> 0.06)
    #         p(3)=0.04 (m=1 -> 0.04 -> monotonic max becomes 0.06)
    p_adj = holm_bonferroni_correction(p_raw)
    assert np.isclose(p_adj[0], 0.03)
    assert np.isclose(p_adj[1], 0.06)
    assert np.isclose(p_adj[2], 0.06)

    # Monotonicity check
    p_large = [0.6, 0.9, 0.7]
    adj_large = holm_bonferroni_correction(p_large)
    for p in adj_large:
        assert 0.0 <= p <= 1.0


def test_compute_pairwise_algorithm_comparisons_and_serialization() -> None:
    rng = np.random.default_rng(123)
    metric_series = {
        "inverse_deployable": {"psnr": rng.normal(15.0, 1.0, 20).tolist()},
        "wiener_deployable": {"psnr": rng.normal(25.0, 1.0, 20).tolist()},
        "cls_deployable": {"psnr": rng.normal(27.0, 1.0, 20).tolist()},
        "richardson_lucy_deployable": {"psnr": rng.normal(26.5, 1.0, 20).tolist()},
    }
    methods = ["inverse", "wiener", "cls", "richardson_lucy"]

    comps = compute_pairwise_algorithm_comparisons(metric_series, methods)

    # Should contain all 6 comparisons
    expected_pairs = [
        "inverse_vs_wiener",
        "inverse_vs_cls",
        "inverse_vs_richardson_lucy",
        "wiener_vs_cls",
        "wiener_vs_richardson_lucy",
        "cls_vs_richardson_lucy",
    ]
    for pair in expected_pairs:
        assert pair in comps
        assert "p_val_adjusted_holm" in comps[pair]
        assert "significant_holm_05" in comps[pair]

    # Check strict JSON serializability (must not raise TypeError for float32 or int64)
    serialized = json.dumps(comps)
    assert isinstance(serialized, str)
    deserialized = json.loads(serialized)
    assert "cls_vs_richardson_lucy" in deserialized


def test_compute_pairwise_algorithm_comparisons_missing_method() -> None:
    # Simulating Poisson condition where inverse filter is omitted
    rng = np.random.default_rng(456)
    metric_series = {
        "wiener_deployable": {"psnr": rng.normal(20.0, 1.0, 20).tolist()},
        "cls_deployable": {"psnr": rng.normal(28.0, 1.0, 20).tolist()},
        "richardson_lucy_deployable": {"psnr": rng.normal(30.0, 1.0, 20).tolist()},
    }
    methods = ["wiener", "cls", "richardson_lucy"]

    comps = compute_pairwise_algorithm_comparisons(metric_series, methods)

    # Should only contain the 3 valid pairs among available methods
    assert "inverse_vs_wiener" not in comps
    assert "inverse_vs_cls" not in comps
    assert "wiener_vs_cls" in comps
    assert "wiener_vs_richardson_lucy" in comps
    assert "cls_vs_richardson_lucy" in comps
    assert len(comps) == 3


def test_summary_json_contains_both_paired_tests() -> None:
    from pathlib import Path
    summary_path = Path("experiments/results/stage8/summary.json")
    if not summary_path.exists():
        pytest.skip("summary.json not present on disk")

    with open(summary_path, encoding="utf-8") as f:
        summary_data = json.load(f)

    # Check that both paired tests sections exist across cells in core_benchmark
    core_benchmark = summary_data.get("core_benchmark", {})
    assert len(core_benchmark) == 16, f"Expected 16 core cells, got {len(core_benchmark)}"

    for cell_key, cell_data in core_benchmark.items():
        assert "paired_tests_vs_degraded" in cell_data, f"{cell_key} missing paired_tests_vs_degraded"
        assert "paired_tests_algorithm_comparisons" in cell_data, f"{cell_key} missing paired_tests_algorithm_comparisons"

