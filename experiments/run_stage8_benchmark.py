"""Stage 8 Comprehensive Restoration Benchmark Runner.

Executes:
1. Core Benchmark Matrix (Cells C01 to C16) covering 4 restoration algorithms,
   2 blur sizes, 4 noise regimes (Clean, 30 dB Gaussian, 20 dB Gaussian, Poisson),
   and 2 resolutions (1x and factor-2 decimated).
2. Paired experimental design with deterministic seeded noise draws.
3. Strict segregation between Deployable (blind/fixed) and Oracle (swept) configurations.
4. Comprehensive metric computation (RMSE, PSNR, SSIM on unpadded interior).
5. Readability proxy evaluation via Otsu binarization and character segmentation.
6. Automated failure mode classification.
7. Extension E1 (CCPD sharp base plates) and Extension E3 (CCPD real blurred gallery).
8. Structured CSV, JSON export, and publication-quality figure generation.

Usage:
    python experiments/run_stage8_benchmark.py [--samples 25] [--quick] [--skip-figures]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

# Use Agg backend for matplotlib headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.stage8_utils import (
    aggregate_metric_records,
    classify_failure,
    collect_hardware_metadata,
    compute_all_metrics,
    compute_interior_psnr,
    compute_interior_rmse,
    compute_interior_ssim,
    compute_paired_statistics,
    compute_pairwise_algorithm_comparisons,
    crop_interior,
    evaluate_readability_proxy,
    generate_paired_degradation,
    make_benchmark_psf,
    run_single_restoration,
)
from license_plate_restoration.degradation import downsample, effective_psf
from license_plate_restoration.renderer import generate_plates
from license_plate_restoration.restoration import (
    clip_unit_interval,
    wiener_k_from_bsnr,
)


# --------------------------------------------------------------------------- #
# Cell Definitions (Approved Stage 8.1 Core Matrix)
# --------------------------------------------------------------------------- #
CORE_CELLS: dict[str, dict[str, Any]] = {
    "C01": {
        "description": "Full-res, 7x7 blur, Clean (Noiseless)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "clean",
        "noise_param": None,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C02": {
        "description": "Full-res, 7x7 blur, Gaussian 30 dB BSNR",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "gaussian",
        "noise_param": 30.0,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C03": {
        "description": "Full-res, 7x7 blur, Gaussian 20 dB BSNR",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "gaussian",
        "noise_param": 20.0,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C04": {
        "description": "Full-res, 7x7 blur, Poisson N=1000 (N_eff=700)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "poisson",
        "noise_param": 1000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],  # Inverse excluded
    },
    "C05": {
        "description": "Full-res, 7x7 blur, Poisson N=5000 (N_eff=3500)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "poisson",
        "noise_param": 5000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],
    },
    "C06": {
        "description": "Full-res, 7x7 blur, Poisson N=20000 (N_eff=14000)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "7x7",
        "noise_type": "poisson",
        "noise_param": 20000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],
    },
    "C07": {
        "description": "Full-res, 51x51 blur, Clean (Noiseless)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "clean",
        "noise_param": None,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C08": {
        "description": "Full-res, 51x51 blur, Gaussian 30 dB BSNR",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "gaussian",
        "noise_param": 30.0,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C09": {
        "description": "Full-res, 51x51 blur, Gaussian 20 dB BSNR",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "gaussian",
        "noise_param": 20.0,
        "margin": 12,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C10": {
        "description": "Full-res, 51x51 blur, Poisson N=1000 (N_eff=700)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "poisson",
        "noise_param": 1000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],
    },
    "C11": {
        "description": "Full-res, 51x51 blur, Poisson N=5000 (N_eff=3500)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "poisson",
        "noise_param": 5000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],
    },
    "C12": {
        "description": "Full-res, 51x51 blur, Poisson N=20000 (N_eff=14000)",
        "resolution": "full",
        "factor": 1,
        "kernel_name": "51x51",
        "noise_type": "poisson",
        "noise_param": 20000,
        "margin": 12,
        "methods": ["wiener", "cls", "richardson_lucy"],
    },
    "C13": {
        "description": "Decimated (F2), Small blur, Gaussian 30 dB BSNR",
        "resolution": "decimated",
        "factor": 2,
        "kernel_name": "7x7",
        "noise_type": "gaussian",
        "noise_param": 30.0,
        "margin": 6,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C14": {
        "description": "Decimated (F2), Small blur, Gaussian 20 dB BSNR",
        "resolution": "decimated",
        "factor": 2,
        "kernel_name": "7x7",
        "noise_type": "gaussian",
        "noise_param": 20.0,
        "margin": 6,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C15": {
        "description": "Decimated (F2), Large blur, Gaussian 30 dB BSNR",
        "resolution": "decimated",
        "factor": 2,
        "kernel_name": "51x51",
        "noise_type": "gaussian",
        "noise_param": 30.0,
        "margin": 6,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
    "C16": {
        "description": "Decimated (F2), Large blur, Gaussian 20 dB BSNR",
        "resolution": "decimated",
        "factor": 2,
        "kernel_name": "51x51",
        "noise_type": "gaussian",
        "noise_param": 20.0,
        "margin": 6,
        "methods": ["inverse", "wiener", "cls", "richardson_lucy"],
    },
}


# --------------------------------------------------------------------------- #
# Parameter Resolution Helpers
# --------------------------------------------------------------------------- #
def get_deployable_param(method: str, noise_type: str,
                         noise_param: float | int | None) -> float | int:
    """Return fixed/blind hyperparameter chosen without access to clean image."""
    m = method.lower()
    if m == "inverse":
        return 1e-12
    elif m == "wiener":
        if noise_type == "clean":
            return 1e-5
        elif noise_type == "gaussian":
            return wiener_k_from_bsnr(float(noise_param))
        else:  # poisson
            return 1e-3
    elif m == "cls":
        if noise_type == "clean":
            return 1e-5
        elif noise_type == "gaussian":
            bsnr = float(noise_param)
            return 1e-3 if bsnr >= 25.0 else 1e-2
        else:  # poisson
            return 1e-3
    elif m in ("richardson_lucy", "rl"):
        if noise_type == "clean":
            return 50
        elif noise_type == "gaussian":
            bsnr = float(noise_param)
            return 30 if bsnr >= 25.0 else 20
        else:  # poisson
            return 20
    raise ValueError(f"Unknown method {method}")


ORACLE_SEARCH_GRIDS: dict[str, list[float | int]] = {
    "wiener": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "cls": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    "richardson_lucy": [1, 5, 10, 15, 20, 30, 50, 100],
}


# --------------------------------------------------------------------------- #
# Benchmark Execution Engine
# --------------------------------------------------------------------------- #
class Stage8BenchmarkRunner:
    """Orchestrates benchmark matrix execution, statistics, and reporting."""

    def __init__(self, out_dir: Path, num_samples: int = 25, seed_offset: int = 0):
        self.out_dir = out_dir
        self.num_samples = num_samples
        self.seed_offset = seed_offset

        # Subdirectories
        self.dir_meta = out_dir / "metadata"
        self.dir_raw = out_dir / "raw"
        self.dir_agg = out_dir / "aggregated"
        self.dir_fig = out_dir / "figures"
        self.dir_qual = out_dir / "qualitative"

        for d in (self.dir_meta, self.dir_raw, self.dir_agg, self.dir_fig, self.dir_qual):
            d.mkdir(parents=True, exist_ok=True)

        # Load clean dataset
        data_path = ROOT_DIR / "license_plate_restoration" / "data" / "plates.npz"
        if not data_path.is_file():
            raise FileNotFoundError(f"Missing plate dataset at {data_path}")

        with np.load(data_path) as data:
            self.synth_imgs = data["synthetic_imgs"]
            self.synth_text = data["synthetic_text"]
            self.base_imgs = data["base_imgs"]
            self.base_text = data["base_text"]
            self.blur_imgs = data["blur_imgs"]
            self.blur_text = data["blur_text"]

        print(f"[Init] Loaded {len(self.synth_imgs)} synthetic, "
              f"{len(self.base_imgs)} CCPD base, {len(self.blur_imgs)} CCPD blur plates.")
        print(f"[Init] Benchmark configured for N={self.num_samples} samples per cell.")

    def run_cell(self, cell_id: str) -> dict[str, Any]:
        """Execute all trials for a given cell ID."""
        cfg = CORE_CELLS[cell_id]
        print(f"\n>>> Running Cell {cell_id}: {cfg['description']} (N={self.num_samples})")

        factor = cfg["factor"]
        psf = make_benchmark_psf(cfg["kernel_name"], factor=1)
        margin = cfg["margin"]
        noise_type = cfg["noise_type"]
        noise_param = cfg["noise_param"]
        methods = cfg["methods"]

        cell_raw_records: list[dict[str, Any]] = []

        # Accumulators for aggregate statistics
        metric_series: dict[str, dict[str, list[float]]] = {
            "degraded": {"rmse": [], "psnr": [], "ssim": []}
        }
        for m in methods:
            metric_series[f"{m}_deployable"] = {"rmse": [], "psnr": [], "ssim": [], "runtime_ms": []}
            if m in ORACLE_SEARCH_GRIDS:
                metric_series[f"{m}_oracle"] = {"rmse": [], "psnr": [], "ssim": [], "runtime_ms": []}

        # Progress tracking
        for idx in range(self.num_samples):
            plate_idx = (self.seed_offset + idx) % len(self.synth_imgs)
            clean_full = self.synth_imgs[plate_idx]
            text = str(self.synth_text[plate_idx])
            clean_ref = downsample(clean_full, factor) if factor > 1 else clean_full

            # Generate paired degradation
            deg_seed = 1000 + idx
            obs, eff_psf, dmeta = generate_paired_degradation(
                clean_full, psf, noise_type, noise_param, factor=factor, seed=deg_seed
            )

            # Measure degraded baseline
            deg_rescaled = obs / dmeta["s_scale"]
            deg_clipped = clip_unit_interval(deg_rescaled)
            deg_metrics = compute_all_metrics(deg_clipped, clean_ref, margin)
            deg_readability = evaluate_readability_proxy(deg_clipped, text)

            metric_series["degraded"]["rmse"].append(deg_metrics["rmse"])
            metric_series["degraded"]["psnr"].append(deg_metrics["psnr"])
            metric_series["degraded"]["ssim"].append(deg_metrics["ssim"])

            trial_record: dict[str, Any] = {
                "cell_id": cell_id,
                "trial_idx": idx,
                "plate_idx": plate_idx,
                "text": text,
                "degraded": {
                    "metrics": deg_metrics,
                    "readability": deg_readability,
                    "measured_bsnr": dmeta.get("measured_bsnr"),
                },
                "restorations": {},
            }

            # Evaluate each algorithm
            for m in methods:
                # 1. Deployable Configuration
                dep_param = get_deployable_param(m, noise_type, noise_param)
                raw_dep, clip_dep, ms_dep = run_single_restoration(
                    m, obs, eff_psf, noise_type, param_val=dep_param, s_scale=dmeta["s_scale"]
                )
                m_dep = compute_all_metrics(clip_dep, clean_ref, margin)
                read_dep = evaluate_readability_proxy(clip_dep, text)
                fails_dep = classify_failure(
                    raw_dep, clip_dep, clean_ref, deg_metrics["rmse"], margin,
                    read_dep["segmented_crops"], read_dep["expected_chars"]
                )

                metric_series[f"{m}_deployable"]["rmse"].append(m_dep["rmse"])
                metric_series[f"{m}_deployable"]["psnr"].append(m_dep["psnr"])
                metric_series[f"{m}_deployable"]["ssim"].append(m_dep["ssim"])
                metric_series[f"{m}_deployable"]["runtime_ms"].append(ms_dep)

                trial_record["restorations"][f"{m}_deployable"] = {
                    "param": dep_param,
                    "metrics": m_dep,
                    "runtime_ms": ms_dep,
                    "readability": read_dep,
                    "failures": fails_dep,
                }

                # 2. Oracle Configuration (Swept grid if applicable)
                if m in ORACLE_SEARCH_GRIDS:
                    best_rmse = float("inf")
                    best_param = None
                    best_clip = None
                    best_raw = None
                    best_ms = 0.0

                    grid = ORACLE_SEARCH_GRIDS[m]
                    for candidate in grid:
                        r_cand, c_cand, ms_cand = run_single_restoration(
                            m, obs, eff_psf, noise_type, param_val=candidate, s_scale=dmeta["s_scale"]
                        )
                        cand_rmse = compute_interior_rmse(c_cand, clean_ref, margin)
                        if cand_rmse < best_rmse:
                            best_rmse = cand_rmse
                            best_param = candidate
                            best_clip = c_cand
                            best_raw = r_cand
                            best_ms = ms_cand

                    m_orc = compute_all_metrics(best_clip, clean_ref, margin)
                    read_orc = evaluate_readability_proxy(best_clip, text)
                    fails_orc = classify_failure(
                        best_raw, best_clip, clean_ref, deg_metrics["rmse"], margin,
                        read_orc["segmented_crops"], read_orc["expected_chars"]
                    )

                    metric_series[f"{m}_oracle"]["rmse"].append(m_orc["rmse"])
                    metric_series[f"{m}_oracle"]["psnr"].append(m_orc["psnr"])
                    metric_series[f"{m}_oracle"]["ssim"].append(m_orc["ssim"])
                    metric_series[f"{m}_oracle"]["runtime_ms"].append(best_ms)

                    trial_record["restorations"][f"{m}_oracle"] = {
                        "best_param": best_param,
                        "metrics": m_orc,
                        "runtime_ms": best_ms,
                        "readability": read_orc,
                        "failures": fails_orc,
                    }

            cell_raw_records.append(trial_record)

        # Aggregate summary statistics
        summary: dict[str, Any] = {
            "cell_id": cell_id,
            "description": cfg["description"],
            "num_samples": self.num_samples,
            "metrics": {},
            "paired_tests_vs_degraded": {},
            "paired_tests_algorithm_comparisons": {},
        }

        for k, v in metric_series.items():
            summary["metrics"][k] = {
                "rmse": aggregate_metric_records(v["rmse"]),
                "psnr": aggregate_metric_records(v["psnr"]),
                "ssim": aggregate_metric_records(v["ssim"]),
            }
            if "runtime_ms" in v:
                summary["metrics"][k]["runtime_ms"] = aggregate_metric_records(v["runtime_ms"])

            # Paired t-test against degraded PSNR
            if k != "degraded":
                summary["paired_tests_vs_degraded"][k] = compute_paired_statistics(
                    v["psnr"], metric_series["degraded"]["psnr"]
                )

        # Algorithm-to-algorithm paired comparisons
        summary["paired_tests_algorithm_comparisons"] = compute_pairwise_algorithm_comparisons(
            metric_series, methods
        )

        # Save cell raw JSON
        cell_raw_path = self.dir_raw / f"raw_{cell_id}.json"
        with open(cell_raw_path, "w", encoding="utf-8") as f:
            json.dump(cell_raw_records, f, indent=2)

        # Console Summary
        deg_p = summary["metrics"]["degraded"]["psnr"]["mean"]
        print(f"  Summary {cell_id} | Degraded: {deg_p:.2f} dB")
        for m in methods:
            dep_p = summary["metrics"][f"{m}_deployable"]["psnr"]["mean"]
            dep_s = summary["metrics"][f"{m}_deployable"]["ssim"]["mean"]
            orc_str = ""
            if f"{m}_oracle" in summary["metrics"]:
                orc_p = summary["metrics"][f"{m}_oracle"]["psnr"]["mean"]
                orc_str = f" | Orc: {orc_p:.2f} dB"
            print(f"    {m:<16s} Dep: {dep_p:.2f} dB (SSIM={dep_s:.4f}){orc_str}")

        return summary

    def run_all_core_cells(self) -> dict[str, Any]:
        """Run complete C01 to C16 core benchmark matrix."""
        print("=" * 75)
        print("  EXECUTING COMPLETE STAGE 8 RESTORATION BENCHMARK (CELLS C01 - C16)")
        print("=" * 75)

        full_results = {}
        for cell_id in sorted(CORE_CELLS):
            summary = self.run_cell(cell_id)
            full_results[cell_id] = summary

        # Save aggregated JSON
        agg_path = self.dir_agg / "aggregated_core_matrix.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        self.export_csv_tables(full_results)
        return full_results

    def reaggregate_from_existing_raw(self) -> dict[str, Any]:
        """Re-aggregate summary statistics and algorithm comparisons from existing raw JSONs.

        Preserves all 320 existing raw trial records and deconvolution outputs
        without running new restoration experiments.
        """
        print("=" * 75)
        print("  RE-AGGREGATING STAGE 8 BENCHMARK RESULTS FROM EXISTING RAW RECORDS")
        print("=" * 75)

        full_results = {}
        for cell_id in sorted(CORE_CELLS):
            cfg = CORE_CELLS[cell_id]
            cell_raw_path = self.dir_raw / f"raw_{cell_id}.json"
            if not cell_raw_path.is_file():
                raise FileNotFoundError(f"Missing raw records for cell {cell_id} at {cell_raw_path}")

            with open(cell_raw_path, "r", encoding="utf-8") as f:
                cell_raw_records = json.load(f)

            methods = cfg["methods"]
            metric_series: dict[str, dict[str, list[float]]] = {
                "degraded": {"rmse": [], "psnr": [], "ssim": []}
            }
            for m in methods:
                metric_series[f"{m}_deployable"] = {"rmse": [], "psnr": [], "ssim": [], "runtime_ms": []}
                if m in ORACLE_SEARCH_GRIDS:
                    metric_series[f"{m}_oracle"] = {"rmse": [], "psnr": [], "ssim": [], "runtime_ms": []}

            for trial in cell_raw_records:
                deg_m = trial["degraded"]["metrics"]
                metric_series["degraded"]["rmse"].append(deg_m["rmse"])
                metric_series["degraded"]["psnr"].append(deg_m["psnr"])
                metric_series["degraded"]["ssim"].append(deg_m["ssim"])

                for mkey, mval in trial["restorations"].items():
                    if mkey in metric_series:
                        m_data = mval["metrics"]
                        metric_series[mkey]["rmse"].append(m_data["rmse"])
                        metric_series[mkey]["psnr"].append(m_data["psnr"])
                        metric_series[mkey]["ssim"].append(m_data["ssim"])
                        metric_series[mkey]["runtime_ms"].append(mval.get("runtime_ms", 0.0))

            summary: dict[str, Any] = {
                "cell_id": cell_id,
                "description": cfg["description"],
                "num_samples": len(cell_raw_records),
                "metrics": {},
                "paired_tests_vs_degraded": {},
                "paired_tests_algorithm_comparisons": {},
            }

            for k, v in metric_series.items():
                summary["metrics"][k] = {
                    "rmse": aggregate_metric_records(v["rmse"]),
                    "psnr": aggregate_metric_records(v["psnr"]),
                    "ssim": aggregate_metric_records(v["ssim"]),
                }
                if "runtime_ms" in v:
                    summary["metrics"][k]["runtime_ms"] = aggregate_metric_records(v["runtime_ms"])

                if k != "degraded":
                    summary["paired_tests_vs_degraded"][k] = compute_paired_statistics(
                        v["psnr"], metric_series["degraded"]["psnr"]
                    )

            summary["paired_tests_algorithm_comparisons"] = compute_pairwise_algorithm_comparisons(
                metric_series, methods
            )
            full_results[cell_id] = summary

        # Save aggregated JSON
        agg_path = self.dir_agg / "aggregated_core_matrix.json"
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)

        self.export_csv_tables(full_results)
        print(f"\n[Re-aggregate] Successfully updated {agg_path}")
        return full_results

    def export_csv_tables(self, full_results: dict[str, Any]) -> None:
        """Export Deployable and Oracle results to clean, publication-ready CSVs."""
        # 1. Deployable Results CSV
        dep_csv_path = self.out_dir / "deployable_results.csv"
        with open(dep_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cell_id", "description", "resolution", "kernel", "noise",
                "degraded_psnr_mean", "degraded_ssim_mean",
                "inv_psnr_mean", "inv_ssim_mean", "inv_runtime_ms",
                "wiener_psnr_mean", "wiener_ssim_mean", "wiener_runtime_ms",
                "cls_psnr_mean", "cls_ssim_mean", "cls_runtime_ms",
                "rl_psnr_mean", "rl_ssim_mean", "rl_runtime_ms",
            ])
            for cid, cdata in full_results.items():
                cfg = CORE_CELLS[cid]
                m = cdata["metrics"]

                def get_m(k: str, field: str) -> str:
                    if k in m and field in m[k]:
                        return f"{m[k][field]['mean']:.4f}"
                    return "N/A"

                writer.writerow([
                    cid, cfg["description"], cfg["resolution"], cfg["kernel_name"], cfg["noise_type"],
                    get_m("degraded", "psnr"), get_m("degraded", "ssim"),
                    get_m("inverse_deployable", "psnr"), get_m("inverse_deployable", "ssim"), get_m("inverse_deployable", "runtime_ms"),
                    get_m("wiener_deployable", "psnr"), get_m("wiener_deployable", "ssim"), get_m("wiener_deployable", "runtime_ms"),
                    get_m("cls_deployable", "psnr"), get_m("cls_deployable", "ssim"), get_m("cls_deployable", "runtime_ms"),
                    get_m("richardson_lucy_deployable", "psnr"), get_m("richardson_lucy_deployable", "ssim"), get_m("richardson_lucy_deployable", "runtime_ms"),
                ])

        # 2. Oracle Results CSV
        orc_csv_path = self.out_dir / "oracle_results.csv"
        with open(orc_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cell_id", "description",
                "wiener_oracle_psnr_mean", "wiener_oracle_ssim_mean",
                "cls_oracle_psnr_mean", "cls_oracle_ssim_mean",
                "rl_oracle_psnr_mean", "rl_oracle_ssim_mean",
            ])
            for cid, cdata in full_results.items():
                cfg = CORE_CELLS[cid]
                m = cdata["metrics"]

                def get_orc(k: str, field: str) -> str:
                    if f"{k}_oracle" in m and field in m[f"{k}_oracle"]:
                        return f"{m[f'{k}_oracle'][field]['mean']:.4f}"
                    return "N/A"

                writer.writerow([
                    cid, cfg["description"],
                    get_orc("wiener", "psnr"), get_orc("wiener", "ssim"),
                    get_orc("cls", "psnr"), get_orc("cls", "ssim"),
                    get_orc("richardson_lucy", "psnr"), get_orc("richardson_lucy", "ssim"),
                ])

        print(f"\n[Export] Saved {dep_csv_path}")
        print(f"[Export] Saved {orc_csv_path}")

    def run_extensions(self) -> dict[str, Any]:
        """Execute Extension E1 (CCPD base plates) and E3 (CCPD real blur gallery)."""
        print("\n" + "=" * 75)
        print("  EXECUTING OPTIONAL EXTENSIONS (E1: CCPD BASE, E3: REAL BLUR GALLERY)")
        print("=" * 75)

        # Extension E1: CCPD Base Plates (10 plates, 92x256)
        psf = make_benchmark_psf("7x7")
        e1_records = []
        for idx in range(min(10, len(self.base_imgs))):
            clean = self.base_imgs[idx]
            text = str(self.base_text[idx])
            obs, eff_psf, dmeta = generate_paired_degradation(clean, psf, "gaussian", 30.0, seed=500 + idx)
            deg_m = compute_all_metrics(obs, clean, margin=12)

            _, w_clip, _ = run_single_restoration("wiener", obs, eff_psf, "gaussian", param_val=1e-3)
            _, c_clip, _ = run_single_restoration("cls", obs, eff_psf, "gaussian", param_val=1e-3)
            _, rl_clip, _ = run_single_restoration("richardson_lucy", obs, eff_psf, "gaussian", param_val=30)

            e1_records.append({
                "idx": idx,
                "text": text,
                "degraded": deg_m,
                "wiener": compute_all_metrics(w_clip, clean, margin=12),
                "cls": compute_all_metrics(c_clip, clean, margin=12),
                "rl": compute_all_metrics(rl_clip, clean, margin=12),
            })

        e1_summary = {
            "degraded_psnr": float(np.mean([r["degraded"]["psnr"] for r in e1_records])),
            "wiener_psnr": float(np.mean([r["wiener"]["psnr"] for r in e1_records])),
            "cls_psnr": float(np.mean([r["cls"]["psnr"] for r in e1_records])),
            "rl_psnr": float(np.mean([r["rl"]["psnr"] for r in e1_records])),
        }
        print(f"  [E1: CCPD Base 30 dB] Degraded: {e1_summary['degraded_psnr']:.2f} dB | "
              f"Wiener: {e1_summary['wiener_psnr']:.2f} dB | "
              f"CLS: {e1_summary['cls_psnr']:.2f} dB | RL: {e1_summary['rl_psnr']:.2f} dB")

        # Extension E3: Naturally Blurred CCPD Plates (Qualitative Gallery & Binarization)
        e3_records = []
        psf_cand = make_benchmark_psf("7x7")  # candidate PSF
        for idx in range(min(10, len(self.blur_imgs))):
            raw_blur = self.blur_imgs[idx]
            text = str(self.blur_text[idx])

            deg_read = evaluate_readability_proxy(raw_blur, text)
            _, w_clip, _ = run_single_restoration("wiener", raw_blur, psf_cand, "gaussian", param_val=1e-3)
            _, c_clip, _ = run_single_restoration("cls", raw_blur, psf_cand, "gaussian", param_val=1e-3)
            _, rl_clip, _ = run_single_restoration("richardson_lucy", raw_blur, psf_cand, "gaussian", param_val=20)

            e3_records.append({
                "idx": idx,
                "text": text,
                "degraded_readability": deg_read,
                "wiener_readability": evaluate_readability_proxy(w_clip, text),
                "cls_readability": evaluate_readability_proxy(c_clip, text),
                "rl_readability": evaluate_readability_proxy(rl_clip, text),
            })

        e3_yields = {
            "degraded_crop_mean": float(np.mean([r["degraded_readability"]["segmented_crops"] for r in e3_records])),
            "wiener_crop_mean": float(np.mean([r["wiener_readability"]["segmented_crops"] for r in e3_records])),
            "cls_crop_mean": float(np.mean([r["cls_readability"]["segmented_crops"] for r in e3_records])),
            "rl_crop_mean": float(np.mean([r["rl_readability"]["segmented_crops"] for r in e3_records])),
        }
        print(f"  [E3: Real Blur Crops] Degraded: {e3_yields['degraded_crop_mean']:.1f} crops | "
              f"Wiener: {e3_yields['wiener_crop_mean']:.1f} crops | "
              f"CLS: {e3_yields['cls_crop_mean']:.1f} crops | RL: {e3_yields['rl_crop_mean']:.1f} crops")

        ext_payload = {
            "extension_e1_ccpd_base": {"summary": e1_summary, "records": e1_records},
            "extension_e3_ccpd_blur": {"yields": e3_yields, "records": e3_records},
        }
        with open(self.dir_agg / "extensions_results.json", "w", encoding="utf-8") as f:
            json.dump(ext_payload, f, indent=2)

        return ext_payload

    def generate_all_figures(self, full_results: dict[str, Any], ext_results: dict[str, Any]) -> None:
        """Generate high-resolution publication-quality comparison figures."""
        print("\n" + "=" * 75)
        print("  GENERATING STAGE 8 BENCHMARK FIGURES")
        print("=" * 75)

        # 1. Figure 1: Panel Comparison - Small Blur 30 dB (Cell C02)
        fig1_path = self.dir_fig / "figure1_panel_comparison_small_30db.png"
        self._plot_panel_comparison(
            cell_id="C02", title="Restoration Comparison: Small Blur (7x7) + Gaussian Noise 30 dB (Cell C02)",
            out_path=fig1_path
        )

        # 2. Figure 2: Panel Comparison - Large Blur 20 dB (Cell C09)
        fig2_path = self.dir_fig / "figure2_panel_comparison_large_20db.png"
        self._plot_panel_comparison(
            cell_id="C09", title="Restoration Comparison: Severe Large Blur (51x51) + Gaussian Noise 20 dB (Cell C09)",
            out_path=fig2_path
        )

        # 3. Figure 3: PSNR Comparison Across All 16 Cells
        fig3_path = self.dir_fig / "figure3_psnr_comparison_core_matrix.png"
        self._plot_metric_matrix_bars(full_results, metric="psnr", out_path=fig3_path)

        # 4. Figure 4: SSIM Comparison Across All 16 Cells
        fig4_path = self.dir_fig / "figure4_ssim_comparison_core_matrix.png"
        self._plot_metric_matrix_bars(full_results, metric="ssim", out_path=fig4_path)

        # 5. Figure 5: Runtime vs Quality Pareto Frontier
        fig5_path = self.dir_fig / "figure5_runtime_vs_quality_pareto.png"
        self._plot_runtime_pareto(full_results, out_path=fig5_path)

        # 6. Figure 6: Richardson-Lucy Semi-Convergence Iteration Trajectories
        fig6_path = self.dir_fig / "figure6_rl_semi_convergence.png"
        self._plot_rl_semi_convergence(out_path=fig6_path)

        # 7. Figure 7: Hyperparameter Sensitivity Curves
        fig7_path = self.dir_fig / "figure7_parameter_sensitivity.png"
        self._plot_parameter_sensitivity(out_path=fig7_path)

        # 8. Figure 8: CCPD Naturally Blurred Real Camera Gallery
        fig8_path = self.dir_fig / "figure8_ccpd_real_blur_gallery.png"
        self._plot_real_blur_gallery(out_path=fig8_path)

        # 9. Figure 9: Characteristic Failure Modes Gallery
        fig9_path = self.dir_fig / "figure9_failure_modes_gallery.png"
        self._plot_failure_modes_gallery(out_path=fig9_path)

        print("[Figures] All 9 publication figures successfully generated.")

    # ----------------------------------------------------------------------- #
    # Plotting Implementation Helpers
    # ----------------------------------------------------------------------- #
    def _plot_panel_comparison(self, cell_id: str, title: str, out_path: Path) -> None:
        cfg = CORE_CELLS[cell_id]
        clean = self.synth_imgs[0]
        text = str(self.synth_text[0])
        factor = cfg["factor"]
        psf = make_benchmark_psf(cfg["kernel_name"], factor=1)
        ref = downsample(clean, factor) if factor > 1 else clean

        obs, eff_psf, dmeta = generate_paired_degradation(
            clean, psf, cfg["noise_type"], cfg["noise_param"], factor=factor, seed=1000
        )
        deg = clip_unit_interval(obs / dmeta["s_scale"])
        d_p = compute_interior_psnr(deg, ref, cfg["margin"])

        # Restorations
        panels = [("Clean Ground Truth", ref, None), (f"Degraded ({d_p:.1f} dB)", deg, None)]
        for m in cfg["methods"]:
            param = get_deployable_param(m, cfg["noise_type"], cfg["noise_param"])
            _, clip, ms = run_single_restoration(m, obs, eff_psf, cfg["noise_type"], param, dmeta["s_scale"])
            p = compute_interior_psnr(clip, ref, cfg["margin"])
            s = compute_interior_ssim(clip, ref, cfg["margin"])
            label = f"{m.capitalize()}\n{p:.1f} dB | SSIM {s:.3f}\n({ms:.1f} ms)"
            panels.append((label, clip, ms))

        fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.2 * len(panels)), dpi=150)
        fig.suptitle(f"{title}\nGround Truth String: '{text}'", fontsize=12, fontweight="bold", y=0.99)

        for ax, (lbl, img, _) in zip(axes, panels):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(lbl, fontsize=10, loc="left", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_metric_matrix_bars(self, full_results: dict[str, Any], metric: str, out_path: Path) -> None:
        cells = list(CORE_CELLS.keys())
        x = np.arange(len(cells))
        width = 0.16

        fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
        methods = [("degraded", "Degraded", "#7f7f7f"),
                   ("inverse_deployable", "Inverse", "#d62728"),
                   ("wiener_deployable", "Wiener", "#1f77b4"),
                   ("cls_deployable", "CLS", "#2ca02c"),
                   ("richardson_lucy_deployable", "RL", "#ff7f0e")]

        for idx, (mkey, mlabel, color) in enumerate(methods):
            vals = []
            for cid in cells:
                mdict = full_results[cid]["metrics"]
                if mkey in mdict:
                    vals.append(mdict[mkey][metric]["mean"])
                else:
                    vals.append(0.0)  # Excluded method
            offset = (idx - 2) * width
            ax.bar(x + offset, vals, width, label=mlabel, color=color, alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels(cells, fontsize=9, fontweight="bold")
        unit = "dB" if metric == "psnr" else "Index"
        ax.set_ylabel(f"Mean Interior {metric.upper()} ({unit})", fontsize=11, fontweight="bold")
        ax.set_title(f"Comprehensive Core Benchmark Matrix: Mean {metric.upper()} Across Cells C01-C16", fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.95)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_runtime_pareto(self, full_results: dict[str, Any], out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        c02 = full_results["C02"]["metrics"]
        c08 = full_results["C08"]["metrics"]

        points = [
            ("Wiener (7x7)", c02["wiener_deployable"]["runtime_ms"]["mean"], c02["wiener_deployable"]["psnr"]["mean"], "#1f77b4", "o"),
            ("CLS (7x7)", c02["cls_deployable"]["runtime_ms"]["mean"], c02["cls_deployable"]["psnr"]["mean"], "#2ca02c", "s"),
            ("RL (7x7, 30 iters)", c02["richardson_lucy_deployable"]["runtime_ms"]["mean"], c02["richardson_lucy_deployable"]["psnr"]["mean"], "#ff7f0e", "^"),
            ("Wiener (51x51)", c08["wiener_deployable"]["runtime_ms"]["mean"], c08["wiener_deployable"]["psnr"]["mean"], "#1f77b4", "o"),
            ("CLS (51x51)", c08["cls_deployable"]["runtime_ms"]["mean"], c08["cls_deployable"]["psnr"]["mean"], "#2ca02c", "s"),
            ("RL (51x51, 30 iters)", c08["richardson_lucy_deployable"]["runtime_ms"]["mean"], c08["richardson_lucy_deployable"]["psnr"]["mean"], "#ff7f0e", "^"),
        ]

        for lbl, rt, psnr, col, marker in points:
            ax.scatter(rt, psnr, color=col, marker=marker, s=120, edgecolors="black", linewidth=1.2, label=lbl)
            ax.annotate(lbl, (rt * 1.08, psnr - 0.2), fontsize=8.5)

        ax.set_xscale("log")
        ax.set_xlabel("Mean Kernel Runtime (milliseconds, log scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean Restored PSNR (dB)", fontsize=11, fontweight="bold")
        ax.set_title("Runtime vs. Quality Pareto Trade-Off (Gaussian 30 dB)", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_rl_semi_convergence(self, out_path: Path) -> None:
        clean = self.synth_imgs[0]
        psf = make_benchmark_psf("7x7")
        it_counts = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

        conditions = [
            ("Clean (Noiseless)", "clean", None, "#2ca02c"),
            ("Gaussian 30 dB", "gaussian", 30.0, "#1f77b4"),
            ("Gaussian 20 dB", "gaussian", 20.0, "#d62728"),
            ("Poisson N=5000", "poisson", 5000, "#9467bd"),
        ]

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        for label, n_type, n_param, col in conditions:
            obs, eff_psf, dmeta = generate_paired_degradation(clean, psf, n_type, n_param, seed=777)
            psnrs = []
            for it in it_counts:
                _, clip, _ = run_single_restoration("richardson_lucy", obs, eff_psf, n_type, param_val=it, s_scale=dmeta["s_scale"])
                psnrs.append(compute_interior_psnr(clip, clean, margin=12))
            ax.plot(it_counts, psnrs, marker="o", linewidth=2.0, color=col, label=label)

        ax.set_xlabel("Richardson-Lucy Iterations (k)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Restored Interior PSNR (dB)", fontsize=11, fontweight="bold")
        ax.set_title("Richardson-Lucy Semi-Convergence & Noise Amplification", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", framealpha=0.95)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_parameter_sensitivity(self, out_path: Path) -> None:
        clean = self.synth_imgs[1]
        psf = make_benchmark_psf("7x7")
        obs, eff_psf, _ = generate_paired_degradation(clean, psf, "gaussian", 30.0, seed=888)

        k_grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
        w_psnrs = []
        c_psnrs = []
        for val in k_grid:
            _, w_c, _ = run_single_restoration("wiener", obs, eff_psf, "gaussian", param_val=val)
            _, c_c, _ = run_single_restoration("cls", obs, eff_psf, "gaussian", param_val=val)
            w_psnrs.append(compute_interior_psnr(w_c, clean, margin=12))
            c_psnrs.append(compute_interior_psnr(c_c, clean, margin=12))

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        ax.plot(k_grid, w_psnrs, marker="o", linewidth=2.0, color="#1f77b4", label="Wiener Filter (varying K)")
        ax.plot(k_grid, c_psnrs, marker="s", linewidth=2.0, color="#2ca02c", label="CLS Filter (varying lambda)")
        ax.axvline(1e-3, color="#7f7f7f", linestyle=":", label="Deployable Baseline (1e-3)")

        ax.set_xscale("log")
        ax.set_xlabel("Regularization Hyperparameter (log scale)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Restored Interior PSNR (dB)", fontsize=11, fontweight="bold")
        ax.set_title("Regularization Parameter Sensitivity Curve (Small Blur 30 dB)", fontsize=13, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="lower center", framealpha=0.95)

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_real_blur_gallery(self, out_path: Path) -> None:
        psf = make_benchmark_psf("7x7")
        fig, axes = plt.subplots(4, 4, figsize=(14, 8), dpi=150)
        fig.suptitle("Naturally Blurred CCPD Plate Gallery (Extension E3: Qualitative Restorations)", fontsize=13, fontweight="bold", y=0.99)

        for idx in range(4):
            raw = self.blur_imgs[idx]
            text = str(self.blur_text[idx])
            _, w_c, _ = run_single_restoration("wiener", raw, psf, "gaussian", param_val=1e-3)
            _, c_c, _ = run_single_restoration("cls", raw, psf, "gaussian", param_val=1e-3)
            _, r_c, _ = run_single_restoration("richardson_lucy", raw, psf, "gaussian", param_val=20)

            row_imgs = [(f"Plate {idx+1}: Raw Camera Blur", raw),
                        ("Wiener (K=1e-3)", w_c),
                        ("CLS (lambda=1e-3)", c_c),
                        ("RL (k=20)", r_c)]

            for col_idx, (lbl, img) in enumerate(row_imgs):
                ax = axes[idx, col_idx]
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
                ax.set_title(lbl, fontsize=9, loc="left", pad=3)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_failure_modes_gallery(self, out_path: Path) -> None:
        clean = self.synth_imgs[0]
        psf = make_benchmark_psf("51x51")
        obs, eff_psf, dmeta = generate_paired_degradation(clean, psf, "gaussian", 20.0, seed=123)

        raw_inv, clip_inv, _ = run_single_restoration("inverse", obs, eff_psf, "gaussian", param_val=1e-12)
        raw_ring, clip_ring, _ = run_single_restoration("wiener", obs, eff_psf, "gaussian", param_val=1e-5)
        raw_smooth, clip_smooth, _ = run_single_restoration("cls", obs, eff_psf, "gaussian", param_val=1e-1)
        raw_rl_over, clip_rl_over, _ = run_single_restoration("richardson_lucy", obs, eff_psf, "gaussian", param_val=100)

        fig, axes = plt.subplots(4, 1, figsize=(10, 8.5), dpi=150)
        fig.suptitle("Characteristic Deconvolution Failure Modes & Degradation Pathologies", fontsize=12, fontweight="bold", y=0.99)

        cases = [
            ("Failure Mode 1: Catastrophic Noise Explosion (Direct Inverse Filter at 20 dB BSNR)\n"
             f"Clipping: {100.0 * np.sum((raw_inv < 0) | (raw_inv > 1)) / raw_inv.size:.1f}% pixels truncated, PSNR={compute_interior_psnr(clip_inv, clean, 12):.1f} dB",
             clip_inv),
            ("Failure Mode 2: Severe High-Frequency Ringing & Boundary Reflection (Under-Regularized Wiener K=1e-5)\n"
             f"Dense Gibb's oscillations distorting character strokes, PSNR={compute_interior_psnr(clip_ring, clean, 12):.1f} dB",
             clip_ring),
            ("Failure Mode 3: Oversmoothing & Stroke Obliteration (Over-Regularized CLS lambda=1e-1)\n"
             f"Loss of thin character contours, PSNR={compute_interior_psnr(clip_smooth, clean, 12):.1f} dB",
             clip_smooth),
            ("Failure Mode 4: Semi-Convergence Noise Amplification (Over-Iterated RL k=100 at 20 dB)\n"
             f"Multiplicative noise buildup in smooth background regions, PSNR={compute_interior_psnr(clip_rl_over, clean, 12):.1f} dB",
             clip_rl_over),
        ]

        for ax, (lbl, img) in zip(axes, cases):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(lbl, fontsize=9.5, loc="left", pad=4)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 8 Restoration Benchmark.")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of paired plate trials per cell (default 20).")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run (samples=2).")
    parser.add_argument("--reaggregate-only", action="store_true",
                        help="Re-aggregate statistics from existing raw records without re-running restorations.")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--out-dir", type=str, default="experiments/results/stage8",
                        help="Output directory.")
    args = parser.parse_args()

    num_samples = 2 if args.quick else args.samples
    out_dir = ROOT_DIR / args.out_dir

    print("=" * 75)
    print("  STAGE 8: RESTORATION BENCHMARKING AND EXPERIMENTAL EVALUATION")
    print("=" * 75)

    runner = Stage8BenchmarkRunner(out_dir=out_dir, num_samples=num_samples)

    if args.reaggregate_only:
        hw_path = runner.dir_meta / "hardware_environment.json"
        if hw_path.is_file():
            with open(hw_path, "r", encoding="utf-8") as f:
                hw_meta = json.load(f)
        else:
            hw_meta = collect_hardware_metadata()

        core_results = runner.reaggregate_from_existing_raw()

        ext_path = runner.dir_agg / "extensions_results.json"
        if ext_path.is_file():
            with open(ext_path, "r", encoding="utf-8") as f:
                ext_results = json.load(f)
        else:
            ext_results = runner.run_extensions()

        # Update Master Summary JSON
        master_summary = {
            "metadata": {
                "stage": "8.10",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "num_samples_per_cell": num_samples,
                "hardware": hw_meta,
                "status": "reaggregated_with_pairwise_algorithm_comparisons",
            },
            "core_benchmark": core_results,
            "extensions": ext_results,
        }
        master_path = out_dir / "summary.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)
        print(f"\n[Complete] Master summary JSON updated at {master_path}")
        return

    # 1. Hardware Metadata
    hw_meta = collect_hardware_metadata()
    hw_path = runner.dir_meta / "hardware_environment.json"
    with open(hw_path, "w", encoding="utf-8") as f:
        json.dump(hw_meta, f, indent=2)
    print(f"[Meta] Saved {hw_path}")

    # 2. Run Core Benchmark Matrix (C01 - C16)
    t0 = time.perf_counter()
    core_results = runner.run_all_core_cells()
    t_core = time.perf_counter() - t0
    print(f"\n[Done] Core benchmark completed in {t_core:.1f} seconds.")

    # 3. Run Optional Extensions (E1 & E3)
    ext_results = runner.run_extensions()

    # 4. Generate Figures
    if not args.skip_figures:
        runner.generate_all_figures(core_results, ext_results)

    # 5. Save Master Summary JSON
    master_summary = {
        "metadata": {
            "stage": "8.2",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_samples_per_cell": num_samples,
            "hardware": hw_meta,
            "core_matrix_runtime_seconds": t_core,
        },
        "core_benchmark": core_results,
        "extensions": ext_results,
    }
    master_path = out_dir / "summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\n[Complete] Master summary JSON exported to {master_path}")


if __name__ == "__main__":
    main()

