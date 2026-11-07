"""Reproducible experimental evaluation of Richardson--Lucy deconvolution (step 4.5).

Evaluates:
1. Iteration study on clean, 30 dB, and 20 dB BSNR on 7x7 and 51x51 blur kernels;
2. Comprehensive multi-seed comparison (10 independent seeds) across Degraded,
   Inverse, Wiener, Constrained Least Squares (CLS), and Richardson--Lucy (RL);
3. Noise-model investigation: Gaussian noise (project pipeline) vs Poisson noise
   (RL's native statistical likelihood model);
4. Convergence, runtime, MAE, RMSE, and clipping metrics.

Usage:
    python experiments/run_richardson_lucy_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from license_plate_restoration.degradation import add_noise, blur_fft
from license_plate_restoration.psf_builder import (
    combine_psfs,
    make_defocus_psf,
    make_motion_psf,
)
from license_plate_restoration.renderer import generate_plates
from license_plate_restoration.restoration import (
    clip_unit_interval,
    cls_filter,
    inverse_filter,
    richardson_lucy_filter,
    wiener_filter,
    wiener_k_from_bsnr,
)


def compute_metrics(estimate: np.ndarray, reference: np.ndarray,
                    margin: int = 12) -> dict[str, float]:
    """Compute RMSE, PSNR, and MAE on the unpadded plate interior."""
    est = estimate[margin:-margin, margin:-margin].astype(np.float64)
    ref = reference[margin:-margin, margin:-margin].astype(np.float64)
    err = est - ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    psnr = float(-20.0 * np.log10(max(rmse, 1e-12)))
    return {"rmse": rmse, "mae": mae, "psnr": psnr}


def run_iteration_study(plate: np.ndarray, kernels: dict[str, np.ndarray],
                        iteration_counts: list[int]) -> dict:
    """Evaluate RL iteration trajectory across kernels and noise levels."""
    results = {}
    for k_name, psf in kernels.items():
        blurred = blur_fft(plate, psf)
        results[k_name] = {}
        for noise_label, bsnr in (("clean", None), ("30 dB", 30.0), ("20 dB", 20.0)):
            if bsnr is None:
                deg = blurred
            else:
                deg = add_noise(blurred, bsnr, seed=3)
            
            deg_metrics = compute_metrics(deg, plate)
            iters_data = []
            for it in iteration_counts:
                t0 = time.perf_counter()
                restored = richardson_lucy_filter(deg, psf, iterations=it)
                t1 = time.perf_counter()
                
                m = compute_metrics(restored, plate)
                m["iterations"] = it
                m["runtime_ms"] = float((t1 - t0) * 1000.0)
                iters_data.append(m)
                
            results[k_name][noise_label] = {
                "degraded": deg_metrics,
                "iterations": iters_data,
            }
    return results


def run_multi_seed_comparison(seeds: list[int],
                              kernels: dict[str, np.ndarray]) -> dict:
    """Compare Degraded, Inverse, Wiener, CLS, and RL across 10 random seeds."""
    methods_data = {}
    for k_name, psf in kernels.items():
        methods_data[k_name] = {}
        for bsnr in (30.0, 20.0):
            noise_key = f"{bsnr:g} dB"
            records = {
                "degraded": [],
                "inverse": [],
                "wiener": [],
                "cls": [],
                "rl_fixed": [],
                "rl_oracle": [],
            }
            
            w_k = wiener_k_from_bsnr(bsnr)
            cls_lam = 1e-3 if bsnr == 30.0 else 1e-2
            rl_fixed_it = 30 if bsnr == 30.0 else 20
            
            for seed in seeds:
                plate = generate_plates(1, seed=seed)[0][0]
                blurred = blur_fft(plate, psf)
                noisy = add_noise(blurred, bsnr, seed=100 + seed)
                
                # 1. Degraded
                records["degraded"].append(compute_metrics(noisy, plate)["rmse"])
                # 2. Inverse
                records["inverse"].append(compute_metrics(inverse_filter(noisy, psf), plate)["rmse"])
                # 3. Wiener
                records["wiener"].append(compute_metrics(wiener_filter(noisy, psf, K=w_k), plate)["rmse"])
                # 4. CLS
                records["cls"].append(compute_metrics(cls_filter(noisy, psf, regularization=cls_lam), plate)["rmse"])
                # 5. RL fixed deployable
                records["rl_fixed"].append(compute_metrics(richardson_lucy_filter(noisy, psf, iterations=rl_fixed_it), plate)["rmse"])
                # 6. RL oracle best swept
                oracle_errs = [compute_metrics(richardson_lucy_filter(noisy, psf, iterations=it), plate)["rmse"]
                               for it in [5, 10, 15, 20, 30, 50]]
                records["rl_oracle"].append(min(oracle_errs))
                
            summary = {}
            for name, vals in records.items():
                summary[name] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            methods_data[k_name][noise_key] = summary
    return methods_data


def run_poisson_noise_experiment(plate: np.ndarray, psf: np.ndarray,
                                 photon_fluxes: list[int]) -> dict:
    """Evaluate RL, Wiener, and CLS under genuine Poisson observation noise."""
    blurred = blur_fft(plate, psf)
    results = {}
    for photons in photon_fluxes:
        rng = np.random.default_rng(42)
        counts = rng.poisson(photons * np.clip(blurred, 0.0, 1.0))
        noisy_poisson = clip_unit_interval(counts / photons)
        
        d_m = compute_metrics(noisy_poisson, plate)
        w_m = min((compute_metrics(wiener_filter(noisy_poisson, psf, K=k), plate)
                   for k in [1e-4, 1e-3, 1e-2]), key=lambda x: x["rmse"])
        c_m = min((compute_metrics(cls_filter(noisy_poisson, psf, regularization=lam), plate)
                   for lam in [1e-4, 1e-3, 1e-2]), key=lambda x: x["rmse"])
        rl_m = min((compute_metrics(richardson_lucy_filter(noisy_poisson, psf, iterations=it), plate)
                    for it in [5, 10, 20, 30, 50]), key=lambda x: x["rmse"])
        
        results[str(photons)] = {
            "degraded": d_m,
            "wiener": w_m,
            "cls": c_m,
            "rl": rl_m,
        }
    return results


def main() -> None:
    print("=" * 70)
    print("  STAGE 4.5: RICHARDSON--LUCY RESTORATION EXPERIMENTAL EVALUATION")
    print("=" * 70)
    
    plate = generate_plates(1, seed=7)[0][0]
    kernels = {
        "7x7": combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0)),
        "51x51": combine_psfs(make_motion_psf(21, 8.0), make_defocus_psf(4)),
    }
    
    # 1. Iteration study
    print("\n[1] Running Iteration Study (1, 5, 10, 15, 20, 30, 50, 100)...")
    it_counts = [1, 5, 10, 15, 20, 30, 50, 100]
    it_results = run_iteration_study(plate, kernels, it_counts)
    
    for k_name, k_data in it_results.items():
        print(f"\n--- Kernel {k_name} ---")
        for n_label, n_data in k_data.items():
            deg_r = n_data["degraded"]["rmse"]
            deg_p = n_data["degraded"]["psnr"]
            print(f"  {n_label} (Degraded RMSE: {deg_r:.4e}, PSNR: {deg_p:.2f} dB):")
            for entry in n_data["iterations"]:
                it = entry["iterations"]
                r = entry["rmse"]
                p = entry["psnr"]
                ms = entry["runtime_ms"]
                print(f"    iter {it:3d}: RMSE={r:.4e}  PSNR={p:5.2f} dB  ({ms:5.1f} ms)")
                
    # 2. Multi-seed comparison
    print("\n[2] Running Multi-Seed Comparison (10 random seeds)...")
    seeds = list(range(10))
    multi_seed_results = run_multi_seed_comparison(seeds, kernels)
    
    for k_name, k_data in multi_seed_results.items():
        print(f"\n--- Kernel {k_name} (10 seeds) ---")
        for n_label, n_data in k_data.items():
            print(f"  {n_label}:")
            for method, stats in n_data.items():
                m = stats["mean"]
                s = stats["std"]
                print(f"    {method:10s}: RMSE = {m:.4e} +- {s:.2e}")
                
    # 3. Poisson noise study
    print("\n[3] Running Poisson Noise Model Study (native RL model)...")
    poisson_results = run_poisson_noise_experiment(plate, kernels["7x7"],
                                                   [1000, 5000, 20000])
    for photons, p_data in poisson_results.items():
        print(f"  Photons={photons:>5s}: "
              f"Degraded={p_data['degraded']['rmse']:.4e} | "
              f"Wiener={p_data['wiener']['rmse']:.4e} | "
              f"CLS={p_data['cls']['rmse']:.4e} | "
              f"RL={p_data['rl']['rmse']:.4e}")
              
    # Save results to json
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "richardson_lucy_experiment_results.json"
    full_output = {
        "iteration_study": it_results,
        "multi_seed": multi_seed_results,
        "poisson_study": poisson_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nAll experimental results saved to {out_file.as_posix()}")


if __name__ == "__main__":
    main()
