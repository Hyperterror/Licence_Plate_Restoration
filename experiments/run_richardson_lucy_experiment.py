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
    psf_frequency_response,
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


def audit_poisson_clipping(plate: np.ndarray, psf: np.ndarray,
                           photon_fluxes: list[int]) -> dict:
    """Audit the mathematical distortion caused by clip_unit_interval(counts / photons).

    Measures:
    * Number and percentage of pixels exceeding 1.0;
    * Maximum unclipped normalized counts;
    * Negative photometric bias introduced on white pixels and overall;
    * Variance reduction on white pixels caused by upper-tail truncation.
    """
    blurred = blur_fft(plate, psf)
    audit = {}
    for photons in photon_fluxes:
        rng = np.random.default_rng(42)
        counts = rng.poisson(photons * np.clip(blurred, 0.0, 1.0))
        raw = counts / photons
        clipped = np.clip(raw, 0.0, 1.0)

        exceed_mask = raw > 1.0
        exceed_count = int(np.sum(exceed_mask))
        exceed_pct = float(100.0 * exceed_count / raw.size)
        max_val = float(raw.max())

        mask_white = blurred >= 0.999
        theory_var = 1.0 / photons
        white_var_clipped = float(np.var(clipped[mask_white]))
        var_reduction = float(100.0 * (1.0 - white_var_clipped / theory_var))

        audit[str(photons)] = {
            "exceed_pixel_count": exceed_count,
            "exceed_pixel_pct": exceed_pct,
            "max_normalized_count": max_val,
            "raw_mean": float(raw.mean()),
            "clipped_mean": float(clipped.mean()),
            "overall_bias": float(clipped.mean() - blurred.mean()),
            "white_pixel_raw_mean": float(raw[mask_white].mean()),
            "white_pixel_clipped_mean": float(clipped[mask_white].mean()),
            "white_pixel_bias": float(clipped[mask_white].mean() - 1.0),
            "white_pixel_variance_theory": theory_var,
            "white_pixel_variance_clipped": white_var_clipped,
            "white_pixel_variance_reduction_pct": var_reduction,
        }
    return audit


def run_poisson_noise_experiment(seeds: list[int], psf: np.ndarray,
                                 photon_fluxes: list[int],
                                 s_scale: float = 0.70) -> dict:
    """Evaluate RL, Wiener, and CLS under genuine Poisson observation noise.

    Uses calibrated photometric headroom scaling (s_scale=0.70) so that genuine
    Poisson count realizations C ~ Poisson(photons * s_scale * blurred)
    strictly satisfy the public API image contract (values in [0, 1]) without
    a single pixel clipped (headroom > 11 sigma below full scale at 1000 photons).

    Evaluated across multiple random seeds, reporting mean and std for both:
    * Deployable configurations: fixed parameters chosen a priori (K=1e-3, lambda=1e-3, iters=20);
    * Oracle benchmarks: parameters swept and selected against ground-truth clean plate.
    """
    results_by_flux = {}
    for photons in photon_fluxes:
        records = {
            "degraded": [],
            "wiener_deployable": [],
            "cls_deployable": [],
            "rl_deployable": [],
            "wiener_oracle": [],
            "cls_oracle": [],
            "rl_oracle": [],
        }
        for s in seeds:
            plate = generate_plates(1, seed=s)[0][0]
            blurred = blur_fft(plate, psf)
            rng = np.random.default_rng(200 + s)
            counts = rng.poisson(photons * s_scale * np.clip(blurred, 0.0, 1.0))
            obs = (counts / photons).astype(np.float32)
            assert float(obs.max()) <= 1.0 and float(obs.min()) >= 0.0

            # Degraded observation scaled back
            deg_rescaled = np.clip(obs / s_scale, 0.0, 1.0)
            records["degraded"].append(compute_metrics(deg_rescaled, plate)["rmse"])

            # Deployable configurations (fixed parameters chosen a priori)
            w_dep = np.clip(wiener_filter(obs, psf, K=1e-3) / s_scale, 0.0, 1.0)
            records["wiener_deployable"].append(compute_metrics(w_dep, plate)["rmse"])

            c_dep = np.clip(cls_filter(obs, psf, regularization=1e-3) / s_scale, 0.0, 1.0)
            records["cls_deployable"].append(compute_metrics(c_dep, plate)["rmse"])

            rl_dep = np.clip(richardson_lucy_filter(obs, psf, iterations=20) / s_scale, 0.0, 1.0)
            records["rl_deployable"].append(compute_metrics(rl_dep, plate)["rmse"])

            # Oracle benchmarks (swept against ground-truth clean reference)
            w_orc = min(compute_metrics(np.clip(wiener_filter(obs, psf, K=k) / s_scale, 0.0, 1.0), plate)["rmse"]
                        for k in [1e-4, 1e-3, 1e-2])
            records["wiener_oracle"].append(w_orc)

            c_orc = min(compute_metrics(np.clip(cls_filter(obs, psf, regularization=lam) / s_scale, 0.0, 1.0), plate)["rmse"]
                        for lam in [1e-4, 1e-3, 1e-2])
            records["cls_oracle"].append(c_orc)

            rl_orc = min(compute_metrics(np.clip(richardson_lucy_filter(obs, psf, iterations=it) / s_scale, 0.0, 1.0), plate)["rmse"]
                         for it in [5, 10, 15, 20, 30, 50])
            records["rl_oracle"].append(rl_orc)

        summary = {}
        for name, vals in records.items():
            summary[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        results_by_flux[str(photons)] = summary

    return {
        "metadata": {
            "model": "Calibrated Photometric Headroom Scaling (Unclipped Poisson)",
            "s_scale": s_scale,
            "clipping_policy": "Zero clipping; observations strictly within [0, 1] via headroom scaling",
            "margin": 12,
            "seeds": seeds,
            "photon_fluxes": photon_fluxes,
        },
        "results": results_by_flux,
    }


def run_unbounded_count_poisson_experiment(seeds: list[int], psf: np.ndarray,
                                           photon_fluxes: list[int]) -> dict:
    """Evaluate RL directly on unbounded integer photon counts C ~ Poisson(photons * blurred).

    This provides an unconstrained theoretical count-domain benchmark.
    Note: Raw normalized counts C / photons exceed 1.0 (up to 1.14) and cannot be
    consumed by the public API filters (richardson_lucy_filter, wiener_filter,
    cls_filter) due to the [0, 1] range contract in validate_image.
    """
    H = psf_frequency_response(psf, (96, 336))
    H_adj = np.conj(H)
    s_map = float(np.real(H[0, 0]))

    results_by_flux = {}
    for photons in photon_fluxes:
        records = {
            "degraded": [],
            "rl_deployable": [],
            "rl_oracle": [],
        }
        for s in seeds:
            plate = generate_plates(1, seed=s)[0][0]
            blurred = blur_fft(plate, psf)
            rng = np.random.default_rng(200 + s)
            counts = rng.poisson(photons * np.clip(blurred, 0.0, 1.0)).astype(np.float64)

            deg_norm = np.clip(counts / photons, 0.0, 1.0)
            records["degraded"].append(compute_metrics(deg_norm, plate)["rmse"])

            def run_raw_rl(it: int) -> np.ndarray:
                f = np.maximum(counts.copy(), 1e-12)
                for _ in range(it):
                    pred = np.real(np.fft.ifft2(H * np.fft.fft2(f)))
                    r = counts / np.maximum(pred, 1e-12)
                    corr = np.maximum(np.real(np.fft.ifft2(H_adj * np.fft.fft2(r))), 0.0)
                    f = f * (corr / s_map)
                return np.clip(f / photons, 0.0, 1.0)

            records["rl_deployable"].append(compute_metrics(run_raw_rl(20), plate)["rmse"])
            records["rl_oracle"].append(min(compute_metrics(run_raw_rl(it), plate)["rmse"]
                                            for it in [5, 10, 15, 20, 30, 50]))

        summary = {}
        for name, vals in records.items():
            summary[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
        results_by_flux[str(photons)] = summary

    return {
        "metadata": {
            "model": "Unconstrained Photon-Count Domain (Raw Integer Counts)",
            "clipping_policy": "Unbounded counts during iteration; unit interval clipping applied only to final normalized output",
            "api_limitation_note": "Requires separate count-domain iteration; cannot be passed to public API filters due to [0, 1] input contract in validate_image",
            "margin": 12,
            "seeds": seeds,
            "photon_fluxes": photon_fluxes,
        },
        "results": results_by_flux,
    }


def run_superseded_clipped_poisson_experiment(plate: np.ndarray, psf: np.ndarray,
                                              photon_fluxes: list[int]) -> dict:
    """Stage 4.5 single-seed clipped simulation retained strictly for historical traceability.

    SUPERSEDED: Truncates 24.3% of pixels by clipping counts/photons to [0, 1].
    """
    blurred = blur_fft(plate, psf)
    results = {}
    for photons in photon_fluxes:
        rng = np.random.default_rng(42)
        counts = rng.poisson(photons * np.clip(blurred, 0.0, 1.0))
        noisy_poisson = clip_unit_interval(counts / photons)

        d_m = compute_metrics(noisy_poisson, plate)
        w_deploy = compute_metrics(wiener_filter(noisy_poisson, psf, K=1e-3), plate)
        c_deploy = compute_metrics(cls_filter(noisy_poisson, psf, regularization=1e-3), plate)
        rl_deploy = compute_metrics(richardson_lucy_filter(noisy_poisson, psf, iterations=20), plate)

        w_m = min((compute_metrics(wiener_filter(noisy_poisson, psf, K=k), plate)
                   for k in [1e-4, 1e-3, 1e-2]), key=lambda x: x["rmse"])
        c_m = min((compute_metrics(cls_filter(noisy_poisson, psf, regularization=lam), plate)
                   for lam in [1e-4, 1e-3, 1e-2]), key=lambda x: x["rmse"])
        rl_m = min((compute_metrics(richardson_lucy_filter(noisy_poisson, psf, iterations=it), plate)
                    for it in [5, 10, 20, 30, 50]), key=lambda x: x["rmse"])

        results[str(photons)] = {
            "degraded": d_m,
            "wiener_deployable": w_deploy,
            "cls_deployable": c_deploy,
            "rl_deployable": rl_deploy,
            "wiener_oracle": w_m,
            "cls_oracle": c_m,
            "rl_oracle": rl_m,
        }
    return {
        "metadata": {
            "status": "SUPERSEDED",
            "defect": "Artificially clips counts/photons to [0, 1], truncating 24.3% of pixels and reducing white pixel variance by 66.8%",
            "seed": 42,
            "plate_seed": 7,
            "margin": 12,
        },
        "results": results,
    }


def main() -> None:
    print("=" * 70)
    print("  STAGE 4.5.2: RICHARDSON--LUCY EXPERIMENTAL EVALUATION & AUDIT")
    print("=" * 70)

    plate = generate_plates(1, seed=7)[0][0]
    kernels = {
        "7x7": combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0)),
        "51x51": combine_psfs(make_motion_psf(21, 8.0), make_defocus_psf(4)),
    }

    # 1. Iteration study (Gaussian Noise)
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

    # 2. Multi-seed comparison (Gaussian Noise)
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

    # 3. Audit of clipping defect in Stage 4.5
    print("\n[3] Auditing Poisson Noise Upper-Tail Clipping Distortion...")
    poisson_audit = audit_poisson_clipping(plate, kernels["7x7"], [1000, 5000, 20000])
    for photons, a_data in poisson_audit.items():
        cnt = a_data["exceed_pixel_count"]
        pct = a_data["exceed_pixel_pct"]
        max_v = a_data["max_normalized_count"]
        b_wh = a_data["white_pixel_bias"]
        v_red = a_data["white_pixel_variance_reduction_pct"]
        print(f"  Photons={photons:>5s}: {cnt}/32256 pixels ({pct:.2f}%) exceed 1.0 (max={max_v:.4f}) | "
              f"White Bias={b_wh:.4f} | Var Red={v_red:.1f}%")

    # 4. Corrected Poisson noise study: Calibrated Headroom Scaling (10 seeds)
    print("\n[4] Running Corrected Poisson Study via Calibrated Headroom Scaling (10 seeds)...")
    headroom_poisson = run_poisson_noise_experiment(seeds, kernels["7x7"],
                                                    [1000, 5000, 20000], s_scale=0.70)
    for photons, p_data in headroom_poisson["results"].items():
        print(f"  Photons={photons:>5s}:")
        print(f"    Degraded:          RMSE = {p_data['degraded']['mean']:.4e} +- {p_data['degraded']['std']:.2e}")
        print(f"    Deployable: Wiener={p_data['wiener_deployable']['mean']:.4e} | CLS={p_data['cls_deployable']['mean']:.4e} | RL={p_data['rl_deployable']['mean']:.4e}")
        print(f"    Oracle:     Wiener={p_data['wiener_oracle']['mean']:.4e} | CLS={p_data['cls_oracle']['mean']:.4e} | RL={p_data['rl_oracle']['mean']:.4e}")

    # 5. Direct unbounded photon-count RL benchmark (10 seeds)
    print("\n[5] Running Direct Unbounded Photon-Count RL Benchmark (10 seeds)...")
    unbounded_poisson = run_unbounded_count_poisson_experiment(seeds, kernels["7x7"],
                                                               [1000, 5000, 20000])
    for photons, u_data in unbounded_poisson["results"].items():
        print(f"  Photons={photons:>5s}: Deployable RL={u_data['rl_deployable']['mean']:.4e} +- {u_data['rl_deployable']['std']:.2e} | "
              f"Oracle RL={u_data['rl_oracle']['mean']:.4e} +- {u_data['rl_oracle']['std']:.2e}")

    # 6. Retain superseded single-seed clipped results for traceability
    superseded_poisson = run_superseded_clipped_poisson_experiment(plate, kernels["7x7"],
                                                                   [1000, 5000, 20000])

    # Save comprehensive results to json
    out_dir = Path("experiments")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "richardson_lucy_experiment_results.json"
    full_output = {
        "iteration_study": it_results,
        "multi_seed_gaussian": multi_seed_results,
        "poisson_clipping_audit": poisson_audit,
        "poisson_headroom_scaling_study": headroom_poisson,
        "poisson_unbounded_count_study": unbounded_poisson,
        "superseded_clipped_stage4_5": superseded_poisson,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nAll experimental results saved to {out_file.as_posix()}")


if __name__ == "__main__":
    main()
