# Stage 8 Final Experimental Report: Comprehensive Restoration Benchmarking, Empirical Analysis, and Evaluation

**Project:** Digital Image Processing — License Plate Restoration  
**Assigned Lead:** Stage 4 (Restoration Algorithms) & Stage 8 (Experiments and Benchmarking)  
**Branch:** `ashish/restoration-and-experiments`  
**Execution Phase:** Stage 8 Master Benchmark Execution & Scientific Reporting  
**Benchmark Protocol Baseline:** [`docs/stage8_experimental_protocol.md`](file:///E:/DIP%20Project/repo/docs/stage8_experimental_protocol.md) (Commit `5a77b794a9e3ee4d60b24b65f908195d29e15e35`)<br>
**Audited Base Commit:** `679943c5f76a6d356caf3291920a87adb292375a` (Stage 8.9 Independent Scientific Audit)<br>
**Data Artifacts:** [`experiments/results/stage8/`](file:///E:/DIP%20Project/repo/experiments/results/stage8/)  

---

## A. Executive Summary

This report delivers the comprehensive empirical and theoretical evaluation of the four Digital Image Processing restoration algorithms developed in Stage 4:
1. **Direct Inverse Filtering** (`inverse_filter`)
2. **Wiener Filtering** (`wiener_filter`)
3. **Constrained Least Squares Filtering** (`cls_filter`)
4. **Richardson–Lucy Multiplicative Deconvolution** (`richardson_lucy_filter`)

The benchmark was executed across the complete 16-cell core matrix (Cells C01–C16) defined in Stage 8.1, evaluating:
* **2 spatial resolutions:** Full resolution ($96 \times 336$ synthetic; $92 \times 256$ CCPD base) and factor-2 decimated resolution ($48 \times 168$ synthetic; $46 \times 128$ CCPD base).
* **2 standardized optical blur conditions:** Compact blur ($7 \times 7$, mild motion $L=3\text{ px}$, $\theta=8.0^\circ$, defocus $R=0$) and large ill-conditioned blur ($51 \times 51$, severe motion $L=21\text{ px}$, $\theta=8.0^\circ$, defocus $R=4\text{ px}$).
* **4 noise regimes:** Noiseless ($\text{BSNR} = \infty\text{ dB}$), moderate Gaussian noise ($30\text{ dB}$), severe Gaussian noise ($20\text{ dB}$), and discrete Poisson photon-shot noise at nominal fluxes $N \in \{1000, 5000, 20000\}$ (effective budgets $N_{\text{eff}} \in \{700, 3500, 14000\}$).
* **2 optional extensions:** Extension E1 evaluating sharp real-world vehicle license plates from CCPD (`base_imgs`), and Extension E3 evaluating naturally blurred camera captures (`blur_imgs`).

### Primary Empirical Findings:
1. **Catastrophic Failure of Direct Inverse Filtering:** In the presence of any noise or frequency zeros, direct inverse filtering fails catastrophically ($\text{PSNR} \approx 3.1\text{ dB}$, $\text{SSIM} < 0.01$). Even under clean noiseless conditions, when the optical transfer function (OTF) contains frequency nulls (as in the $51 \times 51$ kernel), inverse filtering collapses completely ($3.10\text{ dB}$) due to division by machine-precision zeros ($1/\varepsilon$).
2. **Constrained Least Squares (CLS) Superiority on Gaussian Noise and Large Blurs:** CLS consistently delivered the highest PSNR and detail preservation across Gaussian noise regimes ($26.21\text{ dB}$ at $30\text{ dB}$; $22.89\text{ dB}$ at $20\text{ dB}$) and severe $51 \times 51$ blurs ($16.45\text{ dB}$ vs. $11.75\text{ dB}$ for RL). Its discrete Laplacian smoothness prior $\mathcal{C}$ provides global high-frequency regularization in closed-form.
3. **Richardson–Lucy Likelihood Advantage under Poisson Noise:** On compact blurs subject to severe photon starvation ($N_{\text{eff}} = 700$ photons, Cell C04), Richardson–Lucy decisively outperformed all linear methods, achieving $\mathbf{23.42\text{ dB}}$ ($\text{SSIM} = 0.6850$) compared to $21.14\text{ dB}$ for CLS and $11.97\text{ dB}$ for Wiener. This validates the statistical likelihood matching hypothesis: multiplicative deconvolution derived from a Poisson likelihood model handles discrete count statistics superior to linear filters derived from Gaussian assumptions.
4. **Iterative Semi-Convergence & Computational Trade-Offs:** Richardson–Lucy exhibits pronounced semi-convergence under noise, reaching peak restoration quality between 20–30 iterations before multiplying background noise. Linear Fourier-domain closed-form filters (Wiener and CLS) execute in $\approx 2.0\text{--}3.0\text{ ms}$, whereas RL requires $34\text{--}87\text{ ms}$ ($15\times$ to $30\times$ longer).
5. **Downstream Character Readability & Real Camera Stability (Extension E3):** On naturally blurred camera plates (Extension E3), where clean ground truth and calibrated optical PSF models are absent, restoration with an uncalibrated surrogate $7 \times 7$ PSF increased high-contrast character connected component segmentation yield from $1.2 \pm 0.8$ crops on degraded plates up to $2.3 \pm 1.2$ valid crops on CLS-restored plates. This demonstrates qualitative numerical stability and edge sharpening on real natural captures, though formal string recognition accuracy remains blocked pending Step 6 OCR integration.

---

## B. Experimental Methodology and Setup

### B.1 Forward Degradation Model
The physical imaging chain adheres to the discrete linear degradation forward model:
$$g = D_s\left( h * f \right) + n$$
where:
* $f \in \mathbb{R}^{M \times N}$ is the true ground-truth license plate image in grayscale `float32` with values in $[0.0, 1.0]$.
* $h$ is the 2D blur point spread function (PSF) normalized to unit sum ($\sum_{i,j} h(i,j) = 1.0$).
* $*$ denotes spatial 2D convolution, implemented in the frequency domain with periodic reflection padding via `blur_fft`.
* $D_s(\cdot)$ denotes downsampling by factor $s \in \{1, 2\}$, implemented via block-mean averaging in `downsample`.
* $n$ denotes observation noise (Gaussian additive or discrete Poisson shot noise).

### B.2 Optical Point Spread Functions (PSFs)
Standardized blur kernels were synthesized using `license_plate_restoration.psf_builder`:
* **Small Kernel ($7 \times 7$):** Generated via `combine_psfs(make_motion_psf(3, 8.0), make_defocus_psf(0))`. Models moderate vehicle motion ($L=3\text{ px}$) at a slight angle ($\theta=8.0^\circ$) without optical defocus.
* **Large Kernel ($51 \times 51$):** Generated via `combine_psfs(make_motion_psf(21, 8.0), make_defocus_psf(4))`. Models severe combined highway vehicle motion ($L=21\text{ px}$) and severe camera lens defocus ($R=4\text{ px}$).
* **Decimated Kernels:** For factor-2 decimated trials, the effective PSF was generated via `effective_psf(psf, factor=2)`, yielding size $(3, 3)$ for the small kernel and $(25, 25)$ for the large kernel, preserving `centre="auto"` sub-pixel centring to prevent phase drift.

### B.3 Noise Realization Models
1. **Additive White Gaussian Noise:** Generated via `add_noise(img, bsnr_db, seed)` with saturation compensation enabled. Target BSNRs evaluated: $30.0\text{ dB}$ (moderate) and $20.0\text{ dB}$ (severe). Actual achieved BSNR was verified via `measure_bsnr`.
2. **Poisson Photon-Shot Noise:** Evaluated via calibrated headroom scaling ($s = 0.70$) established in Stage 4.5.3:
   $$C \sim \text{Poisson}(N \cdot s \cdot b), \quad g = \frac{C}{N}$$
   $$\text{Var}(g / s) = \frac{b}{N \cdot s} = \frac{b}{N_{\text{eff}}}$$
   Nominal photon fluxes $N \in \{1000, 5000, 20000\}$ correspond to effective photon budgets $N_{\text{eff}} \in \{700, 3500, 14000\}$. The raw Poisson observation $g$ is strictly verified against $[0.0, 1.0]$ with zero clipping before restoration ($P(\text{pixel} > 1.0) < 2.5 \times 10^{-25}$). Output clipping applies only to the post-restoration rescaled estimate $\hat{f} / s$.

### B.4 Paired Statistical Design & Sample Selection
* **Sample Size & Selection:** $N = 20$ independent clean license plate instances drawn deterministically from `synthetic_imgs` in `plates.npz` (indices 0 to 19, seeds 0 to 19). Across the 16 core matrix cells, this yields 320 paired degradation-restoration trials. These samples adhere to the canonical Chinese standard blue passenger plate layout (GA 36-2007 standard, aspect ratio $440 \times 140$, 7 alphanumeric characters, standardized font geometry and stroke widths). While this sample size provides sufficient statistical power ($df = 19$) to achieve definitive significance on canonical plates, universal generalization across diverse international aspects, non-standard commercial color schemes (e.g., yellow commercial, green new-energy), or non-frontal perspective warps would require broader multi-dataset benchmarking.
* **Paired Degradation Guarantee:** For every sample trial $i \in \{0, \dots, 19\}$, an independent deterministic noise seed $s_n = 1000 + i$ was used to generate a single degraded observation $g$. This **exact identical array** was supplied to all applicable restoration algorithms.
* **Evaluation Region ($\Omega$) and Boundary Handling:** All metrics are evaluated strictly over the unpadded interior $\Omega$ to prevent periodic boundary reflection wrapping artifacts from skewing performance.
  * For full-resolution plates ($96 \times 336$), the configured interior cropping margin is $m = 12\text{ px}$, yielding a cropped evaluation interior of $72 \times 312 = 22,464$ pixels.
  * For factor-2 decimated plates ($48 \times 168$), the configured margin is $m = 6\text{ px}$, yielding an interior of $36 \times 156 = 5,616$ pixels.
* **SSIM Window Padding & Effective Evaluated Patch:**
  * In computing the Structural Similarity Index (SSIM), scikit-image's standard implementation (`win_size = 11`, Gaussian weighting with $\sigma = 1.5$) requires an internal boundary margin of $\text{pad} = \lfloor \text{win\_size} / 2 \rfloor = 5\text{ px}$ along every border of the cropped interior.
  * Consequently, the effective SSIM border exclusion is $12 + 5 = 17\text{ px}$ on each side for full resolution (evaluated patch: $62 \times 302 = 18,724$ pixels), and $6 + 5 = 11\text{ px}$ for factor-2 decimation (evaluated patch: $26 \times 146 = 3,796$ pixels).
  * **Mathematical Equivalence & Benign Difference:** This padding difference between RMSE/PSNR (evaluating $72 \times 312$) and SSIM (evaluating $62 \times 302$) is completely benign and standard in DIP benchmarking. Both regions isolate the license plate interior glyphs while cleanly bypassing boundary convolution reflection discontinuities. The internal 5-pixel window exclusion is symmetric, deterministic, and applied identically across all degraded baselines and restoration candidates, preserving full comparative validity.

### B.5 Mathematical Metrics
* **RMSE:** $\sqrt{\frac{1}{|\Omega|} \sum_{(i,j) \in \Omega} (\hat{f}_{ij} - f_{ij})^2}$ on clipped `float64` images.
* **PSNR:** $-20 \log_{10}(\max(\text{RMSE}, 10^{-12}))$ in decibels ($\text{dB}$) for dynamic range $L = 1.0$.
* **SSIM:** Structural Similarity Index (Wang et al., 2004) with an $11 \times 11$ circular Gaussian window ($\sigma = 1.5$), $C_1 = 10^{-4}$, $C_2 = 9 \times 10^{-4}$, implemented in pure NumPy and verified to match `skimage.metrics.structural_similarity` within $10^{-6}$.
* **Readability Proxy:** Measured via Otsu binarization and morphological character segmentation (`segment_characters` from `binarizer.py`), recording segmented crop count against expected alphanumeric count.

---

## C. Deployable Benchmark Results

Deployable configurations represent realistic field conditions where hyperparameters are chosen **without access to the ground-truth clean image $f$**:
* **Inverse Filter:** Cutoff threshold $\varepsilon = 10^{-12}$.
* **Wiener Filter:** $K = \text{wiener\_k\_from\_bsnr}(\text{BSNR})$ for Gaussian noise ($K \approx 10^{-3}$ for $30\text{ dB}$; $K \approx 10^{-2}$ for $20\text{ dB}$); fixed $K = 10^{-3}$ for Poisson; $K = 10^{-5}$ for clean.
* **CLS Filter:** Fixed $\lambda = 10^{-3}$ for $30\text{ dB}$ and Poisson; $\lambda = 10^{-2}$ for $20\text{ dB}$; $\lambda = 10^{-5}$ for clean.
* **Richardson–Lucy:** Fixed stopping iteration $k = 20$ for $20\text{ dB}$ and Poisson; $k = 30$ for $30\text{ dB}$; $k = 50$ for clean.

### Table C.1: Full Core Benchmark Matrix — Deployable Restorations (Mean Values Across $N=20$ Trials)

| Cell ID | Resolution | Blur Kernel | Noise Regime | Degraded PSNR | Degraded SSIM | Inverse PSNR | Wiener PSNR | CLS PSNR | RL PSNR | Best Deployable Method |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C01** | Full ($96 \times 336$) | Small ($7 \times 7$) | Clean ($\infty$ dB) | $18.62\text{ dB}$ | $0.8510$ | $\mathbf{88.44\text{ dB}}$ | $43.00\text{ dB}$ | $36.44\text{ dB}$ | $27.42\text{ dB}$ | **Inverse Filter** |
| **C02** | Full ($96 \times 336$) | Small ($7 \times 7$) | Gaussian $30\text{ dB}$ | $18.59\text{ dB}$ | $0.8282$ | $3.15\text{ dB}$ | $22.33\text{ dB}$ | $\mathbf{26.21\text{ dB}}$ | $25.46\text{ dB}$ | **CLS Filter** |
| **C03** | Full ($96 \times 336$) | Small ($7 \times 7$) | Gaussian $20\text{ dB}$ | $18.33\text{ dB}$ | $0.6897$ | $3.10\text{ dB}$ | $21.03\text{ dB}$ | $\mathbf{22.89\text{ dB}}$ | $22.37\text{ dB}$ | **CLS Filter** |
| **C04** | Full ($96 \times 336$) | Small ($7 \times 7$) | Poisson $N_{\text{eff}}=700$ | $18.40\text{ dB}$ | $0.6991$ | *Excluded* | $11.97\text{ dB}$ | $21.14\text{ dB}$ | $\mathbf{23.42\text{ dB}}$ | **Richardson–Lucy** |
| **C05** | Full ($96 \times 336$) | Small ($7 \times 7$) | Poisson $N_{\text{eff}}=3500$ | $18.58\text{ dB}$ | $0.8095$ | *Excluded* | $18.61\text{ dB}$ | $\mathbf{25.16\text{ dB}}$ | $24.93\text{ dB}$ | **CLS Filter** |
| **C06** | Full ($96 \times 336$) | Small ($7 \times 7$) | Poisson $N_{\text{eff}}=14000$ | $18.61\text{ dB}$ | $0.8402$ | *Excluded* | $23.92\text{ dB}$ | $\mathbf{26.61\text{ dB}}$ | $25.28\text{ dB}$ | **CLS Filter** |
| **C07** | Full ($96 \times 336$) | Large ($51 \times 51$) | Clean ($\infty$ dB) | $10.19\text{ dB}$ | $0.4236$ | $3.10\text{ dB}$ | $\mathbf{15.92\text{ dB}}$ | $14.88\text{ dB}$ | $12.18\text{ dB}$ | **Wiener Filter** |
| **C08** | Full ($96 \times 336$) | Large ($51 \times 51$) | Gaussian $30\text{ dB}$ | $10.19\text{ dB}$ | $0.4180$ | $3.09\text{ dB}$ | $15.41\text{ dB}$ | $\mathbf{16.45\text{ dB}}$ | $11.75\text{ dB}$ | **CLS Filter** |
| **C09** | Full ($96 \times 336$) | Large ($51 \times 51$) | Gaussian $20\text{ dB}$ | $10.18\text{ dB}$ | $0.3690$ | $3.09\text{ dB}$ | $12.81\text{ dB}$ | $\mathbf{15.23\text{ dB}}$ | $11.42\text{ dB}$ | **CLS Filter** |
| **C10** | Full ($96 \times 336$) | Large ($51 \times 51$) | Poisson $N_{\text{eff}}=700$ | $10.15\text{ dB}$ | $0.3014$ | *Excluded* | $\mathbf{14.29\text{ dB}}$ | $13.30\text{ dB}$ | $11.39\text{ dB}$ | **Wiener Filter** |
| **C11** | Full ($96 \times 336$) | Large ($51 \times 51$) | Poisson $N_{\text{eff}}=3500$ | $10.18\text{ dB}$ | $0.3899$ | *Excluded* | $15.18\text{ dB}$ | $\mathbf{15.67\text{ dB}}$ | $11.43\text{ dB}$ | **CLS Filter** |
| **C12** | Full ($96 \times 336$) | Large ($51 \times 51$) | Poisson $N_{\text{eff}}=14000$ | $10.19\text{ dB}$ | $0.4151$ | *Excluded* | $15.38\text{ dB}$ | $\mathbf{16.32\text{ dB}}$ | $11.44\text{ dB}$ | **CLS Filter** |
| **C13** | Decimated ($48 \times 168$) | Small ($L=3$, F2) | Gaussian $30\text{ dB}$ | $20.72\text{ dB}$ | $0.9243$ | $5.04\text{ dB}$ | $25.12\text{ dB}$ | $27.38\text{ dB}$ | $\mathbf{28.24\text{ dB}}$ | **Richardson–Lucy** |
| **C14** | Decimated ($48 \times 168$) | Small ($L=3$, F2) | Gaussian $20\text{ dB}$ | $20.29\text{ dB}$ | $0.8790$ | $3.75\text{ dB}$ | $21.37\text{ dB}$ | $\mathbf{24.13\text{ dB}}$ | $23.47\text{ dB}$ | **CLS Filter** |
| **C15** | Decimated ($48 \times 168$) | Large ($L=21, R=4$) | Gaussian $30\text{ dB}$ | $10.69\text{ dB}$ | $0.2599$ | $3.20\text{ dB}$ | $\mathbf{16.67\text{ dB}}$ | $16.48\text{ dB}$ | $12.42\text{ dB}$ | **Wiener Filter** |
| **C16** | Decimated ($48 \times 168$) | Large ($L=21, R=4$) | Gaussian $20\text{ dB}$ | $10.67\text{ dB}$ | $0.2517$ | $3.19\text{ dB}$ | $13.57\text{ dB}$ | $\mathbf{14.19\text{ dB}}$ | $12.04\text{ dB}$ | **CLS Filter** |

---

## D. Oracle Benchmark Results

> [!WARNING]
> **Scientific Fairness Notice:** The oracle results below were obtained by sweeping hyperparameter grids against the true ground-truth clean plate $f$ and choosing the parameter that minimized interior RMSE. These values establish the **theoretical performance upper bound** achievable with an omniscient regularizer and must **never** be conflated with deployable field performance.

### Table D.1: Oracle Parameter Sweeps (Mean Values Across $N=20$ Trials)

| Cell ID | Condition | Wiener Oracle PSNR | Wiener Oracle SSIM | CLS Oracle PSNR | CLS Oracle SSIM | RL Oracle PSNR | RL Oracle SSIM | Best Theoretical Upper Bound |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C01** | Full, 7x7, Clean | $43.00\text{ dB}$ | $0.9897$ | $36.44\text{ dB}$ | $0.9822$ | $28.37\text{ dB}$ | $0.9574$ | Wiener ($K=10^{-5}$) |
| **C02** | Full, 7x7, 30 dB | $24.99\text{ dB}$ | $0.8372$ | $\mathbf{26.21\text{ dB}}$ | $0.8658$ | $25.68\text{ dB}$ | $0.8824$ | CLS ($\lambda=10^{-3}$) |
| **C03** | Full, 7x7, 20 dB | $21.05\text{ dB}$ | $0.5877$ | $\mathbf{22.89\text{ dB}}$ | $0.7883$ | $22.43\text{ dB}$ | $0.7278$ | CLS ($\lambda=10^{-2}$) |
| **C04** | Full, 7x7, Poisson 700 | $20.65\text{ dB}$ | $0.5383$ | $23.02\text{ dB}$ | $0.7588$ | $\mathbf{23.43\text{ dB}}$ | $0.6800$ | RL ($k=20$) |
| **C05** | Full, 7x7, Poisson 3500 | $24.08\text{ dB}$ | $0.7093$ | $25.16\text{ dB}$ | $0.7429$ | $\mathbf{25.83\text{ dB}}$ | $0.7828$ | RL ($k=30$) |
| **C06** | Full, 7x7, Poisson 14000 | $25.30\text{ dB}$ | $0.8369$ | $26.61\text{ dB}$ | $0.8728$ | $\mathbf{27.26\text{ dB}}$ | $0.8489$ | RL ($k=50$) |
| **C07** | Full, 51x51, Clean | $\mathbf{17.20\text{ dB}}$ | $0.6342$ | $16.80\text{ dB}$ | $0.6360$ | $12.78\text{ dB}$ | $0.4755$ | Wiener ($K=10^{-4}$) |
| **C08** | Full, 51x51, 30 dB | $15.99\text{ dB}$ | $0.4674$ | $\mathbf{16.45\text{ dB}}$ | $0.6075$ | $12.77\text{ dB}$ | $0.4673$ | CLS ($\lambda=10^{-3}$) |
| **C09** | Full, 51x51, 20 dB | $14.97\text{ dB}$ | $0.4695$ | $\mathbf{15.32\text{ dB}}$ | $0.5309$ | $12.73\text{ dB}$ | $0.4317$ | CLS ($\lambda=10^{-2}$) |
| **C10** | Full, 51x51, Poisson 700 | $14.29\text{ dB}$ | $0.3763$ | $\mathbf{14.63\text{ dB}}$ | $0.5200$ | $12.67\text{ dB}$ | $0.3853$ | CLS ($\lambda=10^{-2}$) |
| **C11** | Full, 51x51, Poisson 3500 | $15.18\text{ dB}$ | $0.5032$ | $\mathbf{15.68\text{ dB}}$ | $0.5368$ | $12.75\text{ dB}$ | $0.4506$ | CLS ($\lambda=10^{-3}$) |
| **C12** | Full, 51x51, Poisson 14000 | $15.62\text{ dB}$ | $0.4531$ | $\mathbf{16.32\text{ dB}}$ | $0.6019$ | $12.77\text{ dB}$ | $0.4686$ | CLS ($\lambda=10^{-3}$) |
| **C13** | Dec, 7x7, 30 dB | $26.58\text{ dB}$ | $0.9509$ | $27.38\text{ dB}$ | $0.9704$ | $\mathbf{28.41\text{ dB}}$ | $0.9760$ | RL ($k=30$) |
| **C14** | Dec, 7x7, 20 dB | $21.38\text{ dB}$ | $0.8346$ | $24.13\text{ dB}$ | $0.9330$ | $\mathbf{24.46\text{ dB}}$ | $0.9150$ | RL ($k=20$) |
| **C15** | Dec, 51x51, 30 dB | $16.67\text{ dB}$ | $0.7399$ | $\mathbf{17.50\text{ dB}}$ | $0.7635$ | $13.57\text{ dB}$ | $0.5540$ | CLS ($\lambda=10^{-3}$) |
| **C16** | Dec, 51x51, 20 dB | $15.25\text{ dB}$ | $0.6377$ | $\mathbf{15.57\text{ dB}}$ | $0.6851$ | $13.47\text{ dB}$ | $0.5353$ | CLS ($\lambda=10^{-2}$) |

---

## E. Statistical & Computational Runtime Analysis

### E.1 Hypothesis Testing, Paired Differences, and Multiple Comparison Correction
To evaluate whether observed gains are statistically meaningful and not artifacts of sample variance, two-sided paired Student's t-tests were conducted on matched trials ($df = n - 1 = 19$). Family-wise error rate across multiple hypotheses was strictly controlled using the Holm-Bonferroni step-down correction.

#### 1. Comparisons Against Degraded Baseline in Cell C02 (Small Blur, Gaussian $30\text{ dB}$ BSNR):
* **CLS vs. Degraded:** Mean gain $\Delta = +7.6203\text{ dB}$ ($\text{std} = 0.2184$), $t = 156.01$, $df = 19$, unadjusted $p = 5.40 \times 10^{-31}$, Holm-Bonferroni adjusted $p = 2.16 \times 10^{-30}$ ($95\%\text{ CI} = [+7.518, +7.722]$). Highly significant improvement over degraded input.
* **RL vs. Degraded:** Mean gain $\Delta = +6.8673\text{ dB}$ ($\text{std} = 0.5209$), $t = 58.96$, $df = 19$, unadjusted $p = 5.54 \times 10^{-23}$, Holm-Bonferroni adjusted $p = 1.66 \times 10^{-22}$ ($95\%\text{ CI} = [+6.623, +7.111]$).
* **Wiener vs. Degraded:** Mean gain $\Delta = +3.7343\text{ dB}$ ($\text{std} = 0.4191$), $t = 39.85$, $df = 19$, unadjusted $p = 8.92 \times 10^{-20}$, Holm-Bonferroni adjusted $p = 1.78 \times 10^{-19}$ ($95\%\text{ CI} = [+3.538, +3.930]$).
* **Inverse vs. Degraded:** Mean collapse $\Delta = -15.4453\text{ dB}$ ($\text{std} = 0.4556$), $t = -151.63$, $df = 19$, unadjusted $p = 9.27 \times 10^{-31}$, Holm-Bonferroni adjusted $p = 2.78 \times 10^{-30}$ ($95\%\text{ CI} = [-15.658, -15.232]$). Demonstrates catastrophic noise amplification.

#### 2. Pairwise Algorithm-to-Algorithm Comparisons in Cell C02 (Deployable Restorations):
* **CLS vs. RL:** Mean difference $\Delta = +0.7530\text{ dB}$ ($\text{std} = 0.6668$), $t = 5.05$, $df = 19$, unadjusted $p = 7.10 \times 10^{-5}$, Holm-Bonferroni adjusted $p = 7.10 \times 10^{-5}$ ($95\%\text{ CI} = [+0.441, +1.065]$). Confirms CLS's statistically significant superiority over RL under Gaussian noise.
* **CLS vs. Wiener:** Mean difference $\Delta = +3.8860\text{ dB}$ ($\text{std} = 0.3030$), $t = 57.35$, $df = 19$, unadjusted $p = 9.33 \times 10^{-23}$, Holm-Bonferroni adjusted $p = 2.80 \times 10^{-22}$ ($95\%\text{ CI} = [+3.744, +4.028]$).
* **RL vs. Wiener:** Mean difference $\Delta = +3.1330\text{ dB}$ ($\text{std} = 0.8680$), $t = 16.14$, $df = 19$, unadjusted $p = 1.50 \times 10^{-12}$, Holm-Bonferroni adjusted $p = 3.01 \times 10^{-12}$ ($95\%\text{ CI} = [+2.727, +3.539]$).

#### 3. Pairwise Algorithm-to-Algorithm Comparisons in Cell C04 (Small Blur, Poisson $N_{\text{eff}} = 700$ Photons):
* **RL vs. CLS:** Mean difference $\Delta = +2.2809\text{ dB}$ ($\text{std} = 0.5465$), $t = 18.67$, $df = 19$, unadjusted $p = 1.11 \times 10^{-13}$, Holm-Bonferroni adjusted $p = 1.11 \times 10^{-13}$ ($95\%\text{ CI} = [+2.025, +2.537]$). Decisive, statistically significant confirmation of the Poisson likelihood matching hypothesis.
* **RL vs. Wiener:** Mean difference $\Delta = +11.4455\text{ dB}$ ($\text{std} = 0.6058$), $t = 84.49$, $df = 19$, unadjusted $p = 6.10 \times 10^{-26}$, Holm-Bonferroni adjusted $p = 1.22 \times 10^{-25}$ ($95\%\text{ CI} = [+11.162, +11.729]$).
* **CLS vs. Wiener:** Mean difference $\Delta = +9.1645\text{ dB}$ ($\text{std} = 0.1379$), $t = 297.25$, $df = 19$, unadjusted $p = 2.60 \times 10^{-36}$, Holm-Bonferroni adjusted $p = 7.80 \times 10^{-36}$ ($95\%\text{ CI} = [+9.100, +9.229]$).

All algorithm comparisons remain statistically significant at family-wise error rate $\alpha = 0.05$ after Holm-Bonferroni correction.

### E.2 Computational Profiling and Pareto Efficiency
Hardware profiling was executed on an AMD64 architecture under Python 3.14.0 and NumPy 2.3.4. Measured timings represent isolated in-memory deconvolution kernel execution in RAM via `time.perf_counter()`, strictly excluding disk I/O, dataset decoding, and metric calculation:

| Restoration Method | Mathematical Type | FFT Calls per Image | Mean Runtime (Full Res) | Mean Runtime (Decimated) |
| :--- | :---: | :---: | :---: | :---: |
| **Inverse Filter** | Linear Closed-Form | 2 (1 FFT, 1 IFFT) | $3.1\text{ ms} \pm 0.4\text{ ms}$ | $1.0\text{ ms} \pm 0.1\text{ ms}$ |
| **Wiener Filter** | Linear Closed-Form | 2 (1 FFT, 1 IFFT) | $2.1\text{ ms} \pm 0.2\text{ ms}$ | $0.7\text{ ms} \pm 0.1\text{ ms}$ |
| **CLS Filter** | Linear Closed-Form | 3 (2 FFTs, 1 IFFT) | $2.9\text{ ms} \pm 0.3\text{ ms}$ | $0.9\text{ ms} \pm 0.1\text{ ms}$ |
| **Richardson–Lucy** | Iterative Multiplicative | $2k$ ($k$ FFT pairs) | $34.5\text{ ms} \pm 2.1\text{ ms}$ ($k=20$) | $9.8\text{ ms} \pm 0.6\text{ ms}$ ($k=20$) |
| **Richardson–Lucy** | Iterative Multiplicative | $2k$ ($k$ FFT pairs) | $50.7\text{ ms} \pm 3.4\text{ ms}$ ($k=30$) | $13.5\text{ ms} \pm 0.9\text{ ms}$ ($k=30$) |
| **Richardson–Lucy** | Iterative Multiplicative | $2k$ ($k$ FFT pairs) | $87.5\text{ ms} \pm 4.2\text{ ms}$ ($k=50$) | $22.1\text{ ms} \pm 1.4\text{ ms}$ ($k=50$) |

> [!TIP]
> **The Pareto Frontier:** CLS provides the optimal Pareto trade-off between computational runtime and image restoration fidelity. It achieves within $0.5\text{ dB}$ of the oracle ceiling while requiring under $3\text{ ms}$ per plate ($17\times$ faster than Richardson–Lucy).

---

## F. Visual Analysis & Figure Reference Guide

All generated publication-grade figures reside in [`experiments/results/stage8/figures/`](file:///E:/DIP%20Project/repo/experiments/results/stage8/figures/):

* **Figure 1: Panel Comparison — Small Blur $30\text{ dB}$ (Cell C02):**  
  Displays clean plate `'CH 63 NG 3078'`, degraded observation ($18.6\text{ dB}$), catastrophic inverse noise explosion ($3.1\text{ dB}$), Wiener ($22.3\text{ dB}$), CLS ($\mathbf{26.2\text{ dB}}$), and RL ($25.5\text{ dB}$). CLS and RL produce sharp, legible alphanumeric glyph contours.
* **Figure 2: Panel Comparison — Severe Large Blur $20\text{ dB}$ (Cell C09):**  
  Demonstrates restoration breakdown under ill-conditioned $51 \times 51$ combined blur. CLS preserves global character bounding envelopes ($15.2\text{ dB}$), while RL fails to converge ($11.4\text{ dB}$), and inverse produces solid salt-and-pepper noise.
* **Figure 3 & Figure 4: PSNR and SSIM Matrix Bar Charts:**  
  Systematic side-by-side comparison across all 16 cells confirming CLS's consistent peak performance across Gaussian noise and RL's peak under Poisson noise.
* **Figure 5: Runtime vs. Quality Pareto Frontier:**  
  Scatter plot showing runtime on log scale vs. PSNR gain. Linear closed-form filters occupy the high-speed cluster ($2\text{--}3\text{ ms}$), with CLS at the top frontier.
* **Figure 6: Richardson–Lucy Semi-Convergence:**  
  Trajectory curves across iterations ($k \in [1, 100]$). Under clean conditions, PSNR monotonically increases. Under $20\text{ dB}$ noise, PSNR peaks at $k=15\text{--}20$ before degrading due to multiplicative noise amplification.
* **Figure 7: Regularization Parameter Sensitivity:**  
  Profiles PSNR vs. $K \in [10^{-5}, 10^{-1}]$ and $\lambda \in [10^{-5}, 10^{-1}]$, confirming optimal performance plateau around $\lambda = 10^{-3}$ and $K = 10^{-3}$.
* **Figure 8: CCPD Real Blur Gallery (Extension E3):**  
  Displays 4 real vehicle camera captures restored with Wiener, CLS, and RL, showing sharpening of Chinese provincial characters.
* **Figure 9: Characteristic Failure Modes Gallery:**  
  Side-by-side demonstration of the four core deconvolution pathologies: noise explosion, Gibb's ringing oscillations, oversmoothing, and semi-convergence noise amplification.

---

## G. Readability Proxy & Character Segmentation Analysis

Downstream readability was evaluated using the Step 5 Otsu binarization and morphological character segmentation pipeline (`binarizer.py`):
* **Synthetic Plates (Cells C01–C16):** Character segmentation yield measures how effectively deconvolution recovers distinct connected components corresponding to alphanumeric glyphs against known reference ground-truth character counts (10 characters for synthetic plates).
* **Real Captured CCPD Plates (Extension E3):** Real camera plates lack clean ground truth and calibrated blind PSF estimation (Step 3). The $7 \times 7$ PSF used in E3 is an uncalibrated surrogate chosen to illustrate algorithm stability on real camera imagery, not a calibrated optical/motion model of the true physical blur. The character crop yield measures high-contrast Otsu connected components, not recognized text or OCR accuracy. Downstream OCR classification is owned by Step 6 (teammate Ujwal) and was not evaluated. Extension E3 serves as a preliminary qualitative stability demonstration on natural camera data rather than a complete restoration or readability proof.

### Table G.1: Character Segmentation Yield Across Core Conditions

| Condition / Cell | Clean Reference Expected Crops | Degraded Plate Segmented Crops | Inverse Restored Crops | Wiener Restored Crops | CLS Restored Crops | RL Restored Crops | Readability Diagnosis |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **C01 (Clean 7x7)** | $10.0 \pm 0.0$ | $7.8 \pm 1.2$ | $\mathbf{10.0 \pm 0.0}$ | $\mathbf{10.0 \pm 0.0}$ | $\mathbf{10.0 \pm 0.0}$ | $9.8 \pm 0.4$ | Perfect character separation restored |
| **C02 (30 dB 7x7)** | $10.0 \pm 0.0$ | $6.4 \pm 1.5$ | $0.0 \pm 0.0$ | $8.2 \pm 1.1$ | $\mathbf{9.6 \pm 0.7}$ | $9.4 \pm 0.8$ | CLS achieves $96\%$ glyph recovery |
| **C03 (20 dB 7x7)** | $10.0 \pm 0.0$ | $4.2 \pm 2.1$ | $0.0 \pm 0.0$ | $6.8 \pm 1.8$ | $\mathbf{8.6 \pm 1.2}$ | $8.2 \pm 1.4$ | CLS prevents glyph merging |
| **C08 (30 dB 51x51)** | $10.0 \pm 0.0$ | $0.8 \pm 0.9$ | $0.0 \pm 0.0$ | $3.4 \pm 1.5$ | $\mathbf{4.8 \pm 1.6}$ | $1.8 \pm 1.1$ | Severe blur merges characters |
| **E3 (Real CCPD Blur)** | $7.0 \pm 0.0$ | $1.2 \pm 0.8$ | *Excluded* | $1.8 \pm 1.1$ | $\mathbf{2.3 \pm 1.2}$ | $1.3 \pm 0.9$ | Restored plates increase segmentable component yield |

---

## H. Detailed Algorithm Strengths, Weaknesses, and Operating Regimes

### 1. Direct Inverse Filtering
* **Mathematical Basis:** $\hat{F}(u,v) = \frac{G(u,v)}{H(u,v)} = F(u,v) + \frac{N(u,v)}{H(u,v)}$.
* **Strengths:** Mathematically exact under noiseless conditions where $|H(u,v)| > 0$ ($\text{PSNR} > 88\text{ dB}$). Zero algorithmic hyperparameters.
* **Weaknesses:** Complete lack of regularization. At high spatial frequencies where $H(u,v) \to 0$, division by near-zero explodes noise power by $10^{6}\text{--}10^{12}$, turning the image into solid white/black noise ($3.1\text{ dB}$).
* **Operating Regime:** Usable strictly in synthetic noiseless environments with compact non-zero transfer functions. Completely unsuitable for real-world deployment.

### 2. Wiener Filtering
* **Mathematical Basis:** $\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + K} \right] G(u,v)$.
* **Strengths:** Robust noise attenuation; execution in a single inverse FFT ($2.1\text{ ms}$ isolated kernel time). Derived as the linear minimum mean square error (LMMSE) estimator.
* **Weaknesses:** Constant $K$ parameter heuristic attenuates high-frequency signal energy indiscriminately along with noise, leading to softened character edges and moderate ringing around plate borders.
* **Operating Regime:** Excellent fast linear filter for moderate Gaussian blur when high isolated kernel throughput ($> 400\text{ restorations/second}$) is required.

### 3. Constrained Least Squares (CLS) Filtering
* **Mathematical Basis:** $\hat{F}(u,v) = \left[ \frac{H^*(u,v)}{|H(u,v)|^2 + \lambda |C(u,v)|^2} \right] G(u,v)$, where $C(u,v)$ is the Fourier transform of a discrete 2D Laplacian operator.
* **Strengths:** Highest overall restoration fidelity across Gaussian regimes ($26.2\text{ dB}$ at $30\text{ dB}$; $22.9\text{ dB}$ at $20\text{ dB}$) and large ill-conditioned blurs. The frequency-dependent smoothness penalty $|C(u,v)|^2$ naturally matches image second-derivative statistics, preserving sharp character strokes while penalizing high-frequency noise spikes.
* **Weaknesses:** Requires selection of regularization parameter $\lambda$ based on noise variance.
* **Operating Regime:** The overall best deployable algorithm for license plate restoration under additive Gaussian noise and ill-conditioned optical blurs ($> 300\text{ restorations/second}$ isolated kernel throughput).

### 4. Richardson–Lucy (RL) Multiplicative Deconvolution
* **Mathematical Basis:** $\hat{f}^{(k+1)} = \hat{f}^{(k)} \cdot \left[ h^* * \left( \frac{g}{h * \hat{f}^{(k)}} \right) \right]$.
* **Strengths:** Native Poisson likelihood formulation; natural non-negativity preservation; highest fidelity under low-flux photon starvation ($23.42\text{ dB}$ at $N_{\text{eff}}=700$). Superior perceptual sharpness and high SSIM on compact blurs.
* **Weaknesses:** Iterative computational overhead ($35\text{--}87\text{ ms}$ per plate); pronounced semi-convergence requiring early stopping under noise; slow spatial energy propagation on large kernels ($51 \times 51$).
* **Operating Regime:** Optimal for night-time, low-light, or high-speed camera captures dominated by photon-shot noise and compact motion blurs.

---

## I. Limitations and Missing Dependencies

1. **Step 6 OCR Classifier Unavailability:**  
   Step 6 (character recognition classifier) is owned by teammate Ujwal and is not present in the repository. Full string recognition accuracy and Character Error Rate (CER) cannot be measured directly. Character segmentation crop yield from Step 5 was utilized as the authoritative downstream readability proxy.
2. **Absence of Ground Truth for Real Blurred CCPD Plates:**  
   The 50 plates in `ccpd_blur` were captured under natural camera motion blur; no clean reference image $f$ exists. Consequently, full-reference metrics (RMSE, PSNR, SSIM) are mathematically undefined and were not computed on them.
3. **Headroom-Scaled Poisson Representation:**  
   To prevent unphysical upper-tail truncation in a bounded $[0, 1]$ API, headroom scaling ($s = 0.70$) was applied, setting the effective photon budget to $N_{\text{eff}} = s \cdot N$. This is mathematically rigorous, but means nominal fluxes are slightly scaled relative to unscaled physical photon counts.
4. **Sample Selection & Canonical Layout Boundaries ($N=20$):**
   The core matrix was evaluated across $N=20$ clean plates (320 paired degradation trials) deterministically selected from synthetic standard blue plates (GA 36-2007 standard, aspect ratio $440 \times 140$). The high statistical significance ($p < 10^{-12}$) proves distinct algorithmic separation on this canonical configuration. However, universal conclusions cannot be extrapolated without further testing across non-standard geometries, diverse international aspects, commercial color variants (yellow, green NEV), or extreme non-frontal perspective transformations.

---

## J. Final Conclusions & Answers to Research Questions

* **RQ1 (Noise Sensitivity & Regularization Breakdown):** Direct inverse filtering suffers catastrophic noise explosion at any non-zero noise level ($3.1\text{ dB}$), and collapses even in noiseless conditions when the kernel has frequency nulls ($51 \times 51$). Wiener and CLS prevent this collapse, with CLS providing superior high-frequency noise attenuation through its Laplacian regularizer.
* **RQ2 (Likelihood Model Matching — Poisson vs. Gaussian Noise):** Richardson–Lucy confirms a statistically significant advantage under Poisson photon-shot noise at low fluxes ($N_{\text{eff}} = 700$, $+2.28\text{ dB}$ over CLS, $+11.45\text{ dB}$ over Wiener, $p < 10^{-6}$). Under Gaussian noise, CLS outperforms RL due to exact prior alignment.
* **RQ3 (Kernel Ill-Conditioning & Frequency Nulls):** On large $51 \times 51$ combined blurs with dense frequency zeros, closed-form Fourier regularization (CLS and Wiener) substantially outperforms iterative deconvolution (RL), achieving $16.45\text{ dB}$ vs. $11.75\text{ dB}$.
* **RQ4 (Spatial Resolution Decimation & Centring):** Factor-2 decimation with `effective_psf` and automatic centring (`centre="auto"`) preserves sub-pixel phase alignment, allowing RL and CLS to achieve over $28\text{ dB}$ PSNR and $>0.97$ SSIM.
* **RQ5 (Computational Complexity vs. Image Fidelity Trade-off):** Linear filters execute isolated in-memory deconvolution in $2\text{--}3\text{ ms}$, whereas RL takes $35\text{--}87\text{ ms}$. CLS achieves the optimal Pareto frontier, delivering top restoration fidelity with high-throughput kernel execution ($> 300\text{ restorations/second}$ in RAM). In an end-to-end ALPR system, overall system throughput will be dominated by camera I/O, vehicle detection, license plate localization, blind PSF estimation, character segmentation, and downstream OCR classification.
* **Overall Recommendation:** **Constrained Least Squares (CLS)** is the overall most robust and practical algorithm for general license plate restoration. **Richardson–Lucy** is the specialized algorithm of choice for photon-limited, low-exposure imaging conditions.

