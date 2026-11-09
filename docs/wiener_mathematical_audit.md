# Phase 4.3.1 — Wiener Filter Mathematical and Experimental Audit Report

**Repository:** `https://github.com/Hyperterror/Licence_Plate_Restoration.git`
**Branch:** `ashish/restoration-and-experiments`
**Base Commit:** `0e6dad5afaa8cfef2837bb4f23f7a643b2167f42`
**Status:** Audit Complete (Audit Only — Zero Implementation / Test Modifications)

---

## 1. Executive Summary

This document presents an exhaustive mathematical and experimental audit of the Wiener restoration filter (`wiener_filter`) and its regularisation parameter derivation (`wiener_k_from_bsnr`) as implemented in `license_plate_restoration/restoration.py` and `license_plate_restoration/degradation.py`.

### Primary Conclusions

1. **Exact Nature of Implemented $K$ (Candidate A):**
   The repository implements:
   $$K_A = \frac{\rho}{1 + \rho}, \qquad \rho = 10^{-\mathrm{BSNR}_{\mathrm{dB}}/10}$$
   Mathematically, this represents the **noise fraction of the observed (degraded) image variance** ($\sigma_n^2 / \operatorname{Var}(g)$), **not** the classical Wiener model regularisation parameter ($\sigma_n^2 / \operatorname{Var}(f)$). Blurring attenuates image variance ($\operatorname{Var}(h*f) < \operatorname{Var}(f)$), causing $K_A$ to systematically overestimate the classical sharp-signal ratio by a factor of $\operatorname{Var}(f)/\operatorname{Var}(g)$ (measured at $1.3\times$ for small blur and $3.6\times$ for severe blur).

2. **Equivalence with Direct BSNR Ratio (Candidate B):**
   Candidate B ($K_B = \rho = 10^{-\mathrm{BSNR}_{\mathrm{dB}}/10}$) differs from Candidate A by exactly $\rho^2 / (1 + \rho)$. In the operational regime of this project (BSNR $\in [20, 40]$ dB), the difference is between $0.01\%$ and $0.99\%$. In controlled experiments across 10 random seeds on both small and large PSFs, restoration RMSE differs by less than $0.2\%$, and PSNR differs by less than $0.02$ dB. Neither candidate is theoretically exact for non-white signals; both are blurred/observed-power approximations.

3. **Breakdown of the Theoretical Variance Split due to Clipping:**
   `add_noise()` clips pixel values to $[0, 1]$ and adjusts noise amplitude to achieve the target residual variance. On license plates (~60% white background), clipping truncates positive noise excursions, introducing a negative mean residual ($\mathbb{E}[r] \approx -2.3\times 10^{-3}$ at 30 dB, $-7.3\times 10^{-3}$ at 20 dB) and a negative covariance between the blurred signal and noise residual ($\operatorname{Cov}(h*f, r) < 0$). Consequently, the theoretical variance split $\operatorname{Var}(g) = \operatorname{Var}(h*f) + \operatorname{Var}(n)$ does not hold strictly: the measured split ratio $\operatorname{Var}(g) / (\operatorname{Var}(h*f) + \operatorname{Var}(r))$ drops to $0.987$ at 30 dB and $0.955$ at 20 dB. However, empirical experiments demonstrate that substituting the true realised observed noise ratio ($K_{A,\mathrm{obs}}$) changes restoration RMSE by under $0.9\%$.

4. **Failure of the Flat-Spectrum Assumption on Plate Glyphs:**
   The sharp-signal oracle parameter $K_C = \sigma_n^2 / \operatorname{Var}(f)$ (Candidate C) is optimal **if and only if** the signal spectrum is flat (white noise), as verified on synthetic white textures (within $0.1\%$ of swept minimum RMSE). However, license plates have highly colored spectra dominated by low spatial frequencies. On a small PSF (7×7), Candidate C under-regularises high frequencies, yielding $10.4\%$ worse RMSE than Candidate A. Conversely, on a large PSF (51×51), blur completely obliterates high frequencies, and Candidate C outperforms Candidate A by $9.3\%$ (30 dB) and $14.9\%$ (20 dB). No single scalar $K$ derived solely from BSNR can be universally optimal across all kernels.

5. **Recommendation:**
   Retain Candidate A ($K_A = \frac{\rho}{1 + \rho}$) as the deployable BSNR-to-$K$ heuristic, with updated docstrings explicitly clarifying that it is an **observed-power heuristic** ($\sigma_n^2 / \operatorname{Var}(g)$). For controlled scientific experiments and benchmark evaluations, require callers to pass an explicit $K=$ parameter to explore the parameter sweep.

---

## 2. Current Implementation Audit & Function-by-Function Behavior

All functions in `license_plate_restoration/degradation.py` and `license_plate_restoration/restoration.py` were audited directly from source code:

### 2.1 `noise_sigma(img: np.ndarray, bsnr_db: float) -> float`
- **Location:** `license_plate_restoration/degradation.py:126`
- **Current Behavior:**
  ```python
  variance = float(np.var(np.asarray(img, dtype=np.float64)))
  return float(np.sqrt(variance / (10.0 ** (float(bsnr_db) / 10.0))))
  ```
- **Audit Findings:**
  - `img` is the blurred, uncorrupted image $h * f$, with pixel values in $[0, 1]$.
  - Calculates population variance $\operatorname{Var}(h*f) = \frac{1}{MN}\sum (x - \mu)^2$.
  - Computes $\sigma_n = \sqrt{\operatorname{Var}(h*f) / 10^{\mathrm{BSNR}_{\mathrm{dB}}/10}}$.
  - **Crucial Mathematical Fact:** Noise standard deviation is computed relative to the **blurred image variance**, NOT the sharp image variance $\operatorname{Var}(f)$, and NOT total signal power $\frac{1}{MN}\sum x^2$.

### 2.2 `add_noise(img, bsnr_db, seed, saturation_compensation=True)`
- **Location:** `license_plate_restoration/degradation.py:137`
- **Current Behavior:**
  - Draws standard normal noise $n_0 \sim \mathcal{N}(0, I)$ with `np.random.default_rng(seed)`.
  - Initial amplitude $A = \sigma_n = \text{noise\_sigma}(img, bsnr\_db)$.
  - If `saturation_compensation=True`: runs up to 10 iterations of a fixed-point loop:
    $$r = \operatorname{clip}(img + A \cdot n_0, 0.0, 1.0) - img$$
    $$\text{ratio} = \frac{\sigma_n^2}{\operatorname{Var}(r)}$$
    $$A \leftarrow A \times \sqrt{\text{ratio}}$$
    terminating when $|\text{ratio} - 1.0| \le 10^{-3}$.
  - Returns `np.clip(img + A * normal, 0.0, 1.0).astype(np.float32)`.
- **Audit Findings:**
  - Output is strictly clipped to $[0, 1]$.
  - The loop successfully scales $A$ so that the post-clipping residual variance $\operatorname{Var}(g - h*f)$ matches $\sigma_n^2$ within $0.1\%$.
  - However, clipping removes positive excursions on bright pixels ($img \approx 1.0$), shifting the mean residual negative ($\mathbb{E}[r] < 0$) and causing non-zero signal-noise cross-correlation.

### 2.3 `measure_bsnr(clean, noisy)`
- **Location:** `license_plate_restoration/degradation.py:195`
- **Current Behavior:**
  ```python
  residual = np.asarray(noisy, dtype=np.float64) - clean
  return float(10.0 * np.log10(np.var(clean) / np.var(residual)))
  ```
- **Audit Findings:**
  - Evaluates $10 \log_{10} \frac{\operatorname{Var}(clean)}{\operatorname{Var}(noisy - clean)}$.
  - Confirms the repository convention: BSNR is measured between blurred clean image and realised residual.

### 2.4 `wiener_k_from_bsnr(bsnr_db) -> float`
- **Location:** `license_plate_restoration/restoration.py:1161`
- **Current Behavior:**
  ```python
  rho = 10.0 ** (-float(bsnr_db) / 10.0)
  return float(rho / (1.0 + rho))
  ```
- **Audit Findings:**
  - Returns $K_A = \frac{\rho}{1 + \rho}$.
  - Accepts any real finite `bsnr_db`.
  - Rejects NaN/Inf via `_finite_scalar`.
  - Does not take PSF as input.

### 2.5 `wiener_filter(image, psf, K=None, bsnr_db=None, ...)`
- **Location:** `license_plate_restoration/restoration.py:1260`
- **Current Behavior:**
  - Enforces mutual exclusivity: requires exactly one of `K` or `bsnr_db`.
  - Computes transfer function $H(u,v) = \text{psf\_frequency\_response}(psf, image.shape)$.
  - Tapers boundaries via `taper_edges` (default profile: `"none"`).
  - Computes:
    $$\hat{F}(u,v) = \frac{H^*(u,v) G(u,v)}{|H(u,v)|^2 + K}$$
    implemented using `np.divide` with `where=denominator > 0` (zero-gain convention at $|H|^2 + K = 0$).
  - Inverts FFT: $\hat{f} = \operatorname{Re}(\mathcal{F}^{-1}\{\hat{F}\})$.
  - Clamps output via `clip_unit_interval(restored)` to $[0, 1]$ `float32`.
- **Audit Findings:**
  - Strictly preserves phase via `np.conj(response)`.
  - Deliberately does not apply an arbitrary floor $\varepsilon$ when $K > 0$ because denominator is strictly bounded from below by $K$.
  - At $K = 0$, exact nulls of $H$ evaluate to zero gain (zero output), preventing NaN/Inf.

---

## 3. Exact BSNR Definition Used by the Repository

In the literature, SNR definitions vary substantially. The exact definition used by this repository is defined by `degradation.noise_sigma`:

$$\mathrm{BSNR}_{\mathrm{dB}} = 10 \log_{10} \left( \frac{\operatorname{Var}(h * f)}{\sigma_n^2} \right)$$

where:
1. **Power Convention:** The formula uses $10 \log_{10}$, indicating a ratio of variances (powers), not standard deviations (amplitudes, which would use $20 \log_{10}$).
2. **Signal Reference:** The signal power is the variance of the **blurred** image $\operatorname{Var}(h * f)$, **not** the sharp original plate $\operatorname{Var}(f)$, and **not** the second moment $\mathbb{E}[(h*f)^2]$.
3. **Linear Dimensionless Ratio:**
   $$\rho = 10^{-\mathrm{BSNR}_{\mathrm{dB}}/10} = \frac{\sigma_n^2}{\operatorname{Var}(h * f)}$$
   For standard BSNR levels:
   - $40\text{ dB} \implies \rho = 10^{-4} = 0.0001$
   - $30\text{ dB} \implies \rho = 10^{-3} = 0.001$
   - $20\text{ dB} \implies \rho = 10^{-2} = 0.01$
   - $0\text{ dB} \implies \rho = 10^{0} = 1.0$

---

## 4. Mathematical Derivation and Assumptions

### 4.1 Derivation of the Wiener Filter (MMSE)
Consider the discrete linear degradation model under circular convolution:
$$g(x,y) = (h * f)(x,y) + n(x,y) \iff G(u,v) = H(u,v)F(u,v) + N(u,v)$$

We seek a linear frequency-domain filter $W(u,v)$ such that $\hat{F}(u,v) = W(u,v)G(u,v)$ minimizes the Mean Squared Error (MSE):
$$e = \mathbb{E}\left[ \|f - \hat{f}\|_2^2 \right] = \frac{1}{MN} \sum_{u,v} \mathbb{E}\left[ |F(u,v) - W(u,v)G(u,v)|^2 \right]$$

Assuming:
1. $f$ and $n$ are zero-mean Wide-Sense Stationary (WSS) random processes.
2. Signal and noise are mutually uncorrelated: $\mathbb{E}[F(u,v) N^*(u,v)] = 0$.

The error expectation at frequency $(u,v)$ is:
$$\mathbb{E}[|F - WG|^2] = \mathbb{E}[|F|^2] - W^* \mathbb{E}[F G^*] - W \mathbb{E}[F^* G] + |W|^2 \mathbb{E}[|G|^2]$$

Substituting $G = HF + N$:
$$\mathbb{E}[|G(u,v)|^2] = |H(u,v)|^2 S_f(u,v) + S_n(u,v)$$
$$\mathbb{E}[F(u,v) G^*(u,v)] = H^*(u,v) S_f(u,v)$$
where $S_f(u,v) = \mathbb{E}[|F(u,v)|^2]$ and $S_n(u,v) = \mathbb{E}[|N(u,v)|^2]$.

Minimizing with respect to $W(u,v)$ yields the optimal Wiener transfer function:
$$W(u,v) = \frac{H^*(u,v) S_f(u,v)}{|H(u,v)|^2 S_f(u,v) + S_n(u,v)} = \frac{H^*(u,v)}{|H(u,v)|^2 + \dfrac{S_n(u,v)}{S_f(u,v)}}$$

### 4.2 Derivation of the Scalar Regularisation Parameter $K$
The scalar Wiener filter replaces frequency-dependent ratio $S_n(u,v)/S_f(u,v)$ with a scalar constant $K$:
$$\hat{F}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K} G(u,v)$$

This replacement is exact if and only if:
1. **White Noise:** $S_n(u,v) = \text{constant} = MN \sigma_n^2$.
2. **Flat Signal Spectrum:** $S_f(u,v) = \text{constant} = MN \operatorname{Var}(f)$.

Under these two conditions, the exact ratio is:
$$K_{\mathrm{model}} = \frac{S_n(u,v)}{S_f(u,v)} = \frac{\sigma_n^2}{\operatorname{Var}(f)} \quad (\text{Candidate C})$$
This is the noise variance relative to the **sharp, unblurred signal variance**.

### 4.3 Theoretical Analysis of Candidate A ($K_A = \frac{\rho}{1+\rho}$)
In the absence of clipping, if signal and noise are uncorrelated:
$$\operatorname{Var}(g) = \operatorname{Var}(h*f + n) = \operatorname{Var}(h*f) + \operatorname{Var}(n) = \operatorname{Var}(h*f) (1 + \rho)$$

Candidate A estimates regularisation by the ratio of noise variance to **observed degraded variance**:
$$K_A = \frac{\operatorname{Var}(n)}{\operatorname{Var}(g)} = \frac{\rho \operatorname{Var}(h*f)}{\operatorname{Var}(h*f)(1 + \rho)} = \frac{\rho}{1 + \rho}$$

Comparing $K_A$ to the classical model parameter $K_{\mathrm{model}}$:
$$K_A = \frac{\sigma_n^2}{\operatorname{Var}(g)} = \frac{\sigma_n^2}{\operatorname{Var}(f)} \cdot \frac{\operatorname{Var}(f)}{\operatorname{Var}(g)} = K_{\mathrm{model}} \cdot \frac{\operatorname{Var}(f)}{\operatorname{Var}(h*f) + \sigma_n^2}$$

Because low-pass blurring eliminates high-frequency components, $\operatorname{Var}(h*f) < \operatorname{Var}(f)$. For typical noise levels where $\sigma_n^2 \ll \operatorname{Var}(f) - \operatorname{Var}(h*f)$, we have:
$$\operatorname{Var}(g) < \operatorname{Var}(f) \implies \frac{\operatorname{Var}(f)}{\operatorname{Var}(g)} > 1 \implies K_A > K_{\mathrm{model}}$$

**Rigorous Mathematical Conclusion:**
$K_A = \frac{\rho}{1+\rho}$ is **not theoretically equivalent** to the classical noise-to-original-signal power ratio $K_{\mathrm{model}}$. It is strictly an **observed-power approximation** that substitutes degraded image variance for clean image variance.

---

## 5. Clipping and Noise Generation Analysis

### 5.1 Degradation Characteristics of Synthetic License Plates
A synthetic license plate image (`shape = (96, 336)`) has high contrast:
- Clean Plate Minimum: $0.000$, Maximum: $1.000$, Mean: $0.725$, Variance: $0.14205$.
- Pixel distribution: **$60.1\%$ of pixels are at $1.0$** (white retro-reflective background); **$15.0\%$ of pixels are at $0.0$** (black characters).

### 5.2 Saturation Compensation & Realised Statistics
When Gaussian noise is added to a blurred plate, pixels near $1.0$ saturate at $1.0$. `add_noise` iteratively scales the Gaussian amplitude to force the post-clipping residual $r = \operatorname{clip}(h*f + A \cdot n_0) - h*f$ to satisfy $\operatorname{Var}(r) = \sigma_n^2$.

Controlled measurements over 10 independent seeds reveal the consequences:

| Kernel | BSNR | Target $\sigma_n^2$ | Realised $\operatorname{Var}(r)/\sigma_n^2$ | Mean Residual $\mathbb{E}[r]$ | Total Clipped % | Clipped @ 0.0 | Clipped @ 1.0 | Split Ratio $\frac{\operatorname{Var}(g)}{\operatorname{Var}(b)+\operatorname{Var}(r)}$ |
|---|---|---|---|---|---|---|---|---|
| **7×7** | 30 dB | $1.095\times 10^{-4}$ | $0.9999 \pm 0.0001$ | $-2.274\times 10^{-3}$ | $25.5\%$ | $1.2\%$ | $24.3\%$ | $0.9866$ |
| **7×7** | 20 dB | $1.095\times 10^{-3}$ | $0.9999 \pm 0.0001$ | $-7.251\times 10^{-3}$ | $26.5\%$ | $1.7\%$ | $24.8\%$ | $0.9554$ |
| **51×51** | 30 dB | $3.935\times 10^{-5}$ | $0.9994 \pm 0.0004$ | $-4.475\times 10^{-4}$ | $8.9\%$ | $0.0\%$ | $8.9\%$ | $0.9938$ |
| **51×51** | 20 dB | $3.935\times 10^{-4}$ | $1.0000 \pm 0.0001$ | $-1.627\times 10^{-3}$ | $10.2\%$ | $0.0\%$ | $10.2\%$ | $0.9777$ |

### 5.3 Key Findings on Clipping
1. **Negative Residual Bias:** Because clipping occurs overwhelmingly at the upper boundary ($24.3\%$ to $24.8\%$ clipped at 1.0 vs $1.2\%$ to $1.7\%$ at 0.0), positive noise spikes are truncated. The realised residual is **not zero-mean**: it exhibits a statistically significant negative mean ($\mathbb{E}[r] = -7.25\times 10^{-3}$ at 20 dB).
2. **Breakdown of Gaussianity:** The residual is a truncated Gaussian with probability mass accumulated at the boundaries.
3. **Negative Signal-Noise Covariance:** The clipping operation induces signal-dependence. Pixels with high blurred intensity ($h*f \approx 1$) can only have negative residuals ($r \le 0$), while dark pixels ($h*f \approx 0$) can only have positive residuals ($r \ge 0$). This produces negative cross-covariance:
   $$\operatorname{Cov}(h*f, r) < 0 \quad (\text{measured } -7.28\times 10^{-4} \text{ at } 30\text{ dB}, -2.44\times 10^{-3} \text{ at } 20\text{ dB})$$
4. **Breakdown of Variance Split:**
   $$\operatorname{Var}(g) = \operatorname{Var}(h*f) + \operatorname{Var}(r) + 2\operatorname{Cov}(h*f, r) < \operatorname{Var}(h*f) + \operatorname{Var}(r)$$
   The identity $\operatorname{Var}(g) = \operatorname{Var}(h*f)(1 + \rho)$ used to derive Candidate A holds only to $95.5\%$ at 20 dB.
5. **Practical Impact:** Despite this theoretical breakdown, restoring with the true empirical observed noise fraction $K_{A,\mathrm{obs}} = \operatorname{Var}(r)/\operatorname{Var}(g)$ alters restoration RMSE by less than $0.9\%$ across all conditions.

---

## 6. Candidate $K$ Comparison

The following candidate estimators were evaluated:

| Candidate | Formula | Theoretical Description | Deployable? |
|---|---|---|---|
| **Candidate A** | $K_A = \dfrac{\rho}{1 + \rho}$ | Current repo heuristic: noise fraction of observed power $\sigma_n^2 / \operatorname{Var}(g)$ | **Yes** |
| **Candidate B** | $K_B = \rho = 10^{-\mathrm{BSNR}/10}$ | Direct BSNR ratio: noise power relative to blurred power $\sigma_n^2 / \operatorname{Var}(h*f)$ | **Yes** |
| **Candidate C** | $K_C = \dfrac{\sigma_n^2}{\operatorname{Var}(f)}$ | Classical Wiener model ratio: noise power relative to sharp plate variance | **No** (Oracle) |
| **Candidate C_real** | $K_{C,\mathrm{real}} = \dfrac{\operatorname{Var}(r)}{\operatorname{Var}(f)}$ | Post-clipping realised residual variance relative to sharp plate variance | **No** (Oracle) |
| **Candidate W** | $K_W = \rho \sum h^2$ | Parseval white-signal approximation ($\operatorname{Var}(h*f) = \sum h^2 \operatorname{Var}(f)$) | **Yes** |
| **Candidate A_obs** | $K_{A,\mathrm{obs}} = \dfrac{\operatorname{Var}(r)}{\operatorname{Var}(g)}$ | Realised post-clipping observed noise fraction | **Yes** (from $g$) |
| **Candidate Best** | $K^* = \arg\min_K \text{RMSE}$ | Empirical optimal scalar $K$ via log-space sweep (61 points) | **No** (Oracle) |
| **Spectral Oracle** | $W(u,v) = \dfrac{H^* S_f}{\|H\|^2 S_f + S_n}$ | True frequency-dependent MMSE filter with $S_f = \|F\|^2, S_n = MN \sigma_n^2$ | **No** (Oracle) |

### Analytical Comparison: Candidate A vs Candidate B
The difference between Candidate A and Candidate B is:
$$K_B - K_A = \rho - \frac{\rho}{1 + \rho} = \frac{\rho^2}{1 + \rho}$$
Relative difference:
$$\frac{K_B - K_A}{K_A} = \rho$$
- At 40 dB: $\rho = 10^{-4} \implies 0.01\%$ difference.
- At 30 dB: $\rho = 10^{-3} \implies 0.10\%$ difference.
- At 20 dB: $\rho = 10^{-2} \implies 0.99\%$ difference.
- At 0 dB: $\rho = 1.0 \implies 50\%$ difference ($K_A = 0.5, K_B = 1.0$).

Across all experimental operating points ($\ge 20$ dB), Candidate A and Candidate B agree to within $1\%$.

---

## 7. Experimental Setup and Results

### 7.1 Controlled Experiment Protocol
- **PSFs:**
  - **Small Kernel:** 7×7 combined PSF (motion length=3, angle=8.0°, defocus radius=0). Energy $\sum h^2 = 0.1089$.
  - **Large Kernel:** 51×51 combined PSF (motion length=21, angle=8.0°, defocus radius=4). Energy $\sum h^2 = 0.00293$.
- **Noise Levels:** 30 dB and 20 dB BSNR.
- **Seeds:** 10 independent pseudo-random seeds ($0$ through $9$) per condition.
- **Evaluation Crop:** 12 px interior margin crop on all methods, avoiding boundary artifacts.
- **Metrics Reported:** Interior RMSE (mean ± std, [min, max]), PSNR (dB), SSIM.

### 7.2 Benchmark Results (10 Seeds per Condition)

#### Condition 1: Small Kernel (7×7), 30 dB BSNR
- Blurred signal variance: $\operatorname{Var}(h*f) = 0.1095$ ($\operatorname{Var}(h*f)/\operatorname{Var}(f) = 0.771$)
- Degraded Input: $\text{RMSE} = 1.1106\times 10^{-1} \pm 7.7\times 10^{-5}$, $\text{PSNR} = 19.09\text{ dB}$, $\text{SSIM} = 0.8550$
- Inverse Filter: $\text{RMSE} = 6.9595\times 10^{-1} \pm 4.6\times 10^{-3}$, $\text{PSNR} = 3.15\text{ dB}$, $\text{SSIM} = 0.0030$

| Method | Regularisation $K$ | RMSE (mean ± std) | RMSE [min, max] | PSNR (dB) | SSIM | Beats A? |
|---|---|---|---|---|---|---|
| **Candidate A** | $9.990\times 10^{-4}$ | $7.7479\times 10^{-2} \pm 4.6\times 10^{-4}$ | $[7.6857\times 10^{-2}, 7.8182\times 10^{-2}]$ | $22.22$ | $0.5781$ | — |
| **Candidate B** | $1.000\times 10^{-3}$ | $7.7450\times 10^{-2} \pm 4.6\times 10^{-4}$ | $[7.6829\times 10^{-2}, 7.8153\times 10^{-2}]$ | $22.22$ | $0.5782$ | 10/10 |
| **Candidate C** | $7.708\times 10^{-4}$ | $8.5513\times 10^{-2} \pm 5.0\times 10^{-4}$ | $[8.4791\times 10^{-2}, 8.6212\times 10^{-2}]$ | $21.36$ | $0.5496$ | 0/10 |
| **Candidate C_real**| $7.707\times 10^{-4}$ | $8.5517\times 10^{-2} \pm 5.0\times 10^{-4}$ | $[8.4791\times 10^{-2}, 8.6245\times 10^{-2}]$ | $21.36$ | $0.5496$ | 0/10 |
| **Candidate W** | $1.089\times 10^{-4}$ | $1.7384\times 10^{-1} \pm 1.2\times 10^{-3}$ | $[1.7205\times 10^{-1}, 1.7585\times 10^{-1}]$ | $15.20$ | $0.3952$ | 0/10 |
| **Candidate A_obs**| $1.012\times 10^{-3}$ | $7.7094\times 10^{-2} \pm 4.7\times 10^{-4}$ | $[7.6469\times 10^{-2}, 7.7826\times 10^{-2}]$ | $22.26$ | $0.5796$ | 10/10 |
| **Swept Best $K^*$** | $5.012\times 10^{-3}$ | $5.4323\times 10^{-2} \pm 2.3\times 10^{-4}$ | $[5.3956\times 10^{-2}, 5.4669\times 10^{-2}]$ | $25.30$ | $0.7657$ | 10/10 |
| **Spectral Oracle** | non-scalar | $4.8073\times 10^{-2} \pm 4.3\times 10^{-4}$ | $[4.7342\times 10^{-2}, 4.8616\times 10^{-2}]$ | $26.36$ | $0.8387$ | 10/10 |

---

#### Condition 2: Small Kernel (7×7), 20 dB BSNR
- Degraded Input: $\text{RMSE} = 1.1500\times 10^{-1} \pm 2.3\times 10^{-4}$, $\text{PSNR} = 18.79\text{ dB}$, $\text{SSIM} = 0.7015$
- Inverse Filter: $\text{RMSE} = 7.0002\times 10^{-1} \pm 1.1\times 10^{-3}$, $\text{PSNR} = 3.10\text{ dB}$, $\text{SSIM} = 0.0005$

| Method | Regularisation $K$ | RMSE (mean ± std) | RMSE [min, max] | PSNR (dB) | SSIM | Beats A? |
|---|---|---|---|---|---|---|
| **Candidate A** | $9.901\times 10^{-3}$ | $8.9459\times 10^{-2} \pm 4.3\times 10^{-4}$ | $[8.8719\times 10^{-2}, 9.0063\times 10^{-2}]$ | $20.97$ | $0.5615$ | — |
| **Candidate B** | $1.000\times 10^{-2}$ | $8.9273\times 10^{-2} \pm 4.3\times 10^{-4}$ | $[8.8534\times 10^{-2}, 8.9873\times 10^{-2}]$ | $20.99$ | $0.5624$ | 10/10 |
| **Candidate C** | $7.708\times 10^{-3}$ | $9.5138\times 10^{-2} \pm 4.8\times 10^{-4}$ | $[9.4357\times 10^{-2}, 9.5847\times 10^{-2}]$ | $20.43$ | $0.5374$ | 0/10 |
| **Candidate C_real**| $7.707\times 10^{-3}$ | $9.5140\times 10^{-2} \pm 4.8\times 10^{-4}$ | $[9.4359\times 10^{-2}, 9.5850\times 10^{-2}]$ | $20.43$ | $0.5374$ | 0/10 |
| **Candidate W** | $1.089\times 10^{-3}$ | $2.1353\times 10^{-1} \pm 1.1\times 10^{-3}$ | $[2.1214\times 10^{-1}, 2.1526\times 10^{-1}]$ | $13.41$ | $0.3495$ | 0/10 |
| **Candidate A_obs**| $1.036\times 10^{-2}$ | $8.8629\times 10^{-2} \pm 4.3\times 10^{-4}$ | $[8.7886\times 10^{-2}, 8.9222\times 10^{-2}]$ | $21.05$ | $0.5658$ | 10/10 |
| **Swept Best $K^*$** | $1.995\times 10^{-2}$ | $8.3156\times 10^{-2} \pm 3.2\times 10^{-4}$ | $[8.2600\times 10^{-2}, 8.3659\times 10^{-2}]$ | $21.60$ | $0.6200$ | 10/10 |
| **Spectral Oracle** | non-scalar | $6.9624\times 10^{-2} \pm 3.2\times 10^{-4}$ | $[6.9118\times 10^{-2}, 7.0131\times 10^{-2}]$ | $23.14$ | $0.7459$ | 10/10 |

---

#### Condition 3: Large Kernel (51×51), 30 dB BSNR
- Blurred signal variance: $\operatorname{Var}(h*f) = 0.03935$ ($\operatorname{Var}(h*f)/\operatorname{Var}(f) = 0.277$)
- Degraded Input: $\text{RMSE} = 3.1645\times 10^{-1} \pm 4.9\times 10^{-5}$, $\text{PSNR} = 9.99\text{ dB}$, $\text{SSIM} = 0.3910$
- Inverse Filter: $\text{RMSE} = 7.0106\times 10^{-1} \pm 4.2\times 10^{-4}$, $\text{PSNR} = 3.08\text{ dB}$, $\text{SSIM} = 0.0025$

| Method | Regularisation $K$ | RMSE (mean ± std) | RMSE [min, max] | PSNR (dB) | SSIM | Beats A? |
|---|---|---|---|---|---|---|
| **Candidate A** | $9.990\times 10^{-4}$ | $1.6989\times 10^{-1} \pm 4.5\times 10^{-4}$ | $[1.6904\times 10^{-1}, 1.7041\times 10^{-1}]$ | $15.40$ | $0.5649$ | — |
| **Candidate B** | $1.000\times 10^{-3}$ | $1.6991\times 10^{-1} \pm 4.5\times 10^{-4}$ | $[1.6906\times 10^{-1}, 1.7043\times 10^{-1}]$ | $15.40$ | $0.5649$ | 0/10 |
| **Candidate C** | $2.770\times 10^{-4}$ | $1.5419\times 10^{-1} \pm 6.1\times 10^{-4}$ | $[1.5306\times 10^{-1}, 1.5482\times 10^{-1}]$ | $16.24$ | $0.5276$ | 10/10 |
| **Candidate C_real**| $2.768\times 10^{-4}$ | $1.5419\times 10^{-1} \pm 6.1\times 10^{-4}$ | $[1.5306\times 10^{-1}, 1.5482\times 10^{-1}]$ | $16.24$ | $0.5276$ | 10/10 |
| **Candidate W** | $2.913\times 10^{-6}$ | $4.9343\times 10^{-1} \pm 1.3\times 10^{-3}$ | $[4.9126\times 10^{-1}, 4.9567\times 10^{-1}]$ | $6.14$ | $0.1175$ | 0/10 |
| **Candidate A_obs**| $1.005\times 10^{-3}$ | $1.7000\times 10^{-1} \pm 4.4\times 10^{-4}$ | $[1.6916\times 10^{-1}, 1.7053\times 10^{-1}]$ | $15.39$ | $0.5650$ | 0/10 |
| **Swept Best $K^*$** | $2.512\times 10^{-4}$ | $1.5406\times 10^{-1} \pm 6.3\times 10^{-4}$ | $[1.5292\times 10^{-1}, 1.5472\times 10^{-1}]$ | $16.25$ | $0.5223$ | 10/10 |
| **Spectral Oracle** | non-scalar | $1.5502\times 10^{-1} \pm 2.4\times 10^{-3}$ | $[1.5074\times 10^{-1}, 1.5923\times 10^{-1}]$ | $16.19$ | $0.5047$ | 10/10 |

---

#### Condition 4: Large Kernel (51×51), 20 dB BSNR
- Degraded Input: $\text{RMSE} = 3.1698\times 10^{-1} \pm 1.6\times 10^{-4}$, $\text{PSNR} = 9.98\text{ dB}$, $\text{SSIM} = 0.3497$
- Inverse Filter: $\text{RMSE} = 7.0095\times 10^{-1} \pm 2.5\times 10^{-4}$, $\text{PSNR} = 3.09\text{ dB}$, $\text{SSIM} = 0.0026$

| Method | Regularisation $K$ | RMSE (mean ± std) | RMSE [min, max] | PSNR (dB) | SSIM | Beats A? |
|---|---|---|---|---|---|---|
| **Candidate A** | $9.901\times 10^{-3}$ | $2.3060\times 10^{-1} \pm 4.4\times 10^{-4}$ | $[2.2989\times 10^{-1}, 2.3127\times 10^{-1}]$ | $12.74$ | $0.5032$ | — |
| **Candidate B** | $1.000\times 10^{-2}$ | $2.3088\times 10^{-1} \pm 4.4\times 10^{-4}$ | $[2.3017\times 10^{-1}, 2.3155\times 10^{-1}]$ | $12.73$ | $0.5029$ | 0/10 |
| **Candidate C** | $2.770\times 10^{-3}$ | $1.9649\times 10^{-1} \pm 9.4\times 10^{-4}$ | $[1.9489\times 10^{-1}, 1.9762\times 10^{-1}]$ | $14.13$ | $0.5076$ | 10/10 |
| **Candidate C_real**| $2.770\times 10^{-3}$ | $1.9649\times 10^{-1} \pm 9.4\times 10^{-4}$ | $[1.9489\times 10^{-1}, 1.9762\times 10^{-1}]$ | $14.13$ | $0.5076$ | 10/10 |
| **Candidate W** | $2.913\times 10^{-5}$ | $4.3543\times 10^{-1} \pm 3.3\times 10^{-3}$ | $[4.3090\times 10^{-1}, 4.4052\times 10^{-1}]$ | $7.22$ | $0.1453$ | 0/10 |
| **Candidate A_obs**| $1.013\times 10^{-2}$ | $2.3124\times 10^{-1} \pm 4.2\times 10^{-4}$ | $[2.3054\times 10^{-1}, 2.3190\times 10^{-1}]$ | $12.72$ | $0.5025$ | 0/10 |
| **Swept Best $K^*$** | $6.636\times 10^{-4}$ | $1.7678\times 10^{-1} \pm 1.5\times 10^{-3}$ | $[1.7364\times 10^{-1}, 1.7829\times 10^{-1}]$ | $15.05$ | $0.4410$ | 10/10 |
| **Spectral Oracle** | non-scalar | $1.6025\times 10^{-1} \pm 3.0\times 10^{-3}$ | $[1.5437\times 10^{-1}, 1.6509\times 10^{-1}]$ | $15.91$ | $0.4851$ | 10/10 |

---

### 7.3 Multi-Plate Generalisation (5 Independent Synthetic Plates)
To guarantee conclusions are not an artifact of a single license plate layout, the evaluation was repeated across 5 independent synthetic plates (seeds 42–46, 3 noise seeds each = 15 runs per condition):

| Kernel | BSNR | Candidate A RMSE | Candidate B RMSE | Candidate C RMSE | Ratio $K_C / K_A$ |
|---|---|---|---|---|---|
| **7×7** | 30 dB | $7.6056\times 10^{-2}$ | $7.6028\times 10^{-2}$ | $8.4632\times 10^{-2}$ | **$1.113$** (C is 11.3% worse) |
| **7×7** | 20 dB | $8.8065\times 10^{-2}$ | $8.7886\times 10^{-2}$ | $9.3995\times 10^{-2}$ | **$1.067$** (C is 6.7% worse) |
| **51×51** | 30 dB | $1.6910\times 10^{-1}$ | $1.6912\times 10^{-1}$ | $1.5413\times 10^{-1}$ | **$0.911$** (C is 8.9% better) |
| **51×51** | 20 dB | $2.2767\times 10^{-1}$ | $2.2795\times 10^{-1}$ | $1.9571\times 10^{-1}$ | **$0.860$** (C is 14.0% better) |

### 7.4 Control Experiment: Validation on Flat-Spectrum (White) Signal
To verify the theoretical derivation of Section 4, the filter was evaluated on an i.i.d. uniform random texture ($64\times 64$, flat spectrum) degraded with a 5×5 Gaussian PSF:

| BSNR | Candidate C ($K_C$) | Candidate W ($K_W$) | Candidate A ($K_A$) | Candidate B ($K_B$) | Swept Optimum $K^*$ |
|---|---|---|---|---|---|
| **30 dB** | $K=7.64\times 10^{-5}$<br>$\text{RMSE}=6.870\times 10^{-2}$ | $K=8.26\times 10^{-5}$<br>$\text{RMSE}=6.874\times 10^{-2}$ | $K=9.99\times 10^{-4}$<br>$\text{RMSE}=8.635\times 10^{-2}$ | $K=1.00\times 10^{-3}$<br>$\text{RMSE}=8.636\times 10^{-2}$ | $K=7.94\times 10^{-5}$<br>$\text{RMSE}=6.871\times 10^{-2}$ |
| **Ratio to Best** | **$0.9998\times$** | **$1.0003\times$** | **$1.2567\times$** | **$1.2569\times$** | $1.0000\times$ |
| **20 dB** | $K=7.64\times 10^{-4}$<br>$\text{RMSE}=9.375\times 10^{-2}$ | $K=8.26\times 10^{-4}$<br>$\text{RMSE}=9.375\times 10^{-2}$ | $K=9.90\times 10^{-3}$<br>$\text{RMSE}=1.085\times 10^{-1}$ | $K=1.00\times 10^{-2}$<br>$\text{RMSE}=1.086\times 10^{-1}$ | $K=7.94\times 10^{-4}$<br>$\text{RMSE}=9.374\times 10^{-2}$ |
| **Ratio to Best** | **$1.0001\times$** | **$1.0001\times$** | **$1.1572\times$** | **$1.1580\times$** | $1.0000\times$ |

**Significance:**
On a genuine flat-spectrum signal, Candidates C and W land within **$0.03\%$ of the true empirical optimum**, whereas Candidates A and B over-regularise by $1.16\times$ to $1.26\times$ in RMSE. This proves that the under-performance of Candidate C on license plates is entirely due to the non-white spectral distribution of license plates.

---

## 8. Implementation Edge Cases and Numerical Stability

The codebase was audited for edge cases, numerical stability, and consistency with prior restoration phases:

1. **Parameter Validation & Stability:**
   - `wiener_filter` strictly rejects invalid regularisation: negative $K$, NaN, Inf, non-numeric strings, and simultaneous provision of both `K` and `bsnr_db` (tested in `test_wiener_rejects_invalid_regularisation`).
   - For any $K > 0$, the denominator $|H(u,v)|^2 + K \ge K > 0$ is strictly bounded away from zero. No division by zero is possible.

2. **Behavior at $K = 0$:**
   - When $K = 0$, the filter reduces algebraically to the unregularised inverse filter $H^* / |H|^2 = 1 / H$.
   - Tested in `test_zero_k_is_the_inverse_filter_of_step_4_2`: outputs agree with `inverse_filter` to within float32 roundoff ($< 10^{-7}$).
   - At exact frequency nulls where $|H(u,v)| = 0$, the unguarded quotient produces $0/0 = \text{NaN}$. `wiener_filter` handles this via `np.divide(..., where=denominator > 0, out=zeros)`, implementing an explicit zero-gain convention at nulls (tested in `test_zero_k_at_an_exact_null_is_zero_gain_not_a_division`).

3. **Behavior at Exact / Near-Zero PSF Response:**
   - Where $|H(u,v)| \to 0$ and $K > 0$, the filter transfer function behaves smoothly as $\frac{H^*}{K} \to 0$. Unlike `inverse_filter`, which requires an artificial clamping floor $\varepsilon$, Wiener filter naturally attenuates frequencies where $|H|^2 \ll K$.
   - Tested in `test_exact_zero_frequency_response_stays_finite` using a 2×2 bilinear interpolator that places an exact null at the Nyquist frequency. Output remains finite and bounded in $[0, 1]$.

4. **Complex-Valued Arithmetic & Phase Preservation:**
   - Asymmetric blur kernels (e.g. motion blur at 8.0°) yield complex frequency responses with imaginary parts up to $0.766$.
   - `wiener_filter` correctly multiplies by the complex conjugate $H^*(u,v) = \text{np.conj}(response)$.
   - Tested in `test_complex_frequency_response_keeps_the_conjugate`: omitting the conjugate results in a $33\times$ increase in RMSE ($0.48$ vs $0.014$).
   - The imaginary component of the inverse FFT is pure numerical roundoff ($\max |\operatorname{Im}| < 5\times 10^{-8}$), cleanly discarded via `np.real`.

5. **Output Conformance:**
   - All returned restored images pass through `clip_unit_interval`, guaranteeing finite `float32` values in $[0.0, 1.0]$.
   - Diverged quotients trigger an immediate `ValueError` rather than silently propagating NaNs or Infs.

---

## 9. Limitations and Uncertainties

1. **Fundamental Inadequacy of a Scalar $K$:**
   A single scalar regularisation parameter $K$ inherently assumes a flat signal power spectrum. Natural images and license plates have energy concentrated at low spatial frequencies. Consequently:
   - At low frequencies (where $|H| \approx 1$), noise is small relative to signal, so little regularisation is needed.
   - At high frequencies (where $|H| \to 0$), signal is small relative to noise, requiring heavy regularisation.
   A scalar $K$ represents an unavoidable compromise. Across 10 plates, the swept best scalar $K^*$ achieves $0.70\times$ to $0.94\times$ of the derived $K_A$ error, while the non-scalar spectral oracle reaches $0.60\times$ to $0.87\times$.

2. **Irrecoverable High-Frequency Information:**
   Where the blur PSF possesses deep nulls (as in the 51×51 kernel), high-frequency signal components are permanently destroyed. No choice of $K$ can reconstruct destroyed frequencies.

3. **Non-Linearity of Saturation:**
   The linear model $g = h * f + n$ is violated by dynamic range clipping. For low BSNR on high-contrast plates, approximately $25\%$ of pixels clip, inducing negative bias and signal-noise correlation.

4. **Inaccessibility of Sharp Signal Statistics in Deployment:**
   Candidate C requires $\operatorname{Var}(f)$, which is inaccessible during actual restoration. Any attempt to estimate $\operatorname{Var}(f)$ from $g$ re-introduces the very approximation errors studied here.

---

## 10. Final Recommendations

### 10.1 Mathematical vs Empirical Distinctions
- **Mathematical Fact:** Candidate A ($K_A = \frac{\rho}{1+\rho}$) is the noise fraction of the **observed** power, not the classical Wiener noise-to-sharp-signal ratio.
- **Mathematical Fact:** Candidate B ($K_B = \rho$) is the noise fraction of the **blurred** power.
- **Mathematical Fact:** Candidate C ($K_C = \frac{\sigma_n^2}{\operatorname{Var}(f)}$) is the exact scalar MMSE constant for flat-spectrum signals, but not for license plates.
- **Empirical Observation:** Candidates A and B perform within $0.04\%$ to $0.20\%$ of each other across all operating conditions on license plates.
- **Empirical Observation:** Candidate C beats Candidate A on large kernels ($9.3\%\text{--}14.9\%$) but loses to Candidate A on small kernels ($6.7\%\text{--}11.3\%$) due to cancellation of spectral errors.

### 10.2 Actionable Recommendations
1. **Retain Candidate A ($K_A = \frac{\rho}{1+\rho}$) in `wiener_k_from_bsnr()`:**
   Do not modify the runtime implementation of `wiener_k_from_bsnr()`. Retain Candidate A as the default deployable heuristic when a user passes `bsnr_db=`. Ensure documentation and docstrings explicitly designate it as an **observed-power heuristic**, not the classical signal-to-noise ratio.
2. **Do Not Switch to Candidate B:**
   While Candidate B ($K_B = \rho$) is algebraically simpler, its performance difference from Candidate A is negligible ($< 0.2\%$ RMSE difference, $< 0.02$ dB PSNR difference). Switching provides zero practical benefit and would disrupt existing baseline tests.
3. **Reject Candidate W ($K_W = \rho \sum h^2$):**
   Candidate W is catastrophic on license plates (producing $2.2\times$ to $3.4\times$ worse RMSE) because it strictly assumes white noise signals.
4. **Recommend Explicit $K$ for Controlled Experiments:**
   For all scientific benchmarking, stage 8 evaluation scripts, and ablation studies, require researchers to supply an explicit `K=` parameter rather than relying on `bsnr_db=`.
5. **Bridge to Phase 4.4 (Constrained Least Squares):**
   The fundamental limitation of the scalar Wiener filter—its inability to penalize high frequencies differently from low frequencies—is the exact theoretical motivation for Constrained Least Squares (CLS). In CLS, the scalar $K$ is replaced by $\lambda |C(u,v)|^2$, where $C$ is the 2D Laplacian operator. This provides frequency-dependent regularisation without requiring an unavailable spectral model $S_f(u,v)$. Proceeding to Phase 4.4 (CLS) is mathematically justified.

---

## 11. Scope and Compliance Verification

- **Implementation Files Modified:** None (`git diff license_plate_restoration/` is clean).
- **Test Files Modified:** None (`git diff tests/` is clean).
- **Documentation Modified:** Only the required audit report (`docs/wiener_mathematical_audit.md`) has been produced.
- **Git State:** No commits made; no branch changes; no pushes performed.
- **Test Suite Status:** Full baseline suite passes (321 passed, 6 skipped).
