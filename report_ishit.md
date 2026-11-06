# Steps 1, 2, 5 — Implementation and Hand-over Report

**Owner:** Ishit Agarwal (2026909005) — steps 1, 2, 5
**Team:** IshIT · Ujwal (2026701052, steps 3, 6, 7) · Ashish (2026909001, steps 4, 8)
**Project:** CS7.404 Digital Image Processing, Monsoon 2026 — *Classical Restoration of Blurred License Plate Images*
**Status:** complete — 105 tests passing, both datasets built and verified

Forward model implemented by these stages:

```
g(x, y) = D( h(x, y) * f(x, y) ) + n(x, y)
```

`f` = sharp plate (Step 1), `h` = PSF (Step 2.1), `*` = convolution (Step 2.2), `D` = block-mean decimation (Step 2.2), `n` = additive Gaussian noise (Step 2.2).

---

## 1. TL;DR for teammates

Ownership per `Work_Division.pdf`: **Ishit** steps 1, 2, 5 — **Ujwal** steps 3, 6, 7 — **Ashish** steps 4, 8.

1. **Ujwal (step 3, PSF estimation) and Ashish (step 4, restoration):** read `<key>_img` and its sibling `<key>_psf` from `degraded.npz` and use **both exactly as stored**. The PSF is already matched to that image's resolution. Never rebuild the kernel from the logged `blur_length` / `angle` / `defocus_radius`.
2. **Ujwal (step 6, OCR):** call `binarize(img)` on the restored plate and get `uint8` pixels in `{0, 255}` where **0 is ink**. Then `segment_characters(binary)` gives tight crops already sorted left to right. Polarity is automatic; do not pass `polarity="dark"` for CCPD.
3. **Ground truth** for every degraded image is the `true_text` column of `degraded.csv`, one row per image. Do not guess it from the filename.
4. **Do not break the AST guards** in `tests/test_integration.py`. They reject `scipy`/`skimage`/`torch`, `np.histogram`, `np.convolve`, `np.correlate`, `np.percentile`, and any `cv2.*` outside the whitelist (image I/O, straightening, rotation, connected components). Put a scikit-image PSNR cross-check in `tests/`, not in the package, or the guard fails.

---

## 2. Where everything lives

| Path | Responsibility |
|---|---|
| `license_plate_restoration/renderer.py` | Step 1.1 — synthetic Indian plate rendering (`f`) |
| `license_plate_restoration/ccpd_loader.py` | Step 1.2 — CCPD filename parsing, perspective straightening |
| `license_plate_restoration/dataset_writer.py` | Step 1.3 — `plates.npz` / `plates.csv` / preview grids |
| `license_plate_restoration/psf_builder.py` | Step 2.1 — motion and defocus PSFs, PSF composition |
| `license_plate_restoration/degradation.py` | Steps 2.2, 2.3 — FFT blur, decimation, noise, grid loop |
| `license_plate_restoration/binarizer.py` | Steps 5.1–5.3 — Otsu, morphology, segmentation |
| `license_plate_restoration/data/` | `plates.npz`, `plates.csv`, preview PNGs, `degraded/` |
| `tests/` | 8 test modules, 105 tests (unit + Hypothesis properties + AST guards) |
| `.kiro/specs/license-plate-restoration/` | requirements / design / tasks (all 52 task boxes ticked) |
| `build_ccpd.py` | **Pre-existing, superseded, do not use** — see §7.6 |

Every module is runnable as a demo (`python -m license_plate_restoration.<module>`), which writes a preview PNG and prints its numbers.

### 2.1 Hand-over chain (`Work_Division.pdf`)

| From | What | To |
|---|---|---|
| Ishit (1, 2) | clean plates + true text | everyone |
| Ishit (1, 2) | degraded images + **true (oracle) PSF** | Ujwal (3), Ashish (4), Ujwal (7) |
| Ujwal (3) | estimated PSF | Ashish (4) |
| Ashish (4) | restored images | Ishit (5), Ujwal (7) |
| **Ishit (5)** | **binary plate + character crops** | **Ujwal (6)** |
| Ujwal (6, 7) | OCR text + confidence, scores, outcome | Ashish (8) |
| Ashish (8) | results table + plots | everyone |

So the contract for my step 5 is exactly two things: a `uint8` binary plate with 0 = ink, and a left-to-right list of tight character crops. §10 gives the recipes.

---

## 3. Public API

### Step 1

| Function | Returns |
|---|---|
| `render_plate(text: str, seed: int = 0) -> np.ndarray` | `float32 (96, 336)`; raises `ValueError` on a malformed registration or unknown state code |
| `generate_plates(n: int, seed: int = 0) -> list[tuple[np.ndarray, str]]` | `n` unique valid registrations |
| `parse_ccpd_filename(filename: str) -> tuple[np.ndarray, str] \| None` | `(4x2 corners, 6-char text)` or `None` |
| `load_ccpd_plate(image_path: str) -> tuple[np.ndarray, str] \| None` | `float32 (92, 256)` straightened plate + text, or `None` |
| `sample_ccpd(folder: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]` | stacked images, texts, filenames |
| `save_dataset(synthetic_imgs, synthetic_text, base_imgs, base_text, blur_imgs, blur_text, out_dir, *, base_filenames=None, blur_filenames=None, n_preview=20) -> None` | writes `.npz`, `.csv`, 3 preview grids |

### Step 2

| Function | Returns |
|---|---|
| `make_motion_psf(length: int, angle: float) -> np.ndarray` | `float32 (2L+1, 2L+1)`, centred, sums to 1 |
| `make_defocus_psf(radius: float) -> np.ndarray` | `float32 (2*ceil(r)+1, ...)`, disc, centred, sums to 1 |
| `combine_psfs(motion_psf, defocus_psf) -> np.ndarray` | direct convolution, `(m+d-1)` square, sums to 1 |
| `psf_is_valid(psf, tol=1e-5) -> bool` | odd, finite, non-negative, sums to 1, centre of mass on the middle pixel |
| `blur_fft(img, psf) -> np.ndarray` | `float32`, same shape as `img`, reflect-padded so there is no wrap-around, not clipped |
| `downsample(img, factor) -> np.ndarray` | `float32 (H//f, W//f)` block mean; `factor=1` returns a copy |
| `noise_sigma(img, bsnr_db) -> float` | `sqrt(var(img) / 10**(bsnr/10))` |
| `add_noise(img, bsnr_db, seed, saturation_compensation=True) -> np.ndarray` | `float32` clipped to `[0, 1]`, reproducible per `seed` |
| `measure_bsnr(clean, noisy) -> float` | measured BSNR in dB |
| `effective_psf(psf, factor) -> np.ndarray` | PSF as seen by an image decimated by `factor` |
| `run_degradation_grid(plates_npz, out_dir, seed=0, *, limit=None, sources=None) -> None` | writes `degraded.npz` + `degraded.csv` |

### Step 5

| Function | Returns |
|---|---|
| `otsu_threshold(img) -> float` | cut in `[0, 1]`; pixels `<= t*255` form the darker class |
| `binarize(img, polarity="auto", margin=8) -> np.ndarray` | `uint8 {0, 255}`, **0 = ink** |
| `erode(img, kernel)`, `dilate(img, kernel)` | `uint8 {0, 255}`, 0 = ink |
| `opening(img, kernel)` = `dilate(erode(...))`, `closing(img, kernel)` = `erode(dilate(...))` | same |
| `square_kernel(size=3) -> np.ndarray` | box structuring element |
| `character_boxes(binary_plate, ...) -> list[tuple[int, int, int, int]]` | `(x, y, w, h)` sorted left to right |
| `segment_characters(binary_plate, min_height_ratio=0.20, max_height_ratio=0.90, min_aspect=1.0, max_aspect=6.0, min_area=50) -> list[np.ndarray]` | tight `uint8` crops, left to right; empty or non-10 counts are possible and never raise |

---

## 4. Data artifacts

### `license_plate_restoration/data/plates.npz` — 10.5 MB

| Key | Shape | Dtype |
|---|---|---|
| `synthetic_imgs` | `(600, 96, 336)` | `float32` in `[0, 1]` |
| `synthetic_text` | `(600,)` | `<U13` |
| `base_imgs` | `(400, 92, 256)` | `float32` in `[0, 1]` |
| `base_text` | `(400,)` | `<U32` |
| `blur_imgs` | `(50, 92, 256)` | `float32` in `[0, 1]` |
| `blur_text` | `(50,)` | `<U32` |

`plates.csv` — 1050 rows, columns `id, source, true_text, filename`; sources `synthetic` 600, `ccpd_base` 400, `ccpd_blur` 50.

Preview grids: `preview_synthetic.png`, `preview_ccpd_base.png`, `preview_ccpd_blur.png` (20 plates each, 5x4, `true_text` as the title).

### `license_plate_restoration/data/degraded/` — `degraded.npz` 471.4 MB, `degraded.csv` 0.7 MB

| Item | Value |
|---|---|
| Rows in `degraded.csv` | 8400 |
| Distinct `plate_id` | 1050 (8 outputs each) |
| Arrays in `degraded.npz` | 16800 = 8400 images + 8400 PSFs |
| Condition cells | 1050 per (`bsnr_db`, `downsample_factor`) pair, 8 pairs |
| Columns | `plate_id, blur_length, angle, defocus_radius, bsnr_db, downsample_factor, seed, key, true_text` |
| Key format | `<plate_id>_<sha1(parameters)[:10]>`, e.g. `synthetic_0000_e5a3683f3d` |
| Grid ranges | `blur_length` 0–21, `angle` [0, 180), `defocus_radius` 0–5, `bsnr_db` {40, 30, 25, 20}, `downsample_factor` {1, 2} |

One blur condition is drawn per plate, as `combine_psfs(motion, disc)`: **194** plates are pure motion, **43** pure defocus, **9** unblurred and **822** a mixture. That split matters to blind PSF estimation — see §10.

Loading recipe (this is all a consumer needs):

```python
import numpy as np, pandas as pd

log = pd.read_csv("license_plate_restoration/data/degraded/degraded.csv")
archive = np.load("license_plate_restoration/data/degraded/degraded.npz")

row = log.iloc[0]
image = archive[f"{row['key']}_img"]     # float32, [0, 1]
kernel = archive[f"{row['key']}_psf"]    # float32, sums to 1, odd-sized, matches image
```

Recovering the clean reference for a row:

```python
clean = np.load("license_plate_restoration/data/plates.npz")
prefix, index = row["plate_id"].rsplit("_", 1)      # "ccpd_base", "0007"
key = {"synthetic": "synthetic_imgs", "ccpd_base": "base_imgs",
       "ccpd_blur": "blur_imgs"}[prefix]
reference = clean[key][int(index)]
```

At `downsample_factor == 2` the reference must be decimated the same way before scoring: `downsample(reference, 2)`.

---

## 5. The frozen contract

Ratified by the owner after step 5 and encoded in the code (`DEFAULT_POLARITY`, the docstrings, and `tests/test_binarizer.py`). Treat as the team contract; do not re-derive or re-open these.

* **Images** are grayscale `float32` in `[0, 1]`, 0 = black. Convert to 8-bit only immediately before OCR.
* **PSFs** are odd-sized, centred (no image shift), and sum to 1.
* **Angles** are degrees counter-clockwise from the +x axis. Image rows grow downwards, so the direction vector is `(cos t, -sin t)`.
* **Plate shapes** are `(96, 336)` synthetic and `(92, 256)` CCPD — the 320x80 / 240x76 drawing area **plus an 8 px grey (128/255) margin**. They are not `320x80`.
* **`make_motion_psf` takes a half-length.** `length=15` builds a `31x31` kernel, i.e. a **31 px smear**, not a 15 px one.
* **`binarize` defaults to `polarity="auto"`** (`DEFAULT_POLARITY`), the minority class of the plate interior. `polarity="dark"` is requirement 8.3 implemented literally and is kept only for the report's comparison.
* **`segment_characters` rejects the largest component only when it is border-like** (its box spans or touches the plate edge). A large glyph survives.
* **PSFs in `degraded.npz` are resolution-matched.** Use the stored `_psf` as-is.
* **`add_noise` calibrates its amplitude after clipping** (`saturation_compensation=False` disables it).

---

## 6. Verification evidence

All numbers below were measured on this machine against the artifacts currently on disk.

| Requirement | Evidence |
|---|---|
| 1.1 render | `float32 (96, 336)`, values in `[0, 1]`; identical SHA-256 across separate interpreter sessions; 35 real state codes; format regex enforced |
| 1.2 CCPD | all 11,228 `ccpd_base` filenames parse, every text is exactly 6 chars, zero `O` characters; vertex order (BR, BL, TL, TR) confirmed by eye against warped plates |
| 1.3 dataset | `plates.npz` 600 / 400 / 50, `plates.csv` 1050 rows; 20-plate previews visually matched on every sampled plate |
| 2.1 PSF | `make_motion_psf(2, 0)` row = `[0.1259, 0.2494, 0.2494, 0.2494, 0.1259]` (exact area-weighted trapezoid); `length=0` and `radius=0` give `[[1.0]]`; sums to 1 within 1e-5 for every tested `(length, angle, radius)` |
| 2.2 blur / decimate | delta PSF leaves the image unchanged; centred PSF does not shift; reflect padding proven to remove circular wrap; decimation is an exact block mean |
| 2.2 noise | measured BSNR error **<= 0.003 dB** at 40/30/25/20/15/10 dB with compensation on, vs **+0.90 to +1.00 dB** with it off (see §7.5) |
| 2.3 grid | 8400 rows, 1050 plates, 8 outputs per plate, no duplicate keys, every stored PSF sums to 1.0 |
| 5.1 Otsu | **0.0 bin** difference vs `cv2.threshold(..., THRESH_OTSU)` on 100 plates (50 CCPD + 50 synthetic) |
| 5.2 morphology | **0 mismatches in 240** random cases against `cv2.erode` / `cv2.dilate` through the polarity duality (see §7.3) |
| 5.3 segmentation | synthetic: median **10** crops, 47/60 plates exactly 10; `ccpd_base`: median **7**, 8/400 return nothing; `ccpd_base` decimated x2: median **7** (filter is scale-invariant); `ccpd_blur`: median **0**, 27/50 empty — expected, the input is already blurred |

Measured BSNR detail (`add_noise` on a known blurred plate, seed 0):

| target | 40 | 30 | 25 | 20 | 15 | 10 |
|---|---|---|---|---|---|---|
| compensated | 40.002 | 30.002 | 25.002 | 20.003 | 15.000 | 10.000 |
| raw formula | 40.897 | 30.915 | 25.924 | 20.938 | 15.959 | 10.999 |

Test suite: **105 passed** — `test_binarizer.py` 23, `test_degradation.py` 20, `test_ccpd_loader.py` 14, `test_psf_builder.py` 11, `test_degradation_grid.py` 10, `test_renderer.py` 10, `test_dataset_writer.py` 9, `test_integration.py` 8.

---

## 7. Deviations from the written spec (and why)

**7.1 Requirement 8.3 — ink polarity.** The literal rule ("the class with the lower mean is the ink") is correct for synthetic plates (black on white) and inverts every CCPD plate (white on dark). Measured on ten `ccpd_base` plates: the literal rule keeps **0–2** character components, `"auto"` keeps **7–8**. `"auto"` picks the minority class of the plate *interior*, because the 8 px margin is a fifth to a quarter of the plate (21 % synthetic, 23 % CCPD) and flips a whole-image area test (interior measured 49.1 % below the cut vs 50.9 % above). Both modes are implemented, `"auto"` is the default, and `tests/test_binarizer.py::test_literal_rule_inverts_ccpd_plates` pins the numbers.

**7.2 Requirement 10.3 — "reject the largest component".** Applied unconditionally, it deletes a real glyph whenever the biggest ink blob is a character, which happens on light-on-dark CCPD plates. It is now applied only when the component is *border-like*: its box spans >= 90 % of a plate dimension or touches within 1 px of the plate edge.

**7.3 Requirements 9.6 / 9.7 are mutually exclusive with 9.2 / 9.3.** 9.2 and 9.3 define erosion/dilation with 0 as foreground. `cv2.erode` and `cv2.dilate` treat **255** as the object. Both criteria cannot hold on the same input; we satisfy the definitions (9.2, 9.3) and match OpenCV through its own duality:

```
erode(img, k)  == 255 - cv2.erode(255 - img, k)
dilate(img, k) == 255 - cv2.dilate(255 - img, k)
```

This is verified to be exact on 240 random cases. If a marker insists on literal 9.6/9.7 identity, the polarity convention of the whole step-5 output has to flip first.

**7.4 Motion PSF length.** `length` is a half-length, so the kernel is `2*length+1` wide and the actual smear spans up to **42 px** at the grid maximum. Both project PDFs say "motion blur 15 px" and "blur length 0–21 px", and the figure captions repeat it (`Motion PSF (15 px, 8°)`). Consequence: the blur axis of the sweep plots is mislabelled by a factor of two unless we fix the legend. Recommended: label the axis `blur length L (kernel 2L+1, smear 0–42 px)` and write "31 px smear" wherever the prose says 15 px. Do not halve the sampled range now — the grid is already built.

**7.5 Noise saturation compensation.** A plate is roughly 30 % pure white, so clipping truncates the positive noise tail and removes energy: the raw formula reads up to +1 dB high, which would fail requirement 6.5's 1 dB tolerance. `add_noise` keeps the single seeded normal realisation and only rescales its amplitude (fixed point, 10 iterations, 1e-3 tolerance) until the *realised* `var(clip(img + n) - img)` hits the target. Consequence: the effective sigma is slightly larger than the textbook formula, but the measured BSNR matches the label. Flag off with `saturation_compensation=False` for a literal-model comparison in the report.

**7.6 `build_ccpd.py` (repo root) is superseded.** It duplicates step 1.2 with the same vertex order and char tables, but it writes a separate `ccpd_plates.npz`, uses `cv2.copyMakeBorder` instead of NumPy, and — the real problem — it strips `'O'` characters out of the ground truth (`text = "".join(c for c in chars if c != "O")`), which would silently turn a 6-character plate into 5 characters instead of rejecting it. Measured: zero `O` characters occur in the 11,228 current filenames, so the branch never fires today; it is still a latent bug. `ccpd_loader.py` does not do this and does not depend on that file.

**7.7 Shape.** `render_plate` returns `(96, 336)`, not the bare `320x80` mentioned in the project prose. Ratified: `plates.npz`, the 8,400-image grid and the segmentation height ratios are all built around the margin, so returning the bare drawing area would force a full rebuild.

**7.8 The font is not a plate face.** The proposal asks for a "plate-style font"; `renderer.py` uses Arial Bold (`C:/Windows/Fonts/arialbd.ttf`, size 43), because no FE-Schrift-like face ships with Windows. `ARIALNB.TTF` (Arial Narrow Bold) is installed and is closer to a real plate — taller and narrower, 39 px glyphs against 31 px for the same fit — and switching is a one-line edit to `_FONT_CANDIDATES`. It changes every rendered pixel, though, so `plates.npz` and the whole 8,400-image grid would have to be rebuilt (about 12 minutes) and every downstream number moves. Decide before step 4's parameter sweeps start, not after.

---

## 8. Known limitations

* **`ccpd_blur` has no clean reference.** The proposal uses those 50 plates for a qualitative before/after gallery, which is right: PSNR/SSIM are undefined for them because we never had `f`. Their `true_text` is still ground truth, so OCR accuracy (step 6) *can* be scored on them — 54 % currently yield zero crops after binarization, so treat them as a hard case.
* **x2 decimated plates are near the segmentation height limit.** Height filters are relative to the image passed in; on a `(46, 168)` synthetic plate a glyph is ~15–16 px, i.e. ~33 % of height, against a 20 % floor. Still passes, but a tighter crop or a larger decimation would break it.
* **Otsu is a two-class split**, so the mid-grey margin joins the dark class and comes out as ink, forming a ring around the plate. That ring is the largest component and is discarded, so character counts are unaffected. Cropping the margin before binarizing avoids it entirely.
* **Pixel determinism is per machine** — rendering depends on the resolved system font (`arialbd.ttf`, size 43 for a 14-char plate → 309x31 glyph box).
* **`downsample` drops a partial final block** when the dimensions are not divisible by `factor` (documented, and `96 % 2 == 0`, `336 % 2 == 0`, so it never fires on this data).
* **The 471 MB grid is not in git-friendly shape.** If the repo has to stay small, regenerate with `limit=` or drop the file and rebuild from `plates.npz` in ~11 minutes.

---

## 9. Reproducing everything

```powershell
# tests
python -m pytest -q

# step 1: plates.npz + plates.csv + previews   (~1 min)
python -m license_plate_restoration.dataset_writer

# step 2: degraded.npz + degraded.csv, 8400 images   (~11 min, 471 MB)
python -m license_plate_restoration.degradation --grid

# individual demos, each writes a preview PNG under license_plate_restoration/data/
python -m license_plate_restoration.renderer
python -m license_plate_restoration.ccpd_loader
python -m license_plate_restoration.degradation
python -m license_plate_restoration.binarizer
```

Requirements: Python 3.13, NumPy 2.2.6, OpenCV 4.12.0.88, Pillow 12.1.1, Matplotlib 3.10.8, Pandas 2.3.3, pytest 9.0.2, Hypothesis 6.151.9 (see `requirements.txt`).

---

## 10. Integration recipes

### For steps 3 and 4 — PSF estimation and restoration (Ujwal, then Ashish)

Step 3 estimates the PSF; step 4 restores with it. Ashish's *oracle* arm uses the stored kernel directly, so this recipe serves both.

```python
import numpy as np, pandas as pd
from license_plate_restoration.degradation import downsample

log = pd.read_csv("license_plate_restoration/data/degraded/degraded.csv")
archive = np.load("license_plate_restoration/data/degraded/degraded.npz")

for _, row in log.iterrows():
    degraded = archive[f"{row['key']}_img"]    # float32 [0,1], (96,336) or (46,168)
    psf      = archive[f"{row['key']}_psf"]    # float32, sums to 1, odd-sized
    restored = my_restore(degraded, psf)       # step 4

    clean = reference_for(row)                 # see §4
    if row["downsample_factor"] == 2:
        clean = downsample(clean, 2)
    interior = (slice(8, -8), slice(8, -8))    # ignore the grey margin
    psnr = compute_psnr(restored[interior], clean[interior])
```

Rules that will bite if ignored:

1. **Never re-derive the kernel.** At `downsample_factor == 2` the stored PSF is already about half as wide (31x31 becomes 15x15 for a 31 px smear); rebuilding it from `blur_length`/`angle`/`defocus_radius` models the blur twice as wide as it really is.
2. **PSF and image always travel together** — a kernel is only valid for the image of the same shape and the same `key`.
3. **Crop the margin before scoring.** About a fifth of the plate is flat grey, and `blur_fft` reflect-pads, so the margin was blurred like any other content. Including it adds a large, easy region that inflates PSNR and compresses exactly the differences between filters that Q1 is asking you to rank.
4. **Clip to `[0, 1]`** before computing any metric, and remember the noise is already clipped, so the floor at 20 dB is high.
5. **`binarize` clips internally** (§10, step 6): a restored image with negative overshoot from inverse/RL will have that undershoot mapped to 0 = ink *before* Otsu sees it. Strong ringing hurts step 5 more than PSNR suggests — worth noting for Q1.
6. Use the `bsnr_db` and `downsample_factor` columns to group results; every condition cell has exactly 1050 images.

**Facts specific to step 3 (blind PSF estimation).** One blur condition is drawn per plate, as `combine_psfs(motion, disc)`. Over the 1050 plates:

| condition | plates | what it means for spectrum / cepstrum estimation |
|---|---|---|
| `defocus_radius == 0` | **194** | pure motion — validate the angle and length estimator here first |
| `blur_length == 0` | **43** | pure disc, no stripes: the angle is undefined, do not score it |
| both zero | **9** | delta PSF, nothing to estimate |
| both > 0 | **822** | motion *multiplied* by the disc spectrum |

`blur_length` quartiles are 0 / 5 / 11 / 16 / 21. The 822 mixed plates are the subtle case: convolution in space is multiplication in frequency, and a disc's transform has deep near-zeros, so the combined log-magnitude is notched by the disc and the motion stripes are partly erased at those frequencies. Blind estimation will look far worse on those than on the 194 pure-motion plates, so report the two groups separately or the headline error is dominated by the composition rule rather than by the estimator. Compare a factor-2 estimate against the stored effective PSF (already half-width), never against the full-resolution kernel.

### For step 6 — OCR (Ujwal)

Step 5 hands over exactly two things: the binary plate and the crop list.

```python
from license_plate_restoration.binarizer import binarize, segment_characters, character_boxes

binary = binarize(restored)                  # float32 in -> uint8 {0,255}, 0 = ink
crops  = segment_characters(binary)          # tight crops, left to right
boxes  = character_boxes(binary)             # same order, as (x, y, w, h)
```

Measured crop sizes, so the padding and upscale factors are not guesswork:

| source | height | width |
|---|---|---|
| synthetic, factor 1 | 32 px | 21–32 px |
| synthetic, factor 2 | 16 px | 11–16 px |
| `ccpd_base`, factor 1 | 24–46 px | 15–26 px |

Rules that will bite if ignored:

1. **`0` is ink, `255` is paper.** Invert if your OCR model expects white glyphs on black.
2. **Do not pass `polarity="dark"`** for CCPD plates; it empties them (`test_literal_rule_inverts_ccpd_plates`).
3. **The crops are tight — there is no quiet zone.** The bounding box is the glyph, edge to edge, so `psm 10` has no margin to work with. Pad 4–8 px of white on each side, and at `downsample_factor == 2` also upscale x2–3 (16 px crops are below Tesseract's comfortable 30–40 px cell height; the factor-1 crops at 32 px already sit in range).
4. **The output count is not guaranteed to be 10** and no exception is raised: synthetic plates give 10, `ccpd_base` median is 7. Sort or pad in the OCR stage, never assume.
5. **The height filters are relative to the array you pass in.** Feeding a tight crop around the plate instead of the full margined plate changes the ratios and will filter out real glyphs.
6. **Ground truth has spaces, the crops do not.** `true_text` is `KA 05 MN 1234` (13 characters) while segmentation returns 10 glyphs. Score against `"".join(c for c in true_text if c.isalnum())`, otherwise three spaces are charged as errors on every clean read and CER never reaches 0.
7. **If you want the cleaned-up plate rather than the raw one,** `opening(closing(binary, square_kernel(3)), square_kernel(3))` removes specks and bridges broken strokes; from scratch, and it matches OpenCV through the duality in §7.3.

### Library rules (enforced, not advisory)

| Allowed as a primitive | Written from scratch |
|---|---|
| NumPy arrays, `np.fft`, `np.pad`, `np.add.at` | motion PSF, defocus PSF, PSF convolution, FFT blur with padding, block-mean decimation, noise at target BSNR, Otsu, erosion/dilation/opening/closing, character filtering and sorting |
| PIL — text rasterisation only | — |
| OpenCV — `imread`, `IMREAD_GRAYSCALE`, `getPerspectiveTransform`, `warpPerspective`, `INTER_CUBIC`, `getRotationMatrix2D`, `warpAffine`, `INTER_LINEAR`, `ROTATE_90_CLOCKWISE`, `ROTATE_180`, `ROTATE_90_COUNTERCLOCKWISE`, `connectedComponentsWithStats`, `error` | everything else |
| Matplotlib / Pandas — plots and tables only | — |

`tests/test_integration.py` parses the package source with `ast` and fails the build on any violation, so the rule cannot drift.

---

## 11. Open items

1. **Per-condition report table is not built yet.** Planned: `binarize` + `segment_characters` over all 8400 degraded images, tabulated per `bsnr_db` x `downsample_factor` and per blur-length band. This is the step-5 baseline that step 6 needs before Tesseract enters, and Ashish's step 8 runner will eventually supersede it — but it is cheap and it isolates segmentation failures from OCR failures.
2. **`degraded.npz` is 471 MB.** Decide whether it ships with the repo, becomes a release asset, or is rebuilt on demand (about 11 minutes).
3. **Font choice is still open** — Arial Bold (current, all artifacts built) versus Arial Narrow Bold (closer to a plate face, full rebuild). See §7.8. Close this before step 4's sweeps produce numbers anyone quotes.
4. **The blur axis label is ambiguous** in both project PDFs. See §7.4 — relabel as `blur length L (2L+1 kernel)` or the sweep plots will read as half the blur they actually show.
5. **Teammate stages do not exist yet:** step 3 (Ujwal), step 4 (Ashish), step 6 (Ujwal), step 7 (Ujwal), step 8 (Ashish). Nothing in this repo imports them, and no interface beyond the artifacts in §4 is assumed.
