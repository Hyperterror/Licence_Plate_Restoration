# Implementation Plan: License Plate Restoration

## Overview

Implement the three pipeline stages (Steps 1, 2, and 5) of the classical license-plate restoration system in six Python modules. All signal-processing primitives are written from scratch with NumPy; OpenCV is limited to I/O, perspective warping, and connected-components; PIL is limited to text rendering; Matplotlib/Pandas are limited to plots and tables.

The implementation order follows the data flow: clean plate acquisition (1.1 → 1.2 → 1.3) → degradation (2.1 → 2.2 → 2.3) → binarization (5.1 → 5.2 → 5.3) → integration smoke test.

---

## Tasks

- [x] 1. Set up project structure and shared test infrastructure
  - Create the `license_plate_restoration/` package directory with an `__init__.py`.
  - Create a `data/` subdirectory and a `data/degraded/` subdirectory.
  - Create `tests/` directory with `conftest.py` defining shared Hypothesis strategies: `valid_plate_text_strategy()`, `invalid_plate_text_strategy()`, `float32_image_strategy()`, `binary_uint8_image_strategy()`, `binary_kernel_strategy()`.
  - Install `pytest`, `hypothesis`, `numpy`, `opencv-python`, `pillow`, `matplotlib`, `pandas` (pin exact versions in `requirements.txt`).
  - _Requirements: all modules_

- [x] 2. Implement `renderer.py` — synthetic Indian plate rendering (Step 1.1)
  - [x] 2.1 Implement `render_plate(text, seed) -> np.ndarray`
    - Validate `text` against the pattern `^[A-Z]{2} [0-9]{2} [A-Z]{2} [0-9]{4}$` with the two-letter prefix in `STATE_CODES`; raise `ValueError` if invalid.
    - Allocate a `(96, 336)` float32 array filled with `GREY = 128/255`.
    - Fill inner `(80, 320)` region with `1.0` (white); draw 2-px black border via NumPy slicing.
    - Use `PIL.ImageDraw` to render `text` centred on the inner region; composite back to NumPy.
    - Return float32 array in `[0, 1]`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.8_
  - [x] 2.2 Implement `generate_plates(n, seed) -> list[tuple[np.ndarray, str]]`
    - Seed a `numpy.random.default_rng(seed)` to pick random state codes, digits, and letters.
    - Generate `n` plates whose text strings all match `^[State_Code][space][0-9]{2}[space][A-Z]{2}[space][0-9]{4}$`.
    - Each call to `render_plate` is made with the generated text.
    - _Requirements: 1.5, 1.6, 1.7_
  - [x]* 2.3 Write property tests for `renderer.py`
    - **Property 1: render_plate output shape and dtype** — `@given(valid_plate_text_strategy(), st.integers(0, 2**31-1))`, assert shape `(96, 336)` and dtype float32.
    - **Property 2: render_plate determinism** — same `(text, seed)` yields pixel-identical arrays.
    - **Property 3: generate_plates text format** — every text matches the regex and state code is in `STATE_CODES`.
    - **Property 4: render_plate rejects invalid text** — `@given(invalid_plate_text_strategy())`, assert `ValueError`.
    - **Validates: Requirements 1.1, 1.4, 1.6, 1.7, 1.8**
  - [x]* 2.4 Write unit tests for `renderer.py`
    - `test_render_plate_white_background`: check inner region is white, border is black, margin is grey.
    - `test_render_plate_invalid_raises`: `"XX 00 AA 1234"` with non-existent state code raises `ValueError`.
    - _Requirements: 1.2, 1.3, 1.8_

- [x] 3. Implement `ccpd_loader.py` — CCPD plate loading (Step 1.2)
  - [x] 3.1 Implement `load_ccpd_plate(image_path) -> tuple[np.ndarray, str] | None`
    - Parse four corner coordinates and seven character indices from the CCPD filename (split on `-`).
    - Use `cv2.getPerspectiveTransform` + `cv2.warpPerspective(INTER_CUBIC)` to warp the plate region to `240×76`.
    - Append 8 px grey margin via `cv2.copyMakeBorder(BORDER_CONSTANT, value=128)`; convert to float32, divide by 255.
    - Drop index 0 (province); decode indices 1–6 via `ALPHABETS`/`ADS` tables; return `(array, text)`.
    - Return `None` on any parse or read failure (do not raise).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.7_
  - [x] 3.2 Implement `sample_ccpd(folder, n, seed) -> tuple[np.ndarray, np.ndarray, list[str]]`
    - Seed `numpy.random.default_rng(seed)` to shuffle the file list.
    - Iterate, calling `load_ccpd_plate`; skip `None` returns; stop when `n` valid plates are collected.
    - Stack images into `(n, H, W)` float32 array and texts into `(n,)` string array.
    - _Requirements: 2.5, 2.6, 2.7_
  - [x]* 3.3 Write property tests for `ccpd_loader.py`
    - **Property 5: load_ccpd_plate output invariants** — for valid CCPD paths, assert shape `(92, 256)`, dtype float32, values in `[0, 1]`, text length 6.
    - **Property 6: sample_ccpd returns exactly n plates** — for `n ≤ parseable files`, first-dim of returned arrays equals `n`.
    - **Property 7: malformed filenames are silently skipped** — folder with mixed valid/malformed files; no exception; only valid plates returned.
    - **Validates: Requirements 2.1, 2.3, 2.4, 2.5, 2.6, 2.7**
  - [x]* 3.4 Write unit test for CCPD filename parsing
    - `test_ccpd_parse_known_filename`: use a known CCPD filename and verify the decoded text matches the expected ground-truth string.
    - _Requirements: 2.2, 2.3_

- [x] 4. Implement `dataset_writer.py` — dataset persistence (Step 1.3)
  - [x] 4.1 Implement `save_dataset(synthetic_imgs, synthetic_text, base_imgs, base_text, blur_imgs, blur_text, out_dir)`
    - Save `plates.npz` with keys `synthetic_imgs`, `synthetic_text`, `base_imgs`, `base_text`, `blur_imgs`, `blur_text`.
    - Write `plates.csv` with columns `id`, `source`, `true_text`, `filename` (one row per plate).
    - Generate three preview PNGs (`preview_synthetic.png`, `preview_ccpd_base.png`, `preview_ccpd_blur.png`): 5×4 grid of 20 plates each, titled with `true_text`, using Matplotlib only; do not modify stored arrays.
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  - [x]* 4.2 Write property test for dataset round-trip
    - **Property 8: dataset round-trip preserves float32 values in [0,1]** — `@given(float32_image_batch_strategy())`, save and reload; assert dtype float32 and all values in `[0, 1]`.
    - **Validates: Requirements 3.3**
  - [x]* 4.3 Write unit tests for `dataset_writer.py`
    - `test_npz_keys_present`: call `save_dataset`, reload with `np.load`, assert all six keys exist.
    - `test_csv_columns`: assert CSV has `id`, `source`, `true_text`, `filename` columns.
    - `test_preview_file_created`: assert three PNG files exist after `save_dataset`.
    - **Property 9: preview does not mutate plate arrays** — arrays are pixel-identical before and after preview generation.
    - _Requirements: 3.1, 3.2, 3.4, 3.5_

- [x] 5. Checkpoint — clean plate pipeline
  - Ensure all tests in steps 2–4 pass. Run `pytest tests/ -k "renderer or ccpd or dataset" --tb=short`. Ask the user if questions arise.

- [x] 6. Implement `psf_builder.py` — PSF construction (Step 2.1)
  - [x] 6.1 Implement `make_motion_psf(length, angle) -> np.ndarray`
    - Return `np.array([[1.0]], dtype=np.float32)` when `length == 0`.
    - Allocate `(2*length+1, 2*length+1)` float32 kernel; convert `angle` degrees to radians; direction vector `(cos θ, -sin θ)`.
    - For each sub-pixel sample `t ∈ [-length, length]` (step 0.1), compute continuous position and distribute weight `0.1` to four nearest pixels via bilinear interpolation.
    - Normalise so kernel sums to 1.
    - _Requirements: 4.1, 4.2, 4.3, 4.7_
  - [x] 6.2 Implement `make_defocus_psf(radius) -> np.ndarray`
    - Return `np.array([[1.0]], dtype=np.float32)` when `radius == 0`.
    - Allocate `(2*ceil(radius)+1, 2*ceil(radius)+1)` float32 kernel.
    - For each pixel, compute distance from centre; weight = `1.0` if `d < radius - 0.5`, `0.0` if `d > radius + 0.5`, else `radius + 0.5 - d`.
    - Normalise so kernel sums to 1.
    - _Requirements: 4.4, 4.5, 4.6, 4.8_
  - [x]* 6.3 Write property tests for `psf_builder.py`
    - **Property 10: PSF kernels are normalised** — `@given(st.integers(0, 30), st.floats(0, 360))` for motion; `@given(st.floats(0, 10))` for defocus; assert `.sum()` within 1e-5 of 1.0, odd size, float32.
    - **Validates: Requirements 4.1, 4.4, 4.7, 4.8**
  - [x]* 6.4 Write unit tests for `psf_builder.py`
    - `test_motion_psf_zero_length`: assert returns `[[1.0]]`.
    - `test_defocus_psf_zero_radius`: assert returns `[[1.0]]`.
    - _Requirements: 4.2, 4.5_

- [x] 7. Implement `degradation.py` — blur, downsample, noise (Steps 2.2a, 2.2b)
  - [x] 7.1 Implement `blur_fft(img, psf) -> np.ndarray`
    - Compute `pad = max(psf.shape) // 2`; reflect-pad `img` with `np.pad(mode='reflect')`.
    - Allocate zero `psf_pad` of padded shape; place PSF with centre at `(0, 0)` using quadrant copy (equivalent to `np.fft.ifftshift`).
    - Multiply `fft2(img_pad) * fft2(psf_pad)`; take real part of `ifft2`; crop back to original shape.
    - Return float32.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  - [x] 7.2 Implement `downsample(img, factor) -> np.ndarray`
    - Return `img.copy()` for `factor == 1`.
    - Crop `img` to `(H//factor * factor, W//factor * factor)`; reshape to `(h, factor, w, factor)`; take `.mean(axis=(1, 3))`.
    - Return float32.
    - _Requirements: 5.6, 5.7, 5.8_
  - [x] 7.3 Implement `add_noise(img, bsnr_db, seed) -> np.ndarray`
    - Compute `sigma = sqrt(var(img) / 10**(bsnr_db / 10))`.
    - Generate noise with `numpy.random.default_rng(seed).normal(0, sigma, img.shape)`.
    - Clip output to `[0, 1]`; return float32.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  - [x]* 7.4 Write property tests for `degradation.py` (blur_fft and downsample)
    - **Property 11: blur_fft with delta PSF is identity** — `@given(float32_image_strategy())`, assert max pixel deviation ≤ 1e-5.
    - **Property 12: downsample shape invariants** — factor 1 returns identical array; factor 2 returns `(H//2, W//2)` shape.
    - **Validates: Requirements 5.5, 5.7, 5.8**
  - [x]* 7.5 Write property tests for `add_noise`
    - **Property 13: add_noise output range and determinism** — float32 values in `[0, 1]`; same arguments return identical array.
    - **Property 14: add_noise BSNR accuracy** — measured BSNR differs from target by no more than 1 dB.
    - **Validates: Requirements 6.1, 6.3, 6.5**
  - [x]* 7.6 Write unit tests for `degradation.py`
    - `test_downsample_block_mean_values`: 2×2 block `[0.4, 0.6, 0.8, 1.0]` → mean `0.7`.
    - `test_blur_fft_reflect_no_wrap_artefacts`: large PSF; border pixel value close to inner pixel value.
    - _Requirements: 5.2, 5.6_

- [x] 8. Implement `run_degradation_grid` in `degradation.py` (Step 2.3)
  - [x] 8.1 Implement `run_degradation_grid(plates_npz, out_dir, seed=0)`
    - Load plates from `plates_npz`.
    - For each plate, sample: `blur_length` (random int `[0, 21]`), `angle` (random float `[0°, 180°)`), `defocus_radius` (random int `[0, 5]`), iterate over `bsnr_db ∈ {40, 30, 25, 20}` and `downsample_factor ∈ {1, 2}`.
    - Construct combined PSF by convolving `make_motion_psf` result with `make_defocus_psf` result via `blur_fft`.
    - Apply `blur_fft`, then `downsample`, then `add_noise`; clip to `[0, 1]`.
    - Log `(plate_id, blur_length, angle, defocus_radius, bsnr_db, downsample_factor, seed)` per output.
    - Save `degraded.npz` (keys `<id>_img` and `<id>_psf`) and `degraded.csv`.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [x]* 8.2 Write property test for combined PSF normalisation
    - **Property 15: combined PSF from grid loop is normalised** — `@given(st.integers(0, 21), st.floats(0, 180), st.integers(0, 5))`, assert combined PSF sum within 1e-5 of 1.0.
    - **Validates: Requirements 7.2**
  - [x]* 8.3 Write unit test for degradation grid logging
    - `test_degraded_log_has_all_fields`: run grid on 2 plates; assert `degraded.csv` rows contain all 7 required fields.
    - _Requirements: 7.3, 7.4_

- [x] 9. Checkpoint — degradation pipeline
  - Ensure all tests in steps 6–8 pass. Run `pytest tests/ -k "psf or degradation or noise" --tb=short`. Ask the user if questions arise.

- [x] 10. Implement `binarizer.py` — Otsu thresholding (Step 5.1)
  - [x] 10.1 Implement `otsu_threshold(img) -> float`
    - Scale float32 image to `[0, 255]` integers: `bins = (img * 255).astype(np.int32)`.
    - Accumulate 256-bin histogram manually: `np.add.at(hist, bins.ravel(), 1)`; normalise to probability vector `p`.
    - Compute prefix sums `w0` and `mu`; compute `between_class_var[t]` for each bin; handle `w0 == 0` and `w0 == 1` edge cases by setting variance to 0.
    - Return `argmax(between_class_var) / 255.0`.
    - _Requirements: 8.1, 8.2_
  - [x] 10.2 Implement `binarize(img) -> np.ndarray`
    - Call `otsu_threshold(img)` internally (no `cv2.threshold(THRESH_OTSU)`).
    - Compare class means to select polarity: lower-mean class → foreground (0); higher-mean class → background (255).
    - Return uint8 array.
    - _Requirements: 8.3, 8.4_
  - [x]* 10.3 Write property tests for Otsu thresholding
    - **Property 16: otsu_threshold accuracy vs. OpenCV reference** — `@given(float32_image_strategy())`, assert `|otsu_threshold(img) * 255 - t_cv| ≤ 1`.
    - **Property 17: binarize output invariants** — `@given(float32_image_strategy())`, assert unique values ⊆ `{0, 255}`.
    - **Validates: Requirements 8.1, 8.3, 8.5**
  - [x]* 10.4 Write unit test for Otsu thresholding
    - `test_otsu_manual_vs_cv2`: run on 5 representative plate images; assert bin-level agreement.
    - _Requirements: 8.5_

- [x] 11. Implement `binarizer.py` — morphological operations (Step 5.2)
  - [x] 11.1 Implement `erode(img, kernel) -> np.ndarray`
    - Reflect-pad `img` by `(kh//2, kw//2)`.
    - Use `np.lib.stride_tricks.as_strided` to create `(H, W, kh, kw)` sliding-window view.
    - `out[i,j] = 0` iff all kernel-covered input pixels are 0; else 255.
    - No `cv2.erode`, `scipy.ndimage`, or `skimage` calls.
    - Return binary uint8.
    - _Requirements: 9.1, 9.2_
  - [x] 11.2 Implement `dilate(img, kernel) -> np.ndarray`
    - Same stride-tricks approach; `out[i,j] = 0` iff any kernel-covered input pixel is 0.
    - No `cv2.dilate`, `scipy.ndimage`, or `skimage` calls.
    - Return binary uint8.
    - _Requirements: 9.1, 9.3_
  - [x] 11.3 Implement `opening(img, kernel)` and `closing(img, kernel)`
    - `opening = dilate(erode(img, kernel), kernel)`.
    - `closing = erode(dilate(img, kernel), kernel)`.
    - _Requirements: 9.4, 9.5_
  - [x]* 11.4 Write property tests for morphological operations
    - **Property 18: erode and dilate match OpenCV reference** — `@given(binary_uint8_image_strategy(), binary_kernel_strategy())`, assert pixel-identical output vs. `cv2.erode` / `cv2.dilate` (reference only).
    - **Property 19: opening and closing composition** — assert `opening(img, k) == dilate(erode(img, k), k)` and `closing(img, k) == erode(dilate(img, k), k)` pixel-for-pixel.
    - **Validates: Requirements 9.4, 9.5, 9.6, 9.7**

- [x] 12. Implement `binarizer.py` — character segmentation (Step 5.3)
  - [x] 12.1 Implement `segment_characters(binary_plate) -> list[np.ndarray]`
    - Call `cv2.connectedComponentsWithStats` on `binary_plate`.
    - Reject the largest connected component (background/border).
    - Filter remaining components: height ∈ `[20%, 90%]` of plate height; aspect ratio `h/w ∈ [1.0, 6.0]`; area ≥ 50 px.
    - Extract tight bounding-box crops from `binary_plate`; sort left-to-right by x-coordinate.
    - Return list without raising even if count < 4 or > 10.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_
  - [x]* 12.2 Write property tests for `segment_characters`
    - **Property 20: segment_characters left-to-right ordering and no exception** — `@given(binary_uint8_image_strategy(shape=(92, 256)))`, assert no exception and x-coordinates non-decreasing.
    - **Validates: Requirements 10.1, 10.5**
  - [x]* 12.3 Write unit tests for `segment_characters`
    - `test_segment_characters_sorted_order`: synthetic binary plate with 7 known character blobs; assert 7 crops returned in correct x-order.
    - `test_segment_characters_rejects_background`: largest component (full border) is excluded from results.
    - _Requirements: 10.2, 10.3_

- [x] 13. Checkpoint — binarization pipeline
  - Ensure all tests in steps 10–12 pass. Run `pytest tests/ -k "otsu or binarize or erode or dilate or segment" --tb=short`. Ask the user if questions arise.

- [x] 14. End-to-end integration and wiring
  - [x] 14.1 Write the integration smoke test `test_full_pipeline_smoke`
    - Call `generate_plates(5, seed=42)` → `save_dataset(...)` → `run_degradation_grid(...)` on 5 plates with a small parameter subset.
    - Assert no exceptions, `plates.npz` and `degraded.npz` files exist, `degraded.csv` has the expected columns and at least one row.
    - _Requirements: 3.1, 7.3, 7.4_
  - [x]* 14.2 Write integration tests for preview and log completeness
    - `test_preview_file_created`: assert `preview_synthetic.png`, `preview_ccpd_base.png`, `preview_ccpd_blur.png` exist after `save_dataset`.
    - `test_degraded_log_has_all_fields`: assert `degraded.csv` rows contain all 7 fields: `plate_id`, `blur_length`, `angle`, `defocus_radius`, `bsnr_db`, `downsample_factor`, `seed`.
    - _Requirements: 3.4, 7.3_

- [x] 15. Final checkpoint — full test suite
  - Run `pytest tests/ --tb=short` and ensure all non-optional tests pass. Ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each task references specific requirements for traceability.
- Checkpoints (tasks 5, 9, 13, 15) ensure incremental validation at the end of each pipeline stage.
- Property tests use Hypothesis with a minimum of 100 examples; OpenCV (`cv2.erode`, `cv2.dilate`, `cv2.threshold`) is used in tests only as a cross-check reference and never in production code.
- All signal-processing primitives (PSF construction, FFT convolution, block-mean downsampling, noise injection, Otsu thresholding, erosion/dilation) are implemented from scratch with NumPy.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "3.1"] },
    { "id": 1, "tasks": ["2.2", "2.3", "2.4", "3.2"] },
    { "id": 2, "tasks": ["3.3", "3.4", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "6.1"] },
    { "id": 4, "tasks": ["6.2", "6.3", "6.4"] },
    { "id": 5, "tasks": ["7.1", "7.2", "7.3"] },
    { "id": 6, "tasks": ["7.4", "7.5", "7.6"] },
    { "id": 7, "tasks": ["8.1"] },
    { "id": 8, "tasks": ["8.2", "8.3", "10.1"] },
    { "id": 9, "tasks": ["10.2", "10.3", "10.4"] },
    { "id": 10, "tasks": ["11.1", "11.2"] },
    { "id": 11, "tasks": ["11.3", "11.4"] },
    { "id": 12, "tasks": ["12.1"] },
    { "id": 13, "tasks": ["12.2", "12.3", "14.1"] },
    { "id": 14, "tasks": ["14.2"] }
  ]
}
```
