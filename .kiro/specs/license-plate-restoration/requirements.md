# Requirements Document

## Introduction

This feature implements three pipeline stages of a classical (non-deep-learning) license-plate restoration system for the Digital Image Processing course project (CS7.404).

**Forward model:** `g(x,y) = D( h(x,y) * f(x,y) ) + n(x,y)`  
where `f` is the sharp plate, `h` is the PSF, `D` is downsampling, and `n` is additive noise.

The three stages owned by this team member are:

- **Step 1 – Clean Plates**: render Indian synthetic plates and load CCPD plates.
- **Step 2 – Degradation**: apply motion/defocus blur, downsampling, and noise.
- **Step 5 – Binarization**: Otsu thresholding, morphology, and character segmentation.

All implementations must use only NumPy, OpenCV (I/O, rotation, connected components), PIL (text rendering only), and Matplotlib/Pandas (plots/tables). All signal-processing primitives (PSF construction, FFT-based convolution, downsampling, noise injection, Otsu, erosion, dilation, morphological operations, character filtering) must be written from scratch without library calls performing the core work.

## Glossary

- **Pipeline**: The end-to-end system from clean plate to OCR output.
- **Plate**: A grayscale float32 image in `[0, 1]` representing a single license plate.
- **PSF**: Point Spread Function — a 2-D kernel describing blur, odd-sized, centred, summing to 1.
- **BSNR**: Blurred Signal-to-Noise Ratio in dB; `BSNR = 10 * log10(var(blurred) / sigma^2)`.
- **CCPD**: Chinese City Parking Dataset, a real-world license-plate image collection.
- **Indian_Plate**: A synthetic plate image with the format `AA 00 AA 0000` (e.g. `KA 05 MN 1234`) using real Indian state codes.
- **Renderer**: The module that creates `Indian_Plate` images using PIL for text rendering.
- **Degradation_Pipeline**: The module that applies PSF convolution, downsampling, and noise.
- **Binarizer**: The module that thresholds, applies morphology, and segments characters.
- **delta PSF**: A PSF that is a single centred 1 at its centre pixel and 0 elsewhere; represents no blur.
- **Block_Mean_Downsampler**: A downsampler that averages non-overlapping blocks; implemented from scratch.
- **State_Code**: The two-letter abbreviation of an Indian state used in vehicle registration (e.g., `KA`, `MH`, `DL`).
- **Margin**: An 8-pixel constant-value border (grey, value 128/255 ≈ 0.502) added around every plate.
- **Seed**: An integer value passed to a random number generator to make outputs reproducible.
- **Between_Class_Variance**: The Otsu criterion; variance of class means weighted by class probabilities.
- **Morphological_Element**: A binary structuring element (flat, rectangular or square) used for erosion and dilation.

---

## Requirements

### Requirement 1: Synthetic Indian Plate Rendering (Step 1.1)

**User Story:** As a pipeline user, I want to generate synthetic Indian license-plate images with known ground-truth text, so that I can train, test, and validate downstream steps without relying on real photographs.

#### Acceptance Criteria

1. THE `Renderer` SHALL expose a function `render_plate(text: str, seed: int) -> np.ndarray` that returns a float32 NumPy array of shape `(96, 336)` (320×80 plate plus 8 px `Margin` on each side).
2. WHEN `render_plate` is called, THE `Renderer` SHALL produce a white background plate with a black rectangular border and black bold text centred on the plate.
3. THE `Renderer` SHALL use PIL exclusively for text rendering and SHALL use NumPy for all other image construction.
4. WHEN `render_plate` is called with a valid `text` string, THE `Renderer` SHALL reproduce the identical pixel array for the same `(text, seed)` pair across separate Python sessions (deterministic output).
5. THE `Renderer` SHALL expose a function `generate_plates(n: int, seed: int) -> list[tuple[np.ndarray, str]]` that returns `n` `(image, text)` pairs.
6. WHEN `generate_plates` is called with `n=600` and a fixed `seed`, THE `Renderer` SHALL produce approximately 600 plates with valid Indian registration numbers in the format `AA 00 AA 0000`, using only real `State_Code` values.
7. WHEN `generate_plates` is called, THE `Renderer` SHALL ensure every generated `text` value conforms to the pattern `[State_Code][space][0-9]{2}[space][A-Z]{2}[space][0-9]{4}`.
8. IF `text` does not conform to the expected format, THEN THE `Renderer` SHALL raise a `ValueError` with a descriptive message.

---

### Requirement 2: CCPD Plate Loading (Step 1.2)

**User Story:** As a pipeline user, I want to load and rectify plates from the CCPD dataset, so that I have real-world degraded plates alongside synthetic ones for testing restoration.

#### Acceptance Criteria

1. THE `CCPD_Loader` SHALL expose a function `load_ccpd_plate(image_path: str) -> tuple[np.ndarray, str] | None` that returns a float32 array of shape `(92, 256)` (240×76 plate plus 8 px `Margin`) and the ground-truth text string, or `None` if the file cannot be parsed.
2. WHEN `load_ccpd_plate` is called with a valid CCPD filename, THE `CCPD_Loader` SHALL read the four plate corner coordinates from the filename, compute a perspective-transform matrix using OpenCV, and warp the plate region to exactly 240×76 pixels.
3. WHEN `load_ccpd_plate` is called, THE `CCPD_Loader` SHALL exclude the first (province) character from the returned ground-truth text and SHALL map the remaining six character indices to the CCPD `ALPHABETS`/`ADS` tables.
4. WHEN `load_ccpd_plate` is called, THE `CCPD_Loader` SHALL append the 8 px grey `Margin` (value 128/255) after warping.
5. THE `CCPD_Loader` SHALL expose a function `sample_ccpd(folder: str, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]` that returns a stacked image array, a text array, and a list of filenames.
6. WHEN `sample_ccpd` is called with `n=400` on `ccpd_base` and `n=50` on `ccpd_blur`, THE `CCPD_Loader` SHALL return arrays of exactly those sizes, provided the folder contains at least that many parseable files.
7. IF a CCPD filename is malformed or the image cannot be read, THEN THE `CCPD_Loader` SHALL skip that file and continue processing remaining files without raising an exception.

---

### Requirement 3: Plate Dataset Persistence (Step 1.3)

**User Story:** As a pipeline user, I want all clean plates and their true text stored in a single, versioned location, so that every downstream step reads identical inputs and results are reproducible.

#### Acceptance Criteria

1. THE `Dataset_Writer` SHALL save all plate images and text labels to a single `.npz` file containing arrays keyed as `synthetic_imgs`, `synthetic_text`, `base_imgs`, `base_text`, `blur_imgs`, and `blur_text`.
2. THE `Dataset_Writer` SHALL save a companion `.csv` file with columns `id`, `source`, `true_text`, and `filename`, one row per plate.
3. WHEN the `.npz` file is reloaded with `np.load`, THE `Dataset_Writer` SHALL ensure that every image array is float32 with values in `[0, 1]`.
4. THE `Dataset_Writer` SHALL save a preview image (PNG) containing a grid of 20 plates per source with the `true_text` as the subplot title, for visual inspection.
5. WHEN the preview grid is generated, THE `Dataset_Writer` SHALL use Matplotlib exclusively and SHALL NOT modify the stored plate arrays.

---

### Requirement 4: Motion and Defocus PSF Construction (Step 2.1)

**User Story:** As a pipeline user, I want analytically constructed PSFs for motion and defocus blur, so that the degradation model is physically meaningful and the true PSF can be handed to restoration algorithms.

#### Acceptance Criteria

1. THE `PSF_Builder` SHALL expose a function `make_motion_psf(length: int, angle: float) -> np.ndarray` that returns a float32 2-D array that is odd-sized, centred, and sums to 1.
2. WHEN `make_motion_psf` is called with `length=0`, THE `PSF_Builder` SHALL return a 1×1 delta PSF (value `[[1.0]]`).
3. WHEN `make_motion_psf` is called with `length > 0`, THE `PSF_Builder` SHALL anti-alias the PSF by computing sub-pixel contributions along the motion line at the given `angle` in degrees counter-clockwise from the x-axis.
4. THE `PSF_Builder` SHALL expose a function `make_defocus_psf(radius: float) -> np.ndarray` that returns a float32 2-D array that is odd-sized, centred, and sums to 1.
5. WHEN `make_defocus_psf` is called with `radius=0`, THE `PSF_Builder` SHALL return a 1×1 delta PSF.
6. WHEN `make_defocus_psf` is called with `radius > 0`, THE `PSF_Builder` SHALL fill a circular disc of the given radius with uniform weights, anti-alias the boundary pixels, and normalise so the kernel sums to 1.
7. FOR ALL valid `(length, angle)` inputs, `make_motion_psf(length, angle).sum()` SHALL be within 1e-5 of 1.0.
8. FOR ALL valid `radius` inputs, `make_defocus_psf(radius).sum()` SHALL be within 1e-5 of 1.0.

---

### Requirement 5: FFT-Based Blurring and Downsampling (Step 2.2a)

**User Story:** As a pipeline user, I want to apply blur and downsampling in the frequency domain, so that the degradation matches the forward model and edge artefacts are minimised.

#### Acceptance Criteria

1. THE `Degradation_Pipeline` SHALL expose a function `blur_fft(img: np.ndarray, psf: np.ndarray) -> np.ndarray` that returns a float32 array of the same shape as `img`.
2. WHEN `blur_fft` is called, THE `Degradation_Pipeline` SHALL reflect-pad `img` on all four sides by at least `max(psf.shape) // 2` pixels before computing the FFT.
3. WHEN `blur_fft` is called, THE `Degradation_Pipeline` SHALL embed the PSF in an array of the same size as the padded image, place the PSF centre at pixel `(0, 0)` (top-left corner), and use `np.fft.ifftshift` or equivalent to achieve this shift before multiplying in the frequency domain.
4. WHEN `blur_fft` is called, THE `Degradation_Pipeline` SHALL crop the output back to the original image dimensions after the inverse FFT.
5. WHEN `blur_fft` is called with a delta PSF, THE `Degradation_Pipeline` SHALL return an output that differs from the input by no more than 1e-5 per pixel (float32 rounding only).
6. THE `Degradation_Pipeline` SHALL expose a function `downsample(img: np.ndarray, factor: int) -> np.ndarray` that implements `Block_Mean_Downsampling` without any library call performing the averaging.
7. WHEN `downsample` is called with `factor=1`, THE `Degradation_Pipeline` SHALL return an array identical to the input.
8. WHEN `downsample` is called with `factor=2`, THE `Degradation_Pipeline` SHALL return an array of shape `(H//2, W//2)` where `H, W` are the input dimensions.

---

### Requirement 6: Noise Injection (Step 2.2b)

**User Story:** As a pipeline user, I want to add Gaussian noise at a specified BSNR, so that the degraded image faithfully represents the forward model's noise term and the noise level is reproducible.

#### Acceptance Criteria

1. THE `Degradation_Pipeline` SHALL expose a function `add_noise(img: np.ndarray, bsnr_db: float, seed: int) -> np.ndarray` that returns a float32 array clipped to `[0, 1]`.
2. WHEN `add_noise` is called, THE `Degradation_Pipeline` SHALL compute noise standard deviation as `sigma = sqrt(var(img) / 10^(bsnr_db / 10))`.
3. WHEN `add_noise` is called with the same `(img, bsnr_db, seed)` triple, THE `Degradation_Pipeline` SHALL return the identical pixel array across separate Python sessions.
4. WHEN `add_noise` is called, THE `Degradation_Pipeline` SHALL use a seeded NumPy `default_rng(seed)` and SHALL NOT call any other random source.
5. FOR ALL valid `(img, bsnr_db, seed)` inputs, the measured BSNR of the output SHALL differ from `bsnr_db` by no more than 1 dB.

---

### Requirement 7: Degradation Grid Loop (Step 2.3)

**User Story:** As a pipeline user, I want to systematically degrade every clean plate across a parameter grid, so that downstream restoration algorithms receive a diverse set of degradation conditions with logged parameters.

#### Acceptance Criteria

1. THE `Degradation_Pipeline` SHALL iterate over: blur length in `[0, 21]` px (random integer, uniform), random angle in `[0, 180)` degrees, defocus radius in `[0, 5]` px (random integer, uniform), BSNR in `{40, 30, 25, 20}` dB, and downsample factor in `{1, 2}`.
2. WHEN the grid loop runs, THE `Degradation_Pipeline` SHALL produce a combined PSF by convolving the motion PSF with the defocus PSF (both constructed from scratch) before calling `blur_fft`.
3. WHEN the grid loop runs, THE `Degradation_Pipeline` SHALL log `(plate_id, blur_length, angle, defocus_radius, bsnr_db, downsample_factor, seed)` for every output.
4. WHEN the grid loop runs, THE `Degradation_Pipeline` SHALL save degraded images and associated PSFs in a single `.npz` file or structured directory, keyed by a reproducible identifier.
5. IF a degradation parameter combination results in an output array outside `[0, 1]`, THEN THE `Degradation_Pipeline` SHALL clip the output to `[0, 1]` before saving.

---

### Requirement 8: Otsu Thresholding from Scratch (Step 5.1)

**User Story:** As a pipeline user, I want a from-scratch Otsu threshold that determines the optimal binarization cut-point, so that the binarized plate has maximum character-background separation without relying on library implementations.

#### Acceptance Criteria

1. THE `Binarizer` SHALL expose a function `otsu_threshold(img: np.ndarray) -> float` that returns the threshold value `t` in `[0, 1]` that maximises `Between_Class_Variance` over a 256-bin histogram.
2. WHEN `otsu_threshold` is called, THE `Binarizer` SHALL compute a 256-bin histogram from the float32 image using only NumPy (no `cv2.calcHist` or `np.histogram` as the primary path — cumulative sums must be computed manually).
3. THE `Binarizer` SHALL expose a function `binarize(img: np.ndarray) -> np.ndarray` that returns a uint8 array with text pixels set to 0 and background pixels set to 255 (black text on white), selecting polarity by checking which class has the lower mean.
4. WHEN `binarize` is called, THE `Binarizer` SHALL call `otsu_threshold` internally and SHALL NOT use `cv2.threshold` with the `THRESH_OTSU` flag.
5. FOR ALL plate images where the true threshold `t_cv` is computed with OpenCV as a reference, `|otsu_threshold(img) - t_cv| / 255` SHALL be no more than 1/256 (one histogram bin).

---

### Requirement 9: Morphological Operations from Scratch (Step 5.2)

**User Story:** As a pipeline user, I want erosion, dilation, opening, and closing implemented from scratch, so that I can clean up binarized plates without calling library morphology functions.

#### Acceptance Criteria

1. THE `Binarizer` SHALL expose `erode(img: np.ndarray, kernel: np.ndarray) -> np.ndarray` and `dilate(img: np.ndarray, kernel: np.ndarray) -> np.ndarray`, each accepting a binary uint8 image and a binary uint8 structuring element and returning a binary uint8 image.
2. WHEN `erode` is called, THE `Binarizer` SHALL set an output pixel to 0 (foreground) only when all kernel-covered pixels in the input are 0 (foreground), implementing the morphological erosion definition without `cv2.erode`, `scipy.ndimage`, or `skimage` calls.
3. WHEN `dilate` is called, THE `Binarizer` SHALL set an output pixel to 0 (foreground) when at least one kernel-covered pixel in the input is 0 (foreground), implementing the morphological dilation definition without `cv2.dilate`, `scipy.ndimage`, or `skimage` calls.
4. THE `Binarizer` SHALL expose `opening(img: np.ndarray, kernel: np.ndarray) -> np.ndarray` implemented as `dilate(erode(img, kernel), kernel)`.
5. THE `Binarizer` SHALL expose `closing(img: np.ndarray, kernel: np.ndarray) -> np.ndarray` implemented as `erode(dilate(img, kernel), kernel)`.
6. FOR ALL binary test images, `erode(img, kernel)` output SHALL be pixel-identical to `cv2.erode(img, kernel)` (used as a reference in tests only).
7. FOR ALL binary test images, `dilate(img, kernel)` output SHALL be pixel-identical to `cv2.dilate(img, kernel)` (used as a reference in tests only).

---

### Requirement 10: Character Segmentation (Step 5.3)

**User Story:** As a pipeline user, I want connected-component analysis to extract individual character crops from the binarized plate, so that OCR can operate on isolated characters.

#### Acceptance Criteria

1. THE `Binarizer` SHALL expose a function `segment_characters(binary_plate: np.ndarray) -> list[np.ndarray]` that returns a list of uint8 image crops, sorted left to right by bounding-box x-coordinate.
2. WHEN `segment_characters` is called, THE `Binarizer` SHALL use `cv2.connectedComponentsWithStats` to find connected components and SHALL filter components by: height ≥ 20 % and ≤ 90 % of plate height, aspect ratio (height/width) between 1.0 and 6.0, and area ≥ 50 pixels.
3. WHEN `segment_characters` is called, THE `Binarizer` SHALL reject the largest connected component (plate background or border) before applying the character-shape filters.
4. WHEN `segment_characters` returns crops, each crop SHALL be the tight bounding-box region extracted from `binary_plate`.
5. IF `segment_characters` finds fewer than 4 or more than 10 components after filtering, THEN THE `Binarizer` SHALL still return the filtered list without raising an exception, and the caller is responsible for handling unexpected counts.
