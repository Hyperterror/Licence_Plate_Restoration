# Step 4.1 — Restoration Foundation

**Owner:** Ashish (2026909001) — steps 4 and 8
**Module:** `license_plate_restoration/restoration.py`
**Tests:** `tests/test_restoration.py`
**Scope of this phase:** the machinery every restoration algorithm needs. Inverse
filtering, Wiener, constrained least squares (CLS) and Richardson–Lucy are
**not** implemented here; they are step 4.2 and they will all sit on top of the
four helpers described below.

This document is written so it can be presented directly in the viva: every
equation that the code evaluates is stated, and every claim about the dataset
was verified against the code in this repository.

---

## 1. Where restoration sits in the pipeline

The forward model implemented by steps 1–2 is

$$
g(x,y) \;=\; D\big(h(x,y) * f(x,y)\big) + n(x,y)
$$

| Symbol | Meaning | Produced by |
|---|---|---|
| $f$ | sharp plate (ground truth) | step 1 (`renderer`, `ccpd_loader`) |
| $h$ | point spread function (PSF) | step 2.1 (`psf_builder`) |
| $*$ | 2-D convolution | step 2.2 (`degradation.blur_fft`) |
| $D$ | block-mean decimation by 1 or 2 | step 2.2 (`degradation.downsample`) |
| $n$ | additive white Gaussian noise at a target BSNR | step 2.2 (`degradation.add_noise`) |
| $g$ | the degraded plate we are given | `degraded.npz` |

Restoration is the *inverse problem*: given $g$ and $h$, estimate $f$. Two facts
make it hard and both show up in this phase:

1. $h$ removes information. A motion smear of width $2L+1$ maps every edge onto
   a line of that length; the corresponding transfer function has near-zeros
   (notches) at frequencies the blur destroyed. Dividing by a near-zero puts
   back noise, not detail.
2. The FFT is periodic. A plain frequency-domain division wraps the right edge
   onto the left edge, and every real plate has a hard brightness step at its
   border, so the wrap is enormous.

The foundation therefore contains: **validation** (so we never silently
deconvolve the wrong pair), the **frequency response** (so a PSF becomes a
pointwise filter), **edge tapering** (so the periodic wrap is not the dominant
feature of the image), and the **numerical helpers** (so the remaining
divisions stay finite and the output stays inside the image convention).

---

## 2. What a PSF is

A point spread function is *the image of a single point of light*. Shine a
point through a lens that is out of focus, or hold the camera still for 30 ms
while it moves, and the point becomes a disc or a line — that shape **is** $h$.

Consequences, and they are exactly the conventions enforced by
`validate_psf`:

* **Non-negative.** Light is added, not subtracted: $h(m,n) \ge 0$.
* **Unit sum.** A point carries unit energy and the blur only redistributes it:

  $$\sum_{m}\sum_{n} h(m,n) = 1 .$$

  This is what keeps the mean brightness of $g$ equal to the mean of $f$, i.e.
  $H(0,0) = 1$.
* **Centred.** The centre of mass must sit on the geometric centre, otherwise
  the blur also *translates* the image:

  $$\sum_{m,n} m\,h(m,n) = c_r \sum_{m,n} h(m,n), \qquad
    \sum_{m,n} n\,h(m,n) = c_c \sum_{m,n} h(m,n).$$

  An odd-sized kernel has an exact centre pixel $(c_r, c_c) = (L, L)$, which is
  why `make_motion_psf` builds $(2L+1)\times(2L+1)$ kernels.
* **Resolution-matched.** $h$ is only valid for the image it was built for. A
  factor-2 image needs a factor-2 kernel; `degraded.npz` stores exactly that.

Blur of a *whole* image is the sum of the blurred images of its points, which
is why the forward model is a convolution:

$$
g(i,j) \;=\; \sum_{m}\sum_{n} h(m,n)\, f(i-m,\, j-n) \;+\; n(i,j).
$$

Two kernels used in this project:

* **Motion / camera shake**, a line at angle $\theta$ (degrees, CCW from $+x$;
  rows grow downwards so the direction vector is $(\cos\theta, -\sin\theta)$).
* **Defocus**, a uniform disc of radius $r$ (the image of a point through a
  circular aperture without focus).

`combine_psfs` convolves the two kernels, because applying both blurs in
sequence means convolving their PSFs — and, in the frequency domain,
*multiplying* their transfer functions.

---

## 3. From a PSF to a frequency response

### 3.1 Why convert at all

Convolution is expensive and, worse, it is not invertible pointwise. In the
discrete Fourier domain the convolution theorem holds *exactly* for the
periodic (circular) case:

$$
G(u,v) \;=\; H(u,v)\, F(u,v).
$$

One complex multiplication per coefficient, and — crucially — inversion is now
a **division** of two numbers per coefficient instead of an integral equation.
All four restoration algorithms live in this domain (Richardson–Lucy alternates
between it and the spatial domain). The cost is $O(MN\log MN)$ per transform.

$H$ is what `psf_frequency_response` returns.

### 3.2 The three steps the code performs

For an image of shape $(M, N)$ and a kernel of shape $(K_r, K_c)$:

**Step 1 — zero-pad the kernel to the image period.** Circulant convolution is
defined on an array of period $M \times N$, so the kernel must be sampled on
that same period. The kernel is written into the top-left corner of a zero
array of shape $(M, N)$.

**Step 2 — move the centre to the FFT origin.** This is the step that is easy
to get wrong, and the reason is the shift theorem:

$$
\text{if } h_0 = \delta_{\,d} \quad\Longrightarrow\quad
H(u,v) = e^{-2\pi i (ud/M + vd/N)}, \qquad
\text{and } \mathcal{F}^{-1}\{H F\}(i,j) = f(i-d,\; j-d).
$$

So a kernel whose mass sits at index $d$ **moves the image by $+d$**. In the
padded array the kernel's centre is at $(K_r/2,\, K_c/2)$, so leaving it there
would shift the restoration by half a kernel — up to 21 px for the largest
grid kernel, which is bigger than a whole glyph. Rolling the padded array by
$(-K_r/2,\,-K_c/2)$ puts the centre at index $(0,0)$, and then $H$ carries
**no linear phase** at all:

$$
h_0 \text{ centred at } 0 \quad\Longrightarrow\quad
H(u,v) = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} h_0(m,n)\,
e^{-2\pi i\left(\frac{um}{M} + \frac{vn}{N}\right)} .
$$

**Step 3 — transform** with `np.fft.fft2`, the same convention
`degradation.blur_fft` uses (full complex `fft2`, not `rfft2`), so step 4 and
step 2 cannot disagree about what "frequency" means.

### 3.3 Properties of $H$ you can quote

* $H(0,0) = \sum h = 1$: DC gain 1, mean brightness preserved. Pinned by
  `test_response_dc_gain_is_one_for_a_normalised_psf`.
* $h$ real and symmetric $\Rightarrow$ $H$ real and even: a symmetric blur adds
  no phase, only attenuates.
* $|H(u,v)| \le \sum|h| = 1$: a normalised PSF can never amplify a frequency.
  It can only *attenuate*, and where it attenuates almost completely
  (motion notches, defocus rings) the ratio $G/H$ explodes. The measured
  minimum of $|H|$ for the 7×7 motion kernel in the demo is
  $2.1\times10^{-6}$ — that single number is the whole argument for
  regularisation in step 4.2.

---

## 4. Circular convolution, and why the border misbehaves

A DFT treats the image as **one period of a periodic signal**. Convolution
with it is therefore *circular*:

$$
g_p(i,j) \;=\; \sum_{m}\sum_{n} h_0(m,n)\, f_p\big((i-m)\bmod M,\; (j-n)\bmod N\big).
$$

Content that leaves the right edge does not disappear: it re-enters on the
left. For a *blur* that is a mild nuisance; step 2.2 avoids it by reflect
padding before the FFT (`blur_fft`). For a *deconvolution* it is fatal, for two
reasons:

1. **The seam is a huge discontinuity.** A plate ends on its 8 px grey margin
   ($\approx 0.5$) and the opposite border is tacked onto it. The periodic
   extension therefore has a step of order $0.5$ at the seam, whose spectrum
   decays only like $1/f$.
2. **Division amplifies exactly what the seam contributes.** The ringing caused
   by that step is multiplied by $1/H$, and $H$ is tiny at high frequencies,
   so a border artefact becomes a bright ripple that propagates across the whole
   plate and buries the glyphs. PSNR collapses even though the interior
   information is still there.

This is the classic *ringing / Gibbs* artefact of frequency-domain
deconvolution, and it is the reason the pipeline scores metrics on the interior
only (see `report_ishit.md` §10, rule 3: crop the 8 px margin before scoring).

---

## 5. Edge tapering

### 5.1 What it does

`taper_edges` multiplies the observation by a window that is 1 in the middle
and falls smoothly to 0 at the border, **before** the FFT:

$$
\tilde g(y,x) \;=\; W(y,x)\, g(y,x).
$$

The periodic extension of $\tilde g$ is now continuous — both sides of the seam
are exactly 0 — so the step disappears and the wrap no longer injects a
$1/f$ ramp into the spectrum.

### 5.2 The window used

A separable raised-cosine (Tukey) window. With $d_i = \min(i,\; n-1-i)$ the
distance of sample $i$ from the nearer edge and $W$ the taper width in pixels:

$$
w(i) \;=\; \tfrac{1}{2}\left(1 - \cos\frac{\pi\,\min(d_i, W)}{W}\right),
\qquad
W(y,x) \;=\; w_{\text{rows}}(y)\; w_{\text{cols}}(x).
$$

Properties, each of which is pinned by a test:

* $w(0) = 0$ — the outermost line is removed, so the seam value is 0 on both
  sides.
* $w = 1$ for $d_i \ge W$ — the window has a **unity plateau**, so the central
  region (where the registration is) is multiplied by exactly 1 and is
  **bit-for-bit unchanged** (`assert np.array_equal` in
  `test_taper_does_not_alter_the_central_plate_region`).
* zero slope at $d_i = W$ — no visible seam where the ramp meets the plateau.
* `width` is in **pixels, on every side**, not a per-axis fraction: plates are
  very anisotropic ($96\times336$), and a per-axis fraction would fade 34 px
  horizontally but 10 px vertically. The default is
  `fraction=0.1` of the *shorter* side: 10 px at factor 1, 5 px at factor 2.

### 5.3 Limitations — state these out loud in the viva

* **The border is gone, not restored.** Tapered pixels carry almost no signal,
  so a restored border is meaningless. Only the interior may be scored or run
  through binarisation.
* **Tapering is not part of the forward model.** The model says
  $g = h*f + n$; the window multiplies the *output* of that model, so near the
  border $\tilde g$ is no longer a sample of $g$. That is an approximation we
  accept in exchange for removing a much larger artefact.
* **It does not replace regularisation.** Windowed ringing at $|H|\approx0$ is
  still present; the taper only removes the *boundary-driven* component.
  Wiener/CLS still need their $K$ or $\lambda$, and RL still needs its iteration
  count.
* **It must not be applied to the reference.** Taper both $g$ and the image you
  compare against only if you want a windowed metric; otherwise taper $g$,
  restore, and compare the interior against the untouched $f$.
* **A wide taper eats the plate.** Glyphs are ~32 px tall at factor 1 and
  ~16 px at factor 2; the taper must stay well below that, which is why it is
  measured against the short side.

---

## 6. Shared numerical helpers

| Helper | What it does | Why it is needed |
|---|---|---|
| `validate_image` | 2-D, non-empty, finite, in $[0,1]$; returns a `float32` copy | the whole pipeline assumes this contract |
| `validate_psf` | 2-D, finite, non-negative (to tolerance), unit sum, odd by default; **never renormalises** | a stored kernel must be consumed as-is |
| `validate_image_shape` | `(rows, cols)` of positive integers | no hard-coded $96\times336$ |
| `psf_frequency_response` | pad, roll centre to origin, `fft2` | turns $h$ into $H$ |
| `edge_taper_window`, `taper_edges` | raised-cosine window / tapered image | removes the periodic seam |
| `safe_divide` | $a/\tilde b$ with $\tilde b = \operatorname{sign}(b)\max(|b|,\varepsilon)$ | $H$ has real near-zeros |
| `near_zero_mask` | which coefficients were clamped | makes the clamp **observable**, not hidden |
| `clip_unit_interval` | clip to $[0,1]$, cast to `float32`, refuse NaN/inf | overshoot is real in inverse/RL output |

Two policies worth defending:

* **We never renormalise a stored PSF.** If $\sum h$ is off by more than
  $10^{-5}$ that is a bug upstream, and rescaling it here would silently change
  the brightness of every restored plate. The error message says so.
* **We never swallow a numerical failure.** `clip_unit_interval` raises on
  NaN/inf instead of letting `np.clip` pass it through; `safe_divide` raises on
  non-finite input and reports clamped coefficients through
  `near_zero_mask`. The only silent repair is the $\varepsilon$ floor, and it is
  countable.

---

## 7. How step 4.2 will use these helpers

All four algorithms share the same preamble:

```python
g      = validate_image(degraded)                    # float32, [0, 1]
H      = psf_frequency_response(psf, g.shape)        # psf straight from the npz
window = edge_taper_window(g.shape, fraction=0.1)
g_t    = taper_edges(g, fraction=0.1)
```

and they differ only in the filter applied to $G = \mathcal{F}\{g_t\}$:

| Algorithm | Filter | What it needs from the foundation |
|---|---|---|
| **Inverse** | $\hat F = \dfrac{G}{H}$ | $H$ with no phase offset; an $\varepsilon$ floor, because $H$ has true zeros |
| **Wiener** | $\hat F = \dfrac{H^{*}}{|H|^{2} + K}\,G$ | $H$ and $H^{*}$ from the same grid; $K \approx$ noise-to-signal ratio $1/\text{BSNR}$ |
| **CLS** | $\hat F = \dfrac{H^{*}}{|H|^{2} + \lambda\,|C|^{2}}\,G$ | same $H$; $C$ is the DFT of the Laplacian, so the same FFT convention |
| **Richardson–Lucy** | $f^{t+1} = f^{t}\left(h^{*} * \dfrac{g}{h * f^{t}}\right)$ | $h$ centred (a PSF that shifts would drag the estimate); `safe_divide`; `clip_unit_interval` for the non-negativity constraint |

then

```python
restored = clip_unit_interval(np.real(np.fft.ifft2(filtered)))
```

The three things every one of them depends on and this phase fixes:

1. $H$ is built at the **image's own** resolution, with the kernel centred, so
   no algorithm has to remember a shift.
2. The image handed to the FFT has **no periodic seam**, so the artefact they
   fight is the one caused by $h$'s zeros and not by the border.
3. The output is `float32` in $[0,1]$ and validated, so `binarize` (step 5) and
   the metrics (step 8) can consume it without special cases.

---

## 8. Verified conventions, and one resolved discrepancy

### 8.1 Motion PSF: `length` is a **half**-length (verified, unchanged)

A hand-over message claims `length` is the full smear extent. The code says
otherwise, and the code wins:

* `psf_builder.make_motion_psf` samples the line with
  `samples = np.arange(-length, length + _LINE_STEP, _LINE_STEP)` and builds a
  kernel of size `2 * length + 1` — mass exists on both sides of the centre, so
  the smear is symmetric with half-width `length`.
* `tests/test_psf_builder.py::test_motion_psf_shape_follows_length` asserts
  `psf.shape == (2 * length + 1, 2 * length + 1)`.
* `tests/test_psf_builder.py::test_motion_psf_is_anti_aliased` calls
  `make_motion_psf(2, 0.0)` a "4 px smear" and checks the five-tap profile.

Measured, to remove all doubt:

| `length` | kernel | non-zero columns at $0°$ | smear |
|---|---|---|---|
| 1 | 3×3 | 0…2 | 3 px |
| 3 | 7×7 | 0…6 | 7 px |
| 15 | 31×31 | 0…30 | 31 px |
| 21 | 43×43 | 0…42 | 43 px |

So `length=15` is a **31 px** smear and the grid maximum `length=21` is a
**43 px** smear. The convention is documented here, **not changed**: altering
it would invalidate `degraded.npz` and every number in
`report_ishit.md` §7.4. For step 8 plots the axis must read
`blur length L (kernel 2L+1, smear 2L+1 px)`.

### 8.2 Factor-2 stored kernels are sometimes **even**-sized (verified)

`run_degradation_grid` stores `effective_psf(combined, factor)`. For the grid's
own ranges, `combine_psfs` gives an odd $(2(L+r)+1)\times(2(L+r)+1)$ kernel, and
`effective_psf` then decimates it by 2, giving a side of $L+r$ — **even
whenever $L+r$ is even**. Measured over 1050 draws from the grid's own
distributions: **556 of 1050** factor-2 kernels have an even side, and they
carry negative weights of order $10^{-17}$ caused by the FFT round-off inside
`effective_psf`.

This matters for step 4 because the contract says "use the stored `_psf`
as-is". Therefore:

* `validate_psf(..., require_odd=True)` is the default (the `psf_builder`
  contract), but `psf_frequency_response` calls it with `require_odd=False`,
  because it must accept the dataset.
* an even side has no exact centre pixel, so the centre is taken at
  `size // 2`; the restoration is then offset by **at most half a pixel** —
  the same quantisation `effective_psf` already documents.
* negative weights down to $-10^{-6}$ are accepted as round-off; a genuinely
  negative kernel is refused.

### 8.3 Conventions inherited without change

| Rule | Where it comes from |
|---|---|
| images `float32` in $[0,1]$, 0 = ink | `__init__.py`, `report_ishit.md` §5 |
| PSF sums to 1, centred | `psf_builder`, `report_ishit.md` §5 |
| FFT convention = full `np.fft.fft2` | `degradation.blur_fft` |
| stored PSF used as-is, matched to its key | `report_ishit.md` §10 |
| no SciPy / scikit-image / torch | `tests/test_integration.py` AST guards |

---

## 9. Verification

`python -m pytest tests/test_restoration.py -q` → **66 passed**.
`python -m pytest -q` (whole suite) → **165 passed, 6 skipped** (the 6 skips are
the CCPD tests that need the external `CCPD2019/` folder).

What the tests actually demonstrate, beyond "it runs":

| Claim | Test |
|---|---|
| a delta PSF is the identity filter ($H\equiv1$) | `test_delta_psf_response_is_unity`, `test_delta_psf_leaves_the_image_unchanged` |
| $H$ has the image's shape and is complex | `test_frequency_response_has_the_image_shape` |
| a centred PSF does not shift (and a naive pad *does*) | `test_centred_psf_does_not_introduce_a_spatial_shift` |
| the response equals a spatial convolution, coefficient for coefficient | `test_response_matches_direct_spatial_convolution` |
| the convention matches `blur_fft` from step 2.2 | `test_response_agrees_with_blur_fft_on_a_zero_bordered_plate` |
| variable sizes and both dataset resolutions | `..._image_shape`, `test_both_dataset_resolutions_work_with_matching_kernels` |
| stored even-sized factor-2 kernels are accepted | `test_response_accepts_stored_factor_two_kernels` |
| malformed kernels are rejected with specific messages | `test_validate_psf_rejects_malformed_kernels` |
| tapering keeps shape, dtype and range | `test_taper_preserves_shape_and_finiteness` |
| tapering leaves the plateau bit-identical | `test_taper_does_not_alter_the_central_plate_region` |
| invalid parameters raise | `test_taper_rejects_invalid_parameters`, `test_numerical_helpers_reject_invalid_input` |

`python -m license_plate_restoration.restoration` prints the demo numbers
quoted in §3.3 on a real rendered plate.

---

## 10. Viva quick answers

**Q: What is a PSF, in one sentence?**
The image of a point source — the kernel $h$ of the convolution model.

**Q: Why is the PSF normalised?**
Because blur conserves energy: $\sum h = 1 \Rightarrow H(0,0) = 1 \Rightarrow$
mean brightness is preserved. A kernel summing to 0.9 would darken the plate.

**Q: Why do we need $H$ instead of $h$?**
The convolution theorem turns convolution into multiplication, so
deconvolution becomes a per-coefficient division instead of an integral
equation — and later a *regularised* division.

**Q: Why move the PSF centre to the FFT origin?**
`np.fft.ifft2(fft2(f) * fft2(h))` computes a *circular* convolution in which a
kernel centred at index $d$ shifts the image by $+d$. Rolling the padded kernel
by $-(K/2)$ removes that shift and makes $H$ free of linear phase.

**Q: What would happen if you forgot the roll?**
The restored plate would be displaced by half the kernel — up to 21 px for the
43 px grid maximum, i.e. more than a glyph width. The test
`test_centred_psf_does_not_introduce_a_spatial_shift` measures exactly that
offset for the naive version.

**Q: What is circular convolution?**
Convolution in which the image is treated as one period of a periodic signal,
so indices are taken modulo the size and content wraps from the right edge to
the left.

**Q: Why does the border ring?**
Because the periodic extension has a discontinuity at the seam (bright margin
against bright margin, plus a jump to 0 outside), which injects a $1/f$
spectrum that the $1/H$ division amplifies into a ripple.

**Q: What does the taper do, mathematically?**
Multiplies $g$ by a separable raised-cosine window that is 0 on the border and
1 on the interior plateau, making the periodic extension continuous so the seam
contributes nothing.

**Q: Why a raised cosine and not a rectangle?**
A rectangle is the identity (no taper). A raised cosine has zero slope at both
ends of the ramp, so it introduces no visible seam and attenuates the spectrum
as little as possible while still reaching 0.

**Q: Doesn't the window destroy information?**
Yes, near the border — that is the trade. The interior is untouched
(exactly 1.0 multiplier), and the interior is what is scored.

**Q: Why can a plain inverse filter not work?**
$|H|$ has genuine near-zeros (measured $2\times10^{-6}$ here), so $G/H$
amplifies noise by $1/|H|$ and produces garbage. That is exactly why step 4.2
adds $K$ (Wiener) or $\lambda|C|^2$ (CLS) to the denominator.

**Q: Is `length` in the motion PSF the smear or half the smear?**
Half. `length=15` builds a 31×31 kernel, i.e. a 31 px smear; the grid maximum
`length=21` is 43 px. Verified against the implementation and pinned by the
step-2 tests — see §8.1.
