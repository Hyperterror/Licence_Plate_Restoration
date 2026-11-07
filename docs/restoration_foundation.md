# Step 4 — Restoration: the foundation (4.1) and inverse filtering (4.2)

**Owner:** Ashish (2026909001) — steps 4 and 8
**Module:** `license_plate_restoration/restoration.py`
**Tests:** `tests/test_restoration.py` (foundation, 66),
`tests/test_restoration_audit.py` (phase 4.1.1 audit, 58),
`tests/test_inverse_filter.py` (phase 4.2, 39)
**Phase 4.1.1:** complex-valued arithmetic (§6.1), even-sided kernels (§3.4,
§8.2), edge-tapering measurements (§4, §5.4). The audit log in §11 records what
changed and what did not.
**Phase 4.2:** `inverse_filter`, the frequency-domain inverse filter (§12), with
the taper and floor measurements that decided its defaults (§12.3) and the log
in §13.
**Not implemented here:** Wiener, constrained least squares (CLS) and
Richardson–Lucy. They are steps 4.3–4.5 and they sit on exactly the same four
helpers (§7).

This document is written so it can be presented directly in the viva: every
equation that the code evaluates is stated, and every claim about the dataset
was verified against the code in this repository. Every number marked
*measured* comes from `python -m license_plate_restoration.restoration` or from
the test that pins it.

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

### 3.4 Even-sided kernels: the half pixel the index rule cannot express

`blur_fft` and `psf_frequency_response(centre="pixel")` both put index
`size // 2` of the kernel on the FFT origin. For an **odd** kernel that is the
true centre. For an **even** kernel there is no centre sample, and
`effective_psf`'s 2×2 block decimation puts the kernel's mass at

$$
c \;=\; \frac{K-1}{2} \;=\; \left\lfloor \frac{K}{2} \right\rfloor - \frac{1}{2},
$$

that is exactly **half a pixel** up and to the left of the index the roll uses.
That is a real displacement, not a rounding detail: rolling by `K // 2` leaves
the mass at $-0.5$, so the convolution moves the content $0.5$ px up and left.

A shift of $\delta$ along an axis is a linear phase ramp in the frequency
domain, so the response can be re-aligned without touching the stored kernel:

$$
R(u,v) \;=\; e^{\,2\pi i\left(\delta_y \frac{u}{M} \;+\; \delta_x \frac{v}{N}\right)},
\qquad
\delta \;=\; \text{centre of mass} - \left\lfloor K/2 \right\rfloor .
$$

$R$ has **unit modulus**, so it changes phase only: $|H|$, the DC gain and the
`.npz` file are all untouched, and the FFT keeps its convention.
`psf_frequency_response(centre="auto")` — the default since phase 4.1.1 —
applies the ramp; `centre="pixel"` reproduces the pre-audit behaviour and
`psf_centre_offset` returns $\delta$ so a caller can log the alignment.

**Controlled experiment** (`tests/test_restoration_audit.py`): blur a plate with
the full-resolution kernel, decimate by 2 — exactly what
`run_degradation_grid` does — then push the decimated *reference* back through
the *stored* kernel and compare intensity centroids. With the right convention
the two land on the same pixel.

| stored kernel | parity | $\delta$ (px) | shift, `auto` | shift, `pixel` |
|---|---|---|---|---|
| 15×15 | odd | $+0.000$ | $+0.003$ | $+0.003$ |
| 21×21 | odd | $+0.000$ | $-0.001$ | $-0.001$ |
| 18×18 | even | $-0.500$ | $+0.003$ | $-0.497$ |
| 12×12 | even | $-0.500$ | $+0.002$ | $-0.497$ |
| 8×8 | even | $-0.500$ | $+0.002$ | $-0.497$ |

Shifts in pixels, rows and columns agreeing to $\pm 0.01$; configurations
$(L, r) \in \{(15,0), (15,3), (5,3), (11,1), (21,0), (3,3)\}$ at 20°.

Two consequences worth stating out loud:

* **Odd kernels are unaffected.** Measured over 600 odd kernels spanning the
grid's own ranges, the mass sits within **0.005 px** of `size // 2`
(rasterisation asymmetry only), so `centre="auto"` returns a response that is
`bit-identical` to `centre="pixel"` — `np.array_equal`, not `allclose`.
* **Even kernels were displaced by half a pixel.** That is
$0.5/48 \approx 1\%$ of a factor-2 glyph: invisible in a figure, but a
systematic registration error that would have shown up in every step-8 PSNR as
a fixed penalty on the ~53 % of low-resolution images whose kernel is even.
`CENTRE_OFFSET_TOL = 0.05` px is what separates the two families: their
measured offsets are $\le 0.005$ px and $\approx 0.5$ px, so nothing lands in
between and the threshold can snap the first family to zero without ever
touching the second.

**Limitation.** For a *genuinely* translated kernel the ramp cancels that
translation too. That is deliberate — a restoration must not move the plate —
but it means a PSF estimated in step 3 must follow the "`size // 2` is the
centre" convention; `centre="pixel"` is there if an intentional offset has to
survive.

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

1. **The seam is a discontinuity.** A bright border tacked onto a dark outside
   — or onto a border of a different brightness — is a step whose spectrum
   decays only like $1/f$.
2. **Division amplifies exactly what the seam contributes.** The ringing caused
   by that step is multiplied by $1/H$, and $H$ is tiny at high frequencies,
   so a border artefact becomes a bright ripple that propagates across the whole
   plate and buries the glyphs. PSNR collapses even though the interior
   information is still there.

**Measured — the correction this audit made.** Item 1 is a worst case, not a
description of this dataset. All **1050** images in `plates.npz` (600 synthetic,
400 CCPD base, 50 CCPD blurred) carry a **uniform 8 px grey margin**, mean corner
value **0.5020**, so the right edge and the left edge are *identical* and the
measured wrap step is **0.0000 on every one of them**. The only discontinuity the
wrap creates here is the one the noise adds: **0.118** at 20 dB BSNR. §5.4
measures what that costs, and it is not what item 1 predicts.

Item 2 still stands, and it is the reason a *small* seam is worth thinking
about: the division multiplies whatever is there by $1/|H|$, measured down to
$2.1\times10^{-6}$ on the demo kernel, i.e. up to a factor $5\times10^{5}$.

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

### 5.4 Measured comparison: none, Tukey, and the two alternatives

`edge_taper_window(..., profile=...)` exposes four **named arms** — `"none"`,
`"tukey"` (default), `"hann"`, `"linear"` — so the comparison below is
reproducible on the project's own plate generator instead of on a textbook
picture. Representative case: `renderer.generate_plates(1, seed=7)[0]`,
$96\times336$, default ramp (`fraction=0.1` of the short side: 10 px at factor 1,
5 px at factor 2), motion kernel `make_motion_psf(3, 8°)` (7×7).

#### What each window costs

`python -m license_plate_restoration.restoration` prints this table.

| window | mean loss $\overline{1-W}$ | value at the centre | slope jump at $d=W$ | area with $W<0.99$ | min $W$ over ink |
|---|---|---|---|---|---|
| `none` | $0.00\%$ | $1.0000$ | $0$ | $0$ | $1.000$ |
| **`tukey`** (current) | $14.36\%$ | $1.0000$ | $0.047$ | $25.5\%$ | $0.818$ |
| `linear` | $14.36\%$ | $1.0000$ | $0.100$ | $25.5\%$ | $0.818$ |
| `hann` | $75.33\%$ | $0.9997$ | $0.0002$ | $100\%$ | $\approx 0$ |

* **Mean loss** is the mean of $1-W$: the fraction of a flat field the window
  removes. Tukey and linear cost the *same* because they enclose the same area
  under the ramp; hann costs five times as much because it has no plateau.
* **Slope jump** is $\max|\Delta^{2}w|$ along the middle row. Tukey's is $0.047$
  rather than $0$ because on a discrete grid the step from $w(W-1)$ to
  $w(W)=1$ is not exactly flat — zero slope holds in the continuum — and a
  20× reduction over `linear` is what "smooth" actually buys.
* **min $W$ over ink** is the number that matters for a plate. The first ink row
  and column of a synthetic plate is 8, so a 10 px ramp does reach ink — but
  only the two dark frame rows around the drawing area, at $W \ge 0.818$. Every
  glyph stroke sits inside the plateau at exactly $W = 1$. Measured over shapes
  and factors: $(96,336)$ and $(48,168)$ → $0.818$; the CCPD shape $(92,256)$
  gets a 9 px ramp → $0.941$. The drawn ink is *not* destroyed, and no
  character is faded at all.
* **Untouched area** is $74.5\%$ for the default ramp (24016 of 32256 pixels on
  a $96\times336$ plate), i.e. the window attenuates a quarter of the frame.

#### What each window does to the deconvolution problem

The window can change a **restored interior pixel** through exactly one route:
the plateau is $W\equiv1$ (ramp 10 px, scored region starts 23 px in), so the
window cannot bias an interior sample directly. What it can do is change the
*model mismatch* that the $1/|H|$ stage amplifies. That mismatch is

$$
E_w \;=\; \mathcal{F}^{-1}\big\{ \mathcal{F}\{W_w f\}\,H \big\} \;-\; W_w\,(h * f),
$$

the disagreement between the circular model of the windowed plate and the true
(reflect-padded) blur. The table pushes $E_w$ through $1/\max(|H|,\varepsilon)$
and reports the RMS that lands in the interior ($>23$ px from any edge), on the
clean plate:

| kernel | window | $\operatorname{rms} E_w$ | leak, $\varepsilon=10^{-3}$ | leak, $\varepsilon=10^{-9}$ |
|---|---|---|---|---|
| 7×7 ($L{=}3$, $r{=}0$) | `none` | $2.0\times10^{-8}$ | $1.2\times10^{-6}$ | $4.5\times10^{-5}$ |
| | `tukey` | $6.5\times10^{-3}$ | $1.16\times10^{-1}$ | $2.9\times10^{+1}$ |
| | `linear` | $5.4\times10^{-3}$ | $1.37\times10^{-1}$ | $2.5\times10^{+1}$ |
| | `hann` | $1.4\times10^{-3}$ | $2.4\times10^{-2}$ | $2.0$ |
| 43×43 ($L{=}21$, $r{=}0$) | `none` | $4.9\times10^{-3}$ | $7.1\times10^{-1}$ | $8.1$ |
| | `tukey` | $7.6\times10^{-2}$ | $2.6$ | $3.3$ |
| | `linear` | $7.0\times10^{-2}$ | $2.5$ | $3.9$ |
| | `hann` | $9.6\times10^{-3}$ | $4.4\times10^{-1}$ | $1.4$ |

At 20 dB BSNR the 7×7 row of `none` becomes $3.1\times10^{-2}$ /
$5.7\times10^{-2}$ / $5.7\times10^{-2}$ and `tukey` $3.0\times10^{-2}$ /
$1.3\times10^{-1}$ / $2.9\times10^{+1}$: noise lifts the untapered mismatch from
round-off to the taper's own order, and `none` stays the better arm even then.

**Three findings, one of them uncomfortable.**

1. **For a kernel narrower than the margin the wrap is already exact.** `none`
   reproduces `blur_fft` to float32 round-off ($2.0\times10^{-8}$). With a
   uniform margin, the wrapped samples and the reflect-padded samples hold the
   *same* value, so there is no seam to remove — and every taper then only
   *adds* error, `tukey` by five orders of magnitude.
2. **A taper pays only when the kernel is wider than the margin.** At 43×43
   (radius 21 > 8 px) the untapered mismatch is real, $4.9\times10^{-3}$, and
   under near-pure inversion it leaks $8.1$; `tukey` leaks $3.3$, `linear`
   $3.9$. That is the regime the window was written for — the grid's largest
   kernel — and the only regime in which these measurements support it.
3. **Hann is the only arm that touches the interior** ($2.0\times10^{-3}$ clean),
   which is the same fact as "hann has no plateau" and rules it out on
   principle, before any PSNR is computed.

#### Decision (and what it is not)

The default stays **`profile="tukey"`** and the alternatives are **named arms**,
not copies of the function:

* the foundation must be safe for *any* input, including a frame cropped at the
  plate edge and a 43 px kernel — the two conditions under which the measurement
  shows the ramp earning its keep;
* the scored interior is identical for `none` and `tukey` by construction apart
  from the leak term, so step 8's metric does not change when step 4.2 picks a
  different arm;
* switching is one keyword, and the quantity that should decide it — restored
  PSNR against $f$ — belongs to step 4.2.

**What step 4.2 then measured.** §12.3 runs the ablation on a real plate with
the inverse filter, and the answer is unambiguous: the untapered arm wins
everywhere, by four orders of magnitude at the tight floor ($3.3\times10^{-5}$
against $0.70$ RMS for a $7\times7$ kernel, clean input) and by 4× at the loose
one even for the 51×51 kernel that is *wider* than the margin ($0.188$
against $0.753$). So `inverse_filter` defaults to `profile="none"`
(`DEFAULT_INVERSE_TAPER`), while `DEFAULT_TAPER_PROFILE` stays `"tukey"` for the
rest of the module. The reason the prediction above ("the ramp is insurance for
kernels wider than the margin") did not survive contact with the metric is
also in §12.3: at a gain of $1/\varepsilon$ the residual the ramp *creates* is
larger than the wrap it removes, and both are multiplied by the same
$1/|H|$.

What these numbers **do** forbid: applying `hann` to a plate, and claiming in
the viva that the taper "removes a huge seam" on this dataset. The defensible
statement is *"the 8 px margin already makes the wrap continuous; the ramp is
insurance for kernels wider than the margin and for noise-driven wrap, and it
costs a measured $8.7\times10^{-3}$ of border model error in exchange."*

---

## 6. Shared numerical helpers

| Helper | What it does | Why it is needed |
|---|---|---|
| `validate_image` | 2-D, non-empty, finite, in $[0,1]$; returns a `float32` copy | the whole pipeline assumes this contract |
| `validate_psf` | 2-D, finite, non-negative (to tolerance), unit sum, odd by default; **never renormalises** | a stored kernel must be consumed as-is |
| `validate_image_shape` | `(rows, cols)` of positive integers | no hard-coded $96\times336$ |
| `psf_frequency_response` | pad, roll the centre to the origin, `fft2`, optional unit-modulus phase ramp (`centre="auto"`) | turns $h$ into $H$ and removes the even-kernel half pixel |
| `psf_centre_offset` | sub-pixel offset of the centre of mass from `size // 2` | makes the alignment of §3.4 **observable** instead of implicit |
| `edge_taper_window`, `taper_edges` | taper window / tapered image, four profiles (`none`, `tukey`, `hann`, `linear`) | removes the periodic seam, and provides the untapered control |
| `safe_divide` | $a/\tilde b$ with $\tilde b = \varepsilon\,b/\lvert b\rvert$ where $\lvert b\rvert \le \varepsilon$ | $H$ has **complex** near-zeros |
| `near_zero_mask` | which coefficients were clamped, decided on $\lvert b\rvert$ | makes the clamp **observable**, not hidden |
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

### 6.1 Complex arithmetic — the audit's first finding

The pre-audit helpers did

```python
array = np.asarray(values, dtype=np.float64)     # drops the imaginary part
```

which is precisely the defect the audit was looking for. $H$ is complex, so that
cast emits a `ComplexWarning` and keeps only $\operatorname{Re}$: a division then
returns *real garbage* ($\operatorname{Re}(G)/\operatorname{Re}(H)$), and the
near-zero test compared $\operatorname{Re}(b)$ with $\varepsilon$ instead of the
modulus, so it never fired. Both were reproduced through the public API before
any fix (`safe_divide` returned `float64`; `near_zero_mask` returned all-`False`
on a spectrum full of notches).

What the helpers do now — every row has a test in
`tests/test_restoration_audit.py`:

| Call | Before | Now |
|---|---|---|
| `safe_divide` on complex spectra | `float64`, imaginary part dropped | `complex128`; $\operatorname{Re}\mathcal{F}^{-1}$ recovers the image to $5\times10^{-8}$ |
| near-zero test | $\operatorname{Re}(b) \le \varepsilon$ | $\lvert b\rvert \le \varepsilon$ |
| clamp | $\operatorname{sign}(b)\max(\lvert b\rvert,\varepsilon)$ | $\varepsilon\, b/\lvert b\rvert$ — the floor is **rotated onto the denominator's own phase** |
| $b = 0$ exactly | $\operatorname{sign}(0)=0 \Rightarrow \infty$ | phase falls back to $+1$, so the quotient is bounded by $\lvert a\rvert/\varepsilon$ |
| real input | — | unchanged: the real branch reduces to $\operatorname{sign}(b)\max(\lvert b\rvert,\varepsilon)$ |
| complex input to a real-only helper | silently truncated | `ValueError` naming the contract (`validate_image`, `validate_psf`, `clip_unit_interval`, `taper_edges`) |

Why the phase-preserving floor, rather than a magnitude-only clamp: replacing a
near-zero $H(u,v)$ by $+\varepsilon$ injects a $\pi$ phase error wherever $H$
pointed the other way, and a $\pi$ error across a ring of frequencies *is* a sign
flip in the spatial domain — visible as grain. Keeping the phase means the
clamped coefficient contributes nothing new: it is the same filter with its gain
limited, so

$$
\arg\!\left(\frac{G}{\tilde H}\right) = \arg G - \arg H
$$

holds for every coefficient, floored or not. $|H|$, the DC gain and the stored
`.npz` are untouched — only the magnitude at the notches is bounded.
`_as_numeric_array` (real *or* complex) and `_as_float_array` (real only, raises
on complex) are the only two entry points, so no public helper has a path that
can truncate silently.

The demo prints two numbers that make this checkable in one command: the
quotient's dtype together with $\max|\operatorname{Im}\mathcal{F}^{-1}\{G/H\}|$
($4.96\times10^{-8}$ for real data and a real kernel — i.e. the imaginary part
*cancels* rather than being discarded), and the phase of a deliberately floored
coefficient ($+0.407149$ rad, matching $\arg(1+2i) - 0.7$ exactly).

---

## 7. The preamble all four algorithms share

All four algorithms share the same preamble, and step 4.2 packaged the first
row of the table below as `inverse_filter` rather than leaving it to be
re-typed:

```python
g      = validate_image(degraded)                    # float32, [0, 1]
H      = psf_frequency_response(psf, g.shape)        # psf straight from the npz
window = edge_taper_window(g.shape, fraction=0.1)
g_t    = taper_edges(g, fraction=0.1)
```

`psf_frequency_response` defaults to `centre="auto"` (§3.4), so `H` is already
aligned with the stored kernel's mass, and `profile=` selects the taper arm
(§5.4) — both defaults are overridable, which is what makes step 4.2's ablation a
two-keyword experiment.

The algorithms differ only in the filter applied to $G = \mathcal{F}\{g_t\}$:

| Algorithm | Filter | What it needs from the foundation |
|---|---|---|
| **Inverse** (§12, implemented) | $\hat F = \dfrac{G}{H}$ | $H$ with no phase offset; an $\varepsilon$ floor, because $H$ has true zeros |
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
* an even side has no exact centre pixel, so `effective_psf` puts the mass at
  $(K-1)/2$ while the roll uses `size // 2`. That is a **real 0.5 px
  displacement**, not a rounding detail: measured by re-blurring the decimated
  reference through the stored kernel, the result lands $0.497$ px up/left of the
  stored image (§3.4). Since phase 4.1.1 `psf_frequency_response` therefore
  defaults to `centre="auto"`, which removes exactly that offset with a
  unit-modulus phase ramp; the stored kernel itself is consumed unchanged, and an
  odd kernel's response is bit-identical (`np.array_equal`) to the old behaviour.
* the tolerance that separates the two families is `CENTRE_OFFSET_TOL = 0.05` px.
  Measured over 792 kernels: every odd kernel lands within **0.0049 px** of
  `size // 2` (rasterisation asymmetry of the anti-aliased line) and every even
  factor-2 kernel within **0.0024 px** of a full half pixel. The interval between
  0.005 and 0.495 is empty, so snapping the small offsets to 0 can never touch an
  even kernel — and `psf_centre_offset` reports which family a kernel is in.
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
`python -m pytest tests/test_restoration_audit.py -q` → **58 passed**.
`python -m pytest tests/test_inverse_filter.py -q` → **39 passed**.
`python -m pytest -q` (whole suite) → **262 passed, 6 skipped** (the 6 skips are
the CCPD tests that need the external `CCPD2019/` folder). Before the phase
4.1.1 audit the suite gave 165 passed / 6 skipped, and before phase 4.2 it gave
218 / 6; **no existing test was removed, relaxed or skipped** to make room for
the new ones. The two new helper tests in the audit file exist because phase
4.2 used an infinite floor and an already-accepted infinite `eps` turned the
quotient into `inf * phase` — the fix is in `_floor_value`, shared by
`safe_divide` and `near_zero_mask`, so the two can no longer disagree.

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

And the audit's own claims, all in `tests/test_restoration_audit.py`:

| Claim | Test |
|---|---|
| complex input survives `safe_divide` with its phase | `test_safe_divide_keeps_complex_quotients`, `test_safe_divide_never_truncates_a_complex_operand` |
| a clamped coefficient keeps the denominator's phase | `test_safe_divide_preserves_the_phase_of_a_floored_coefficient` |
| a degenerate quotient stays bounded | `test_safe_divide_bounds_a_degenerate_complex_quotient` |
| near-zero detection uses $\lvert b\rvert$, not $\operatorname{Re}(b)$ | `test_near_zero_mask_measures_the_modulus_of_a_complex_spectrum` |
| real-only helpers refuse complex data instead of truncating | `test_real_only_helpers_reject_complex_input` |
| the inverse quotient of real data stays Hermitian | `test_inverse_filter_quotient_of_real_data_stays_hermitian` |
| the even-kernel half pixel is real and `auto` removes it | `test_auto_centre_removes_the_stored_kernel_displacement`, `test_full_resolution_kernels_have_no_displacement` |
| the offset tolerance separates the two families | `test_centre_offset_tolerance_separates_the_two_families`, `test_psf_centre_offset_follows_the_kernel_parity` |
| the correction is phase-only (property test) | `test_property_auto_centre_is_phase_only`, `test_centre_correction_is_phase_only` |
| the taper arms are distinct, safe and attributable | `test_none_profile_is_the_identity_window`, `test_hann_profile_has_no_plateau`, `test_linear_profile_keeps_the_plateau_but_jumps_in_slope`, `test_taper_rejects_profile_misuse` |
| the default ramp never fades a glyph stroke | `test_default_taper_ramp_leaves_the_glyph_strokes_untouched` |

And phase 4.2's claims, all in `tests/test_inverse_filter.py` unless marked
*audit*:

| Claim | Test |
|---|---|
| $H\equiv1$ is an exact identity, not merely close | `test_delta_psf_returns_the_image_unchanged` |
| a noise-free blur is undone to $4.7\times10^{-6}$ RMS | `test_noise_free_blur_is_inverted_exactly` |
| an odd `psf_builder` kernel restores a plate to $3.3\times10^{-5}$ | `test_odd_motion_psf_restores_a_plate` |
| an even kernel lands on the reference ($0.0000$ px); the pixel rule lands $+0.5000$ px off | `test_even_sized_psf_is_restored_without_a_half_pixel_shift` |
| the stored factor-2 pair stays aligned ($0.003$ px drift against $0.176$) | `test_even_stored_kernel_keeps_the_decimated_plate_aligned` |
| a complex $H$ is divided by in full ($1.4\times10^{-2}$ against $2.7\times10^{5}$ when only $\operatorname{Re}H$ is used) | `test_complex_frequency_response_is_not_truncated` |
| a response with 79 exact zeros still returns an image | `test_near_zero_frequency_response_still_returns_an_image` |
| the floor trades accuracy against noise in both directions | `test_floor_trades_restoration_against_noise_amplification` |
| an unusable floor is refused by both helpers | `test_helpers_refuse_a_floor_that_cannot_be_used`, `test_safe_divide_rejects_a_floor_that_is_not_a_number` (audit) |
| the output is always a finite `float32` image in $[0,1]$ | `test_property_output_is_always_a_valid_image`, `test_output_keeps_the_image_convention` |
| the taper only moves the border | `test_taper_choice_only_changes_the_border` |
| bad images, kernels and parameters raise | `test_inverse_filter_rejects_invalid_input` |

`python -m license_plate_restoration.restoration` prints the demo numbers
quoted in §3.3, §3.4, §5.4, §6.1, §12.3 and §12.5 on a real rendered plate —
the DC gain, the $|H|$ minimum, the agreement with `blur_fft`, the parity/offset
table, the complex-quotient check, the four-window table, the inverse-filter
table (kernel × noise × taper × floor), the floor sweep that locates the
20 dB optimum, and the factor-2 centroid drift.

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
offset for the naive version, and `test_auto_centre_removes_the_stored_kernel_displacement`
measures the residual half pixel that the roll alone cannot remove.

**Q: What is circular convolution?**
Convolution in which the image is treated as one period of a periodic signal,
so indices are taken modulo the size and content wraps from the right edge to
the left.

**Q: Why does the border ring?**
Because the periodic extension has a discontinuity at the seam, which injects a
$1/f$ spectrum that the $1/H$ division amplifies into a ripple. On *this*
dataset the step is essentially absent — all 1050 plates carry the same uniform
8 px margin, so opposite borders are identical and the measured wrap step is
0.0000 — but the noise sitting on those two border rows still wraps, and $1/|H|$
does not care that the input was small.

**Q: Your helpers take complex spectra. Where could the imaginary part get lost?**
In a `np.asarray(x, dtype=np.float64)` cast, which emits `ComplexWarning` and
keeps only the real part — exactly what `safe_divide` used to do, so `G/H` came
back real and wrong. Now `_as_numeric_array` keeps `complex128`, the floor is
applied to $|b|$ and rotated onto $\arg b$, and a complex array handed to a
real-only helper (an image, a PSF, a clipped result) raises instead of being
truncated.

**Q: What does the even-sized kernel actually do wrong?**
Nothing at the level of the stored file: `effective_psf` centres its mass at
$(K-1)/2$, half a pixel left/up of `size // 2`, so the old roll left the kernel's
mass 0.5 px off the FFT origin and the convolution dragged the content by that
amount. Measured: $0.497$ px. `centre="auto"` multiplies $H$ by the unit-modulus
ramp $\exp(2\pi i(\delta_y u/M + \delta_x v/N))$ and the displacement disappears,
without touching `degraded.npz`.

**Q: Why is the taper still the default if the measurement says `none` is more
accurate?**
Because on this dataset the taper costs accuracy and buys nothing — that is the
honest finding of §5.4 — while on an input *without* a margin it is the
difference between a usable restoration and a ringing mess, and the grid
contains 43 px kernels that are wider than the 8 px margin. The default stays
conservative, `profile="none"` is a named arm for step 4.2's ablation, and the
decision is made where it can be measured: on restored PSNR.

**Q: What does the taper do, mathematically?**
Multiplies $g$ by a separable raised-cosine window that is 0 on the border and
1 on the interior plateau, making the periodic extension continuous so the seam
contributes nothing.

**Q: Why a raised cosine and not a rectangle?**
A rectangle is the identity (no taper). A raised cosine has zero slope at both
ends of the ramp, so it introduces no visible seam and attenuates the spectrum
as little as possible while still reaching 0. Measured on a real plate: 14.4 % of
the mean intensity, over a 25.5 % border area, while every glyph stroke stays at
exactly 1.0.

**Q: Doesn't the window destroy information?**
Yes, near the border — that is the trade. The interior is untouched
(exactly 1.0 multiplier), and the interior is what is scored.

**Q: What is inverse filtering?**
$\hat F = G/H$ — one complex division per frequency coefficient, then an
inverse FFT (§12.1). It is the *closed-form* solution of the noise-free model,
not an approximation to it: if $g = h*f$ exactly and $H$ has no zeros, the
restored image is the original image.

**Q: Then why is it not the final answer?**
Because $|H|$ has genuine near-zeros (measured $2.1\times10^{-6}$ for a 7×7 and
$1.1\times10^{-9}$ for a 51×51 kernel), and the error it produces is
$N/|H|$ plus a model-residual term over the same $|H|$. Dividing by a
near-zero restores noise, not detail. That is why steps 4.3–4.5 add $K$
(Wiener) or $\lambda|C|^2$ (CLS) to the denominator, or iterate instead of
dividing (Richardson–Lucy).

**Q: So what stops the division from exploding?**
The floor $\varepsilon$: wherever $|H|\le\varepsilon$ the denominator is
replaced by $\varepsilon\,H/|H|$, so no coefficient is divided by less than
$\varepsilon$ and the gain is capped at $1/\varepsilon$. Measured: $0$, $2779$
and $19136$ of 32256 coefficients are floored for the 7×7 kernel at
$\varepsilon = 10^{-12}, 10^{-2}, 10^{-1}$ (§12.3). The floor keeps $\arg H$, so
the clamped coefficient adds no phase error (§6.1).

**Q: How do you choose $\varepsilon$?**
It should match the noise-to-signal amplitude ratio of the input, which is why
the 20 dB rows improve monotonically as the floor rises to $10^{-1}$ and the
clean rows get worse as it rises past $10^{-12}$ (§12.3). A floor below the
input's own quantisation level is worse than useless: float32 stores 0.5 to
$6\times10^{-8}$, and $6\times10^{-8}\times8.9\times10^{8} \gg 1$, which is why
the *noise-free* 51×51 case needs $\varepsilon\ge10^{-2}$ to beat the blur.

**Q: Which is the honest headline result of this phase?**
That the inverse filter is a baseline, not a solution. For a small kernel on
clean input it is spectacular — the 7×7 blur's interior error
$1.1\times10^{-1}$ falls to $3.3\times10^{-5}$, a factor 3400 — and for a
kernel whose nulls sit below the input's quantisation it is *worse than doing
nothing* ($3.2\times10^{-1}$ blurred against $7.0\times10^{-1}$ restored,
§12.3, §12.6).

**Q: Why does `inverse_filter` default to `profile="none"` when the module
default is `tukey`?**
Because it was measured, not assumed. On this dataset the ramp does not remove
a seam (the 8 px margin already made the wrap continuous) but it does create a
border model residual, and the inverse filter multiplies that residual by the
same $1/|H|$ — 0.70 RMS against $3.3\times10^{-5}$ for a 7×7 kernel, and 4×
worse even for a 51×51 kernel that is wider than the margin (§12.3). The global
`DEFAULT_TAPER_PROFILE` is untouched.

**Q: Why does the even-kernel half pixel still matter when there is a floor?**
Because the floor bounds the *magnitude* of the response, not its *phase*, and
a 0.5 px displacement is a unit-modulus phase ramp. No $\varepsilon$ removes
it. Measured end-to-end through the filter: `centre="auto"` lands the blob on
the reference ($0.0000$ px, RMSE $1.1\times10^{-6}$), `centre="pixel"` lands it
$+0.5000$ px off (RMSE $2.8\times10^{-2}$) — §12.5.

**Q: Is `length` in the motion PSF the smear or half the smear?**
Half. `length=15` builds a 31×31 kernel, i.e. a 31 px smear; the grid maximum
`length=21` is 43 px. Verified against the implementation and pinned by the
step-2 tests — see §8.1.

---

## 11. Audit log (phase 4.1.1)

Four questions were asked of the foundation before anything was built on top of
it. Each is answered above; this table is the index, and it records what was
*changed* as opposed to merely *measured*.

| # | Question | Finding | Evidence | Action |
|---|---|---|---|---|
| 1 | Can the helpers handle a complex $H(u,v)$? | **No** — `np.asarray(..., float64)` dropped $\operatorname{Im}$, so `safe_divide` returned `float64` and `near_zero_mask` never fired | reproduced through the public API; `ComplexWarning` | helpers take real *or* complex; modulus floor with the denominator's phase; real-only helpers raise (§6.1) |
| 2 | Do even-sized PSFs displace the image? | **Yes** — $0.497$ px up/left on every even factor-2 kernel (~53 % of the low-resolution dataset) | re-blur the decimated reference through the stored kernel; centroid offset | `centre="auto"` (default) applies the unit-modulus ramp; `psf_centre_offset` exposes it; odd kernels bit-identical (§3.4) |
| 3 | Is the edge taper earning its keep? | **Not on this dataset** — the 8 px margin already makes the wrap continuous (seam 0.0000 on 1050/1050 images), so the ramp only adds a $8.7\times10^{-3}$ border model error; it *does* pay for 43 px kernels under near-pure inversion | seam survey + model-error/leak tables | current default preserved, `profile="none"` added as a first-class arm, choice deferred to step 4.2 with this evidence (§5.4) |
| 4 | Did the audit touch the data or the teammate modules? | **No** | `git diff` on `data/`, `psf_builder`, `degradation`, `renderer`, `ccpd_loader`: empty | none |

**Unresolved, and deliberately so.**

* `effective_psf` still emits even-sided kernels; changing it would invalidate
  `degraded.npz` and every number in `report_ishit.md` §7.4, so the alignment is
  corrected on the *consumer* side instead. That is a team decision, not an
  audit decision.
* The real `degraded.npz` (471 MB, gitignored) was not present, so every number
  here was measured on the generative pipeline (`plates.npz`, `blur_fft`,
  `effective_psf`) rather than on the stored degraded images. The forward model
  is the same code, but this is a limitation worth stating.
* The taper's effect on restored accuracy could not be settled by the audit
  alone; §5.4 measured the mechanism and deferred the decision to a metric.
  **Step 4.2 settled it** (§12.3): interior RMSE against $f$, over two kernels,
  two noise levels, three floors and two windows, and the untapered arm wins
  everywhere. The formal PSNR/SSIM comparison is still step 8's, because this
  repository has no implementation of either (§12.6).
* With `eps` now validated in one place (`_floor_value`), a floor given as
  `np.float32` is compared and multiplied as the same number in `safe_divide`
  and `near_zero_mask`; before, a string floor passed the mask and then crashed
  on `eps * phase`.
* `report_ishit.md` §5 still says the PSFs are odd-sized, which is true of
  `psf_builder` and false of the stored factor-2 kernels (§8.2). That file is
  teammate-owned and was not edited; the discrepancy is reported here instead.

`python -m pytest -q` → **218 passed, 6 skipped** at the end of the audit, and
**262 passed, 6 skipped** after step 4.2 (§13).

---

## 12. Inverse filtering (step 4.2)

### 12.1 The algorithm

The forward model of §1, evaluated at the image's own resolution, is
$g = h*f + n$, and by the convolution theorem of §3.1 every DFT coefficient
satisfies

$$
G(u,v) \;=\; H(u,v)\,F(u,v) \;+\; N(u,v).
$$

If there were no noise and $H$ had no zeros, the answer would be exact and
would be one division per coefficient:

$$
\hat F(u,v) \;=\; \frac{G(u,v)}{H(u,v)},
\qquad
\hat f \;=\; \operatorname{Re}\,\mathcal{F}^{-1}\{\hat F\} .
$$

That is the whole algorithm. Everything else in `inverse_filter` exists to make
that division **legal on a real file**: the kernel has to be validated and
centred, the image has to be given a continuous periodic extension,
denominators that are numerically zero have to be floored, and the result has
to come back in the project's image convention.

```python
inverse_filter(image, psf, eps=DEFAULT_EPS, width=None,
               fraction=DEFAULT_TAPER_FRACTION, profile=DEFAULT_INVERSE_TAPER,
               centre=DEFAULT_CENTRE_MODE, require_odd=False, tol=PSF_SUM_TOL)
```

| Parameter | Meaning | Default |
|---|---|---|
| `image` | the degraded plate, 2-D `float32` in $[0,1]$ | required |
| `psf` | the kernel **as stored**, resolution-matched, summing to 1 | required |
| `eps` | denominator floor $\varepsilon$: finite and $>0$ | $10^{-12}$ |
| `width` | taper width in pixels, on every side | `None` → `fraction` |
| `fraction` | taper width as a fraction of the shorter side | $0.1$ |
| `profile` | `none`, `tukey`, `hann` or `linear` | `none` |
| `centre` | `auto` (stored mass on the FFT origin) or `pixel` | `auto` |
| `require_odd` | refuse an even-sided kernel | `False` |
| `tol` | unit-sum tolerance for the kernel | $10^{-5}$ |

`eps` is the only parameter that changes the *algorithm*; the others change the
preprocessing (§12.4) or the geometry (§12.5). Both defaults that could have
been guessed were **measured** instead: the floor is the textbook
"guard against a zero" value, and the taper arm is the one that wins §12.3.

### 12.2 Why $1/H$ is ill-conditioned, and what the floor does

Per coefficient the estimate is

$$
\hat F - F \;=\; \frac{N + E}{H},
\qquad\text{so}\qquad
\bigl|\hat F - F\bigr| \;\le\; \frac{|N| + |E|}{|H|},
$$

where $E$ is everything the model cannot explain: the wrap discontinuity, the
residual a taper leaves behind, and the fact that the input is stored in
`float32`. The error at a frequency is the input error there **divided by the
gain the blur applied**. That is the whole difficulty — $H$ attenuates, the
division un-attenuates, and it un-attenuates the noise exactly as faithfully as
the signal. Measured on the two demo kernels, $|H|$ bottoms out at
$2.1\times10^{-6}$ (7×7) and $1.1\times10^{-9}$ (51×51), i.e. gains of
$4.8\times10^{5}$ and $8.9\times10^{8}$.

The floor replaces the denominator wherever it is smaller than $\varepsilon$:

$$
\tilde H(u,v) =
\begin{cases}
H(u,v), & |H(u,v)| > \varepsilon,\\[2pt]
\varepsilon\,\dfrac{H(u,v)}{|H(u,v)|}, & |H(u,v)| \le \varepsilon,
\end{cases}
\qquad
\left|\frac{G}{\tilde H}\right| \;\le\; \frac{|G|}{\varepsilon} .
$$

Three consequences worth saying out loud in the viva:

* **The gain is capped at $1/\varepsilon$** — 240 dB, 40 dB and 20 dB for
  $\varepsilon = 10^{-12}, 10^{-2}, 10^{-1}$. A floor of $10^{-1}$ means "no
  coefficient may be amplified by more than 20 dB", which is *exactly* the BSNR
  of this phase's noisy experiment — and §12.3's sweep puts that experiment's
  optimum at $\varepsilon \approx 0.3$, the same order of magnitude. The knob
  and the channel have a shared unit.
* **The phase is preserved** ($\arg\tilde H = \arg H$), so a floored
  coefficient still contributes $\arg G - \arg H$ and no $\pi$ flips appear in
  the output (§6.1).
* **The floor is a global dial, not a per-coefficient rule.** It cannot tell
  where the signal is strong, so it trades the restoration of *every* small
  coefficient for the suppression of noise at *some* of them. Wiener (§7)
  replaces the flat $\varepsilon$ by $|H|^2/(|H|^2+K)$ — the same idea, made
  frequency-selective and applied to the numerator as well.

A floor below the input's own quantisation level is worse than useless, and
this is the sharpest fact in the phase. A `float32` near 0.5 is stored to
$6.0\times10^{-8}$, and

$$
6.0\times10^{-8} \;\times\; 8.9\times10^{8} \;\approx\; 53 ,
$$

so for the 51×51 kernel the *round-off of a noise-free file* is amplified to
order 1 by the unfloored division. That is why the noise-free 51×51 row of
§12.3 needs $\varepsilon \ge 10^{-2}$ to beat the blur at all: the floor is not
there for the noise you can see, it is there for the noise the file format
put there.

### 12.3 The measured trade-off

`python -m license_plate_restoration.restoration` prints this table and every
number in it.

**Setup.** `renderer.generate_plates(1, seed=7)[0][0]` — a $96\times336$ plate
at factor 1. Two kernels, both from the grid's own builders at 8°:
$\texttt{combine}(\texttt{motion}(3), \texttt{defocus}(0))$, a $7\times7$
kernel with $|H|_{\min} = 2.073\times10^{-6}$, and
$\texttt{combine}(\texttt{motion}(21), \texttt{defocus}(4))$, a $51\times51$
kernel with $|H|_{\min} = 1.127\times10^{-9}$. The degraded image is
`blur_fft(plate, kernel)` — the step-2.2 forward model, reflect-padded; the
noisy one is `add_noise(blurred, 20.0, seed=3)`. The score is the **interior
RMSE** against the clean plate, 12 px margin, so the border the taper touches
is excluded from every number. For reference, doing nothing scores
$1.106\times10^{-1}$ (7×7) and $3.164\times10^{-1}$ (51×51) on clean input.

| kernel | input | `none` $10^{-12}$ | `tukey` $10^{-12}$ | `none` $10^{-2}$ | `tukey` $10^{-2}$ | `none` $10^{-1}$ | `tukey` $10^{-1}$ |
|---|---|---|---|---|---|---|---|
| 7×7 | clean | $3.3\times10^{-5}$ | $7.0\times10^{-1}$ | $8.5\times10^{-3}$ | $3.7\times10^{-2}$ | $3.3\times10^{-2}$ | $3.4\times10^{-2}$ |
| 7×7 | 20 dB | $7.0\times10^{-1}$ | $7.0\times10^{-1}$ | $5.4\times10^{-1}$ | $5.4\times10^{-1}$ | $1.8\times10^{-1}$ | $1.8\times10^{-1}$ |
| 51×51 | clean | $7.0\times10^{-1}$ | $6.8\times10^{-1}$ | $1.9\times10^{-1}$ | $7.5\times10^{-1}$ | $1.8\times10^{-1}$ | $3.8\times10^{-1}$ |
| 51×51 | 20 dB | $7.0\times10^{-1}$ | $7.0\times10^{-1}$ | $6.0\times10^{-1}$ | $6.8\times10^{-1}$ | $2.4\times10^{-1}$ | $4.0\times10^{-1}$ |

and how much of the spectrum each floor touches (of 32256 coefficients):

| $\varepsilon$ | max gain | 7×7 floored | 51×51 floored |
|---|---|---|---|
| $10^{-12}$ | $10^{+12}$ (240 dB) | 0 | 0 |
| $10^{-2}$ | $10^{+2}$ (40 dB) | 2779 (8.6 %) | 29995 (93.0 %) |
| $10^{-1}$ | $10^{+1}$ (20 dB) | 19136 (59.3 %) | 31779 (98.5 %) |

**What the table says.**

1. **The floor has a genuine interior optimum, and the noise level is what
   sets it.** For the 7×7 kernel on clean input accuracy is *best* at the
   tightest floor and degrades monotonically as the floor rises
   ($3.3\times10^{-5} \to 8.5\times10^{-3} \to 3.3\times10^{-2}$): every
   floored coefficient is resolution thrown away. At 20 dB BSNR the same sweep
   falls and then turns around — the demo prints that line too:

   ```
   floor sweep  (7, 7) at 20 dB BSNR -> 0.01:5.442e-01 0.1:1.816e-01 0.3:8.938e-02 1:1.068e-01
   floor sweep (51, 51) at 20 dB BSNR -> 0.01:5.967e-01 0.1:2.370e-01 0.3:2.468e-01 1:2.914e-01
   ```

   a real minimum, at $\varepsilon \approx 0.3$ for the 7×7 kernel and
   $\varepsilon \le 0.1$ for the 51×51 one. A floor of 1 is worse again,
   because by then the filter has stopped restoring anything. The 20 dB
   channel's noise-to-signal amplitude is $10^{-20/20} = 0.1$, so the optimum
   sits within a small factor of $N/S$: the parameter has a *physical*
   reading, not merely a legal range, and it moves when the noise level moves.
   That is precisely the quantity a Wiener filter derives from the stored BSNR
   instead of leaving to the caller (§7).
2. **Too small a floor is worse than doing nothing.** The 51×51 kernel on
   clean input restores to $7.0\times10^{-1}$ at $\varepsilon=10^{-12}$, while
   the blurred image itself scores $3.2\times10^{-1}$. The unfloored division
   turned float32 round-off into a plate-wide ripple. This is the honest
   counter-example to "inverse filtering works": it does not, once $|H|$ falls
   below the input's noise or quantisation.
3. **The taper loses on every row it does not tie.** `tukey` costs four orders
   of magnitude at the tight floor (0.70 against
   $3.3\times10^{-5}$) because the ramp *creates* a border residual that the
   same $1/|H|$ amplifies — and at $\varepsilon=10^{-2}$ it costs 4× even for
   the 51×51 kernel that is wider than the 8 px margin
   ($7.5\times10^{-1}$ against $1.9\times10^{-1}$), which was exactly the case
   §5.4 predicted the ramp would win. It ties (to 1 %) only in the 7×7 20 dB
   rows at $\varepsilon=10^{-1}$, where 59 % of the spectrum is already
   floored and the border residual is floored with it. Hence
   `DEFAULT_INVERSE_TAPER = "none"`.
4. **A floored filter still restores.** At $\varepsilon=10^{-1}$, 59 % (7×7) and
   98.5 % (51×51) of the coefficients are clamped, yet the output still beats
   the blurred input by 3.3× and 1.7× respectively: the well-conditioned low
   frequencies — the ones that carry the plate — are recovered exactly, and only
   the destroyed high frequencies are given up. That is the correct way to
   describe what a floor does: it keeps the part of the spectrum the blur did
   not annihilate.

A second, smaller experiment pins the same mechanism on synthetic data inside
`tests/test_inverse_filter.py`: a $40\times40$ texture blurred with a 5×5
Gaussian inverts to $4.7\times10^{-6}$ clean against a blur error of
$1.3\times10^{-1}$ (a factor $2.8\times10^{4}$), while at 20 dB BSNR the same
pair scores $4.4\times10^{-1}$ at $\varepsilon=10^{-12}$ and
$9.2\times10^{-2}$ at $\varepsilon=10^{-1}$ — the same reversal, on data small
enough to check by hand.

### 12.4 The exact preprocessing applied before the FFT

In the order the code performs it:

1. **`validate_image(image)`** — 2-D, non-empty, finite, inside $[0,1]$;
   returns a `float32` copy. A complex or out-of-range array raises here, not
   three lines later inside the FFT.
2. **`psf_frequency_response(psf, image.shape, tol=tol,
   require_odd=require_odd, centre=centre)`** — validates the stored kernel
   (shape, finiteness, non-negativity to $10^{-6}$, unit sum to $10^{-5}$,
   **never renormalised**), zero-pads it to the image period, rolls
   `size // 2` onto index $(0,0)$, applies a unit-modulus phase ramp when
   `centre="auto"` measures a sub-pixel offset (§3.4), and transforms with
   `np.fft.fft2`. Returns `complex128` with the image's shape.
3. **`taper_edges(degraded, width=..., fraction=..., profile=...)`** —
   multiplies by the separable window of §5.2. For `profile="none"` the
   window is identically 1, so this step is a copy and nothing is tapered;
   that is the arm the default selects and the arm §12.3 wins with.
4. **`np.fft.fft2(windowed.astype(np.float64))`** — the full `fft2` on the
   image period, the same convention as `degradation.blur_fft`, evaluated in
   double precision although the image is stored in `float32`.
5. **`safe_divide(spectrum, response, eps=eps)`** — the division of §12.2:
   `complex128` in, `complex128` out, modulus-floored, phase-preserving.
6. **`np.real(np.fft.ifft2(estimate))`** — for a real image and a real kernel
   the quotient keeps Hermitian symmetry, so the inverse transform is real to
   round-off; measured $\max|\operatorname{Im}| = 5.0\times10^{-8}$, and
   `test_inverse_filter_quotient_of_real_data_stays_hermitian` pins it.
7. **`clip_unit_interval(restored)`** — clip to $[0,1]$, cast to `float32`, and
   **raise** on NaN/inf. The filter never repairs a non-finite value and never
   substitutes a plausible pixel.

What is deliberately **not** done, because each of these would be a silent lie:

* no resizing, resampling or cropping, and no re-deriving of the kernel — the
  stored `_psf` is consumed at the size it was written;
* no renormalisation, sign flip or re-centring of the stored *array*: the
  centring is a phase ramp on $H$, not an edit of $h$;
* no padding beyond the zero-pad to the image period inside step 2 (and no
  reflect padding, which belongs to the forward `blur_fft` and would make the
  restoration model differ from the one it inverts);
* no post-hoc brightness or contrast matching against a reference, which would
  be test-set leakage;
* no parity assumption: odd and even kernels both go through step 2 — the even
  case is what §12.5 measures.

### 12.5 Even-sized PSFs: the half pixel a floor cannot remove

§3.4 established that `effective_psf` puts an even kernel's mass at
$(K-1)/2$, half a pixel up/left of the index the plain rule centres, and that
`centre="auto"` corrects it with a unit-modulus ramp. Run end-to-end through
the filter, that geometry is unchanged by the divide:

| experiment | `centre` | centroid drift | interior RMSE |
|---|---|---|---|
| blob at row 20.3, 8×8 Gaussian, $\varepsilon = 10^{-3}$ | `auto` | $0.0000$ px | $1.1\times10^{-6}$ |
| | `pixel` | $+0.5000$ px | $2.8\times10^{-2}$ |
| stored factor-2 pair (18×18), $\varepsilon = 10^{-2}$ | `auto` | $+0.0027$ px | $0.080$ |
| | `pixel` | $+0.1762$ px | $0.201$ |

**Why a bigger floor cannot fix this.** A translation is a *phase* ramp of
unit modulus (the shift theorem of §3.2), so $|H|$ is identical under both
centring rules and the floor — which acts on $|H|$ alone — cannot see the
difference. The only repair is a phase correction, which is why it lives in
the construction of $H$ and not in the division. The synthetic row is the
clean demonstration (zero drift to 4 decimals against a full half pixel); in
the stored-pair row the displacement is partly absorbed by the clipping and by
the decimated plate's own centroid, which is why the *RMSE ratio* (2.5×) is the
sturdier signature there than the centroid. Two tests pin it:
`test_even_sized_psf_is_restored_without_a_half_pixel_shift` and
`test_even_stored_kernel_keeps_the_decimated_plate_aligned`.

### 12.6 Limitations — say these out loud

1. **It is not universally effective, and this document does not claim it is.**
   The 51×51 kernel on clean input restores *worse than the blur it was given*
   ($7.0\times10^{-1}$ against $3.2\times10^{-1}$, §12.3).
2. **It is a baseline, not a regulariser.** The floor bounds the gain; it
   cannot distinguish signal from noise, so it cannot beat a filter that uses
   the noise statistics (Wiener, step 4.3).
3. **It cannot recover what $H$ destroyed.** At a notch the information is
   gone; the floor substitutes a bounded, wrong value. No downstream step
   recovers it — that is why binarisation (step 5) and metrics (step 8) must not
   expect detail the blur removed.
4. **Clipping is an extra nonlinearity inside the measured error.** The
   restored plate is clipped to $[0,1]$ before scoring, so the RMSE above
   includes the clipping error. That is deliberate (the image convention is
   non-negotiable) but it means the numbers are not the pure linear-filter
   error.
5. **The border is meaningless under a taper.** Every number here is an
   interior crop; a tapered restored frame must never be scored or binarised
   (§5.3).
6. **No PSNR or SSIM is reported in this phase.** This repository has no
   implementation of either and step 8 owns them; adding one here would
   duplicate a teammate's deliverable and silently define the team's metric.
   The metric used is interior RMSE against $f$, and RMSE and PSNR are monotone
   in each other for a fixed peak, so none of the orderings above depends on
   the choice.
7. **The experiments use the generative pipeline, not `degraded.npz`.** The
   stored grid (471 MB) is gitignored and absent from this checkout; the
   forward model here is the same step-2.2 code applied to `plates.npz`.
8. **Two kernels, one plate, one seed per condition.** Enough to pin the
   defaults and demonstrate the mechanism; not a survey. Step 8 runs the grid.
9. **Wiener, CLS and Richardson–Lucy are not implemented.** They are steps
   4.3–4.5 and they need nothing that this phase did not build (§7).

---

## 13. Phase 4.2 log

| File | Change | Tests |
|---|---|---|
| `license_plate_restoration/restoration.py` | `inverse_filter` (+ the same preamble every later algorithm shares); `DEFAULT_INVERSE_TAPER`; `_floor_value`, now the single validator behind `near_zero_mask` and `safe_divide`; module docstring; demo table extended with the floor sweep | existing 124 tests on the module unchanged |
| `tests/test_inverse_filter.py` | **new**, 39 tests, ten required categories plus a Hypothesis property | 39 |
| `tests/test_restoration_audit.py` | two tests for the floor validator ($\varepsilon = \infty$, NaN, $0$, negative, and a non-numeric floor) | 58 |
| `docs/restoration_foundation.md` | §5.4 answered, §10 inverse-filtering Q&As, §12, §13 | — |

**Decisions, and what each cost.**

| Decision | Alternative rejected | Evidence |
|---|---|---|
| default `profile="none"` **for this function only** | reuse the module default `tukey` | §12.3: `none` wins every row it does not tie |
| default `eps = 1e-12` (textbook guard) | a "sensible" $10^{-2}$ | §12.3: any larger floor costs accuracy on clean input; the caller supplies the noise level, the library does not guess it |
| `centre="auto"` inherited, not re-implemented | a local half-pixel fix inside the filter | §12.5: a shift is a phase, so it must be corrected in $H$ |
| the floor validates once, in `_floor_value` | two independent checks | the audit found `eps=inf` accepted, and a string floor passing the mask then crashing on `eps * phase` |
| `clip_unit_interval` still raises on non-finite output | replacing NaN with a plausible value | the phase spec forbids silent repair; the floor is what keeps the output finite |

**Scope held.** `degradation.py`, `psf_builder.py`, `renderer.py`,
`ccpd_loader.py`, `binarization.py` and everything under `data/` are
untouched; the dataset was not rebuilt; `DEFAULT_TAPER_PROFILE` still reads
`"tukey"` for the rest of the module; no public signature changed, and the
two helper changes are backward-compatible except that they now *refuse*
inputs that used to produce `inf`.

**Verification.** `python -m pytest -q` → **262 passed, 6 skipped** (the six
CCPD tests need the external `CCPD2019/` folder; no test was skipped, relaxed
or deleted to accommodate this phase). The inverse-filter module alone gives
**39 passed**, and the two module suites give 66 and 58. Module import is free
of `SyntaxWarning` (`python -W error::SyntaxWarning`), which matters because the
docstrings carry LaTeX.

**Next.** Step 4.3 (Wiener) replaces the flat floor by
$|H|^2/(|H|^2 + K)$ and uses the same `psf_frequency_response`,
`taper_edges`, `safe_divide` and `clip_unit_interval`; the `K` it needs is the
inverse of the BSNR the degradation grid already stores.
