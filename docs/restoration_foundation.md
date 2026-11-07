# Step 4 — Restoration: foundation (4.1), inverse (4.2), Wiener (4.3), CLS (4.4), Richardson–Lucy (4.5)

**Owner:** Ashish (2026909001) — steps 4 and 8
**Module:** `license_plate_restoration/restoration.py`
**Tests:** `tests/test_restoration.py` (foundation, 66),
`tests/test_restoration_audit.py` (phase 4.1.1 audit, 58),
`tests/test_inverse_filter.py` (phase 4.2, 39),
`tests/test_wiener_filter.py` (phase 4.3, 72),
`tests/test_cls_filter.py` (phase 4.4, 42),
`tests/test_richardson_lucy.py` (phase 4.5, 59)
**Phase 4.1.1:** complex-valued arithmetic (§6.1), even-sided kernels (§3.4,
§8.2), edge-tapering measurements (§4, §5.4). The audit log in §11 records what
changed and what did not.
**Phase 4.2:** `inverse_filter`, the frequency-domain inverse filter (§12), with
the taper and floor measurements that decided its defaults (§12.3) and the log
in §13.
**Phase 4.3:** `wiener_filter` and the BSNR → $K$ conversion (§14), with the
measured three-way comparison against the degraded image and the inverse
filter on two kernels and three noise levels (§14.3) and the log in §15.
**Phase 4.3.1:** audit of the BSNR → $K$ conversion: mathematical derivation and candidate evaluation (`docs/wiener_mathematical_audit.md`).
**Phase 4.3.2:** finalisation of Wiener mathematical documentation and regression tests (§16): exact observed-power heuristic documentation, clipping analysis, candidate comparison summary, and BSNR regression tests. §14.2–§14.4 and §14.7 were updated accordingly.
**Phase 4.4:** `cls_filter` — Constrained Least Squares (CLS) restoration (§17) with the 4-neighbour discrete Laplacian constraint response $C(u, v)$, smoothness-regularised frequency-domain filtering, zero-gain null handling, and controlled comparison against Wiener across 7×7 and 51×51 blurs (§17.8), logged in §18. Audited in `docs/cls_mathematical_audit.md`.
**Phase 4.5:** `richardson_lucy_filter` — Richardson–Lucy iterative maximum likelihood deconvolution (§19), with exact adjoint back-projection duality, sub-pixel centring for even kernels, flux conservation, Poisson vs Gaussian noise-model analysis, and iteration trajectory study, logged in §20. Audited in `docs/richardson_lucy_audit.md`.
**All four restoration algorithms are now fully implemented and integrated** in `license_plate_restoration/restoration.py`.

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
| **Wiener** (§14, implemented) | $\hat F = \dfrac{H^{*}}{|H|^{2} + K}\,G$ | $H$ and $H^{*}$ from the same grid; $K$ = noise over **sharp**-signal power in the flat-spectrum model, or the observed-power heuristic $\rho/(1+\rho)$ from the BSNR (§14.4, §16) |
| **CLS** (§17, implemented) | $\hat F = \dfrac{H^{*}}{|H|^{2} + \lambda\,|C|^{2}}\,G$ | same $H$; $C$ is the DFT of the 4-neighbour Laplacian; $\lambda \ge 0$ trades smoothness against residual power |
| **Richardson–Lucy** (§19, implemented) | $f^{(k+1)} = f^{(k)} \odot H^T \left[ \dfrac{g}{\max(H f^{(k)}, \varepsilon)} \right]$ | same $H$; adjoint $H^T$ via $H^*$; non-negativity constraint; semi-convergence controlled by iteration count |

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
`python -m pytest tests/test_wiener_filter.py -q` → **72 passed**.
`python -m pytest tests/test_cls_filter.py -q` → **42 passed**.
`python -m pytest tests/test_richardson_lucy.py -q` → **59 passed**.
`python -m pytest -q` (whole suite) → **435 passed, 6 skipped** (the 6 skips are
the CCPD tests that need the external `CCPD2019/` folder). Before the phase
4.1.1 audit the suite gave 165 passed / 6 skipped, before phase 4.2 it gave
218 / 6, before phase 4.3 it gave 262 / 6, before the phase 4.3.1 audit
it gave 315 / 6, after phase 4.3.2 it gave 334 / 6, after phase 4.4 it gave
376 / 6, and after phase 4.5 it gives 435 / 6; **no existing test was removed,
relaxed or skipped** to make room for the new ones. The two new helper tests in
the audit file exist because phase 4.2 used an infinite floor and an
already-accepted infinite `eps` turned the quotient into `inf * phase` — the
fix is in `_floor_value`, shared by `safe_divide` and `near_zero_mask`, so the
two can no longer disagree.

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

And phase 4.3's claims, all in `tests/test_wiener_filter.py`:

| Claim | Test |
|---|---|
| $H\equiv1$ with $K=0$ is an exact identity; the passband gain is $1/(1+K)$ | `test_delta_psf_with_zero_k_is_the_identity`, `test_delta_psf_scales_the_passband_by_one_over_one_plus_k` |
| the filter is the inverse filter times $T=|H|^2/(|H|^2+K)$ | `test_estimate_is_the_inverse_filter_times_the_wiener_taper` |
| a noise-free blur comes back to $4.7\times10^{-6}$ | `test_noise_free_blur_is_recovered_by_a_tiny_k` |
| $K=0$ reproduces step 4.2 to one float32 ULP | `test_tiny_k_agrees_with_the_inverse_filter_and_with_the_formula`, `test_zero_k_is_the_inverse_filter_of_step_4_2` |
| at 20 dB it beats both the degraded image and the inverse filter | `test_wiener_beats_both_the_degraded_image_and_the_inverse_filter` |
| $0/0$ at an exact null returns zero gain, never NaN | `test_zero_k_at_an_exact_null_is_zero_gain_not_a_division` |
| an even kernel still needs `centre="auto"` ($6.5\times10^{-7}$ against $2.8\times10^{-2}$) | `test_even_sized_psf_is_restored_without_a_half_pixel_shift` |
| the stored factor-2 pair stays aligned ($0.006$ px against $0.151$) | `test_even_stored_kernel_keeps_the_decimated_plate_aligned` |
| dropping $H^*$ costs a factor 33 | `test_complex_frequency_response_keeps_the_conjugate` |
| larger $K$ lowers contrast monotonically; huge $K$ goes black | `test_larger_k_lowers_the_output_contrast`, `test_large_k_returns_an_almost_black_image` |
| $\rho/(1+\rho)$ is the noise fraction of the observed power on a signal that never clips (formula over measurement 1.0003–1.0028) | `test_bsnr_conversion_matches_the_realised_noise_ratio` |
| on a plate clipping biases the residual ($-7.2\times10^{-3}$) and breaks the variance split (0.956), yet moves the RMSE by under 1 % | `test_clipping_breaks_the_variance_split_but_not_the_restoration` |
| $\rho/(1+\rho)$ exceeds the sharp-signal ratio $\sigma^2/\operatorname{var}(f)$ on a blurred plate (1.28× and 3.57×) | `test_k_from_bsnr_is_not_the_sharp_signal_ratio` |
| the oracle $\sigma^2/\operatorname{var}(f)$ loses on 7×7 and wins on 51×51 | `test_neither_deployable_rule_nor_the_oracle_wins_on_every_kernel` |
| on a white texture $\sigma^2/\operatorname{var}(f)\approx\rho\sum h^2$ is optimal and $\rho/(1+\rho)$ is not | `test_on_a_white_signal_the_sharp_signal_ratio_is_optimal` |
| `bsnr_db=` is exactly sugar for the equivalent `K=` | `test_bsnr_argument_is_exactly_the_equivalent_k_argument` |
| the output is a finite `float32` image in $[0,1]$, both taper arms | `test_property_output_is_always_a_valid_image`, `test_output_keeps_the_image_convention` |
| bad parameters, images and kernels raise | `test_wiener_rejects_invalid_regularisation`, `test_wiener_rejects_invalid_input` |

And phase 4.4's claims, all in `tests/test_cls_filter.py`:

| Claim | Test |
|---|---|
| `DEFAULT_LAPLACIAN` is 3×3, symmetric, sums to 0, centre is 4.0 | `test_default_laplacian_properties` |
| $C(0,0)=0$ exactly, so DC gain is identically 1.0 for any $\lambda \ge 0$ | `test_laplacian_frequency_response_dc_is_exactly_zero` |
| $C(u, v)$ matches analytical $4 - 2\cos(2\pi u / M) - 2\cos(2\pi v / N)$ to float64 precision | `test_laplacian_frequency_response_matches_analytical_formula` |
| `operator_frequency_response` rejects non-2D, empty, non-finite, or oversized operators | `test_operator_frequency_response_rejects_invalid_operator` |
| `cls_filter` rejects negative, NaN, inf, or non-numeric regularisation | `test_cls_rejects_invalid_regularization` |
| `cls_filter` rejects non-2D, non-finite, empty, or out-of-range images | `test_cls_rejects_invalid_image` |
| `cls_filter` rejects malformed or unnormalised PSFs | `test_cls_rejects_invalid_psf` |
| delta PSF with $\lambda=0$ is the exact identity | `test_delta_psf_with_zero_lambda_is_the_identity` |
| constant images are strictly preserved for any $\lambda \ge 0$ (DC preservation) | `test_constant_image_is_preserved_for_any_lambda` |
| CLS formula agrees with manual NumPy calculation | `test_cls_formula_agrees_with_manual_evaluation` |
| $\lambda=0$ reduces identically to unregularised `inverse_filter` | `test_zero_lambda_matches_unregularised_inverse_filter` |
| exact frequency nulls with $\lambda > 0$ stay finite without NaN or divergence | `test_exact_zero_frequency_response_stays_finite` |
| $0/0$ at exact nulls with $\lambda=0$ returns zero gain, never NaN | `test_zero_lambda_at_exact_null_returns_zero_gain_not_nan` |
| asymmetric kernels preserve conjugate phase via $H^*$ | `test_complex_frequency_response_preserves_conjugate_phase` |
| even-sized stored kernels restore without half-pixel shift using `centre="auto"` | `test_even_sized_psf_restores_without_shift` |
| increasing $\lambda$ monotonically increases smoothness and lowers noise variance | `test_increasing_lambda_increases_smoothness_and_lowers_contrast` |
| CLS outperforms inverse filter on noisy plates | `test_cls_beats_inverse_filter_on_noisy_plate` |
| supports all edge-taper profiles (`none`, `tukey`, `hann`, `linear`) | `test_cls_accepts_all_edge_taper_profiles` |
| property test: restored image is always finite `float32` in $[0, 1]$ | `test_property_cls_output_is_always_valid_float32` |

And phase 4.5's claims, all in `tests/test_richardson_lucy.py`:

| Claim | Test |
|---|---|
| parameter validation rejects negative, non-integer, non-finite, and string iterations | `test_rl_rejects_invalid_iterations` |
| parameter validation rejects non-positive, non-finite, and string `eps` floors | `test_rl_rejects_invalid_eps` |
| parameter validation rejects non-boolean `clip_intermediate` | `test_rl_rejects_invalid_clip_intermediate` |
| input validation rejects malformed images and PSFs | `test_rl_rejects_invalid_image`, `test_rl_rejects_invalid_psf` |
| zero-sum PSFs are rejected | `test_rl_rejects_zero_sum_psf` |
| initial estimate validation rejects mismatched shapes, non-finite, or negative values | `test_rl_rejects_invalid_initial_estimate` |
| inputs are immutable (caller arrays are never modified in-place) | `test_rl_does_not_mutate_inputs` |
| output keeps grayscale float32 convention, range $[0, 1]$, and is deterministic | `test_rl_output_properties`, `test_rl_is_deterministic` |
| `iterations=0` returns the initial estimate clipped to $[0, 1]$ | `test_rl_iterations_zero_returns_initial_estimate` |
| delta PSF leaves clean image unchanged | `test_delta_psf_leaves_image_unchanged` |
| constant images are exact fixed points for any blur PSF | `test_constant_image_is_exact_fixed_point` |
| forward/adjoint duality $\langle Hx, y \rangle = \langle x, H^T y \rangle$ holds to machine precision ($< 10^{-14}$) | `test_forward_adjoint_inner_product_identity` |
| total flux $\sum f^{(k)} = \sum g$ is conserved at each unclipped iteration | `test_flux_conservation_unclipped` |
| zero-valued pixels and black regions stay finite without division by zero | `test_zero_and_near_zero_regions_stay_finite` |
| all four project resolutions execute cleanly ($96\times336$, $92\times256$, $48\times168$, $46\times128$) | `test_all_four_project_resolutions_work` |
| even-sized stored kernels restore without half-pixel shift using `centre="auto"` ($< 0.05$ px vs $> 0.45$ px for `pixel`) | `test_even_sized_psf_restores_without_half_pixel_shift` |
| asymmetric kernels preserve conjugate phase via $H^*$ in back-projection | `test_asymmetric_kernel_preserves_conjugate_phase` |
| on clean blur, RL error monotonically decreases over iterations | `test_rl_deblurs_noise_free_plate` |
| on 20 dB noise, RL demonstrates semi-convergence (early iterations improve, late iterations amplify noise) | `test_rl_noise_amplification_semi_convergence` |
| `clip_intermediate=True` controls overshoot on bright plate backgrounds | `test_rl_clip_intermediate_controls_overshoot` |
| custom non-negative initial estimates are respected | `test_rl_custom_initial_estimate` |
| supports all edge-taper profiles (`none`, `tukey`, `hann`, `linear`) | `test_rl_accepts_all_edge_taper_profiles` |
| property test: restored image is always finite `float32` in $[0, 1]$ | `test_property_rl_output_is_always_valid_float32` |

`python -m license_plate_restoration.restoration` prints the demo numbers
quoted in §3.3, §3.4, §5.4, §6.1, §12.3, §12.5 and §14.3 on a real rendered
plate — the DC gain, the $|H|$ minimum, the agreement with `blur_fft`, the
parity/offset table, the complex-quotient check, the four-window table, the
inverse-filter table (kernel × noise × taper × floor), the floor sweep that
locates the 20 dB optimum, the factor-2 centroid drift, the Wiener table
(kernel × noise × four explicit $K$ values plus the BSNR-derived one), the
$K$ sweeps that locate the empirical optimum, and the ten-plate $K$ audit of
§16 (candidates, oracles, clipping diagnostics, white-texture control, and the
observed noise fraction per BSNR level).

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

**Q: What does the Wiener filter actually divide by?**
Nothing different in kind: it is still one multiplier per coefficient, but the
multiplier is $H^*/(|H|^2+K)$ instead of $1/H$. The two are related by the real
taper $T=|H|^2/(|H|^2+K)$, so the Wiener filter *is* the inverse filter with a
frequency-dependent shutter: open where the blur passed the signal, closed
where it did not (§14.2).

**Q: How does that stop the noise amplification?**
Because the shutter multiplies the noise too. The inverse filter's gain is
$1/|H|$, so noise power is magnified by $1/|H|^2$ — at the 7×7 kernel's
smallest $|H|=2.07\times10^{-6}$ that is $+114$ dB, and $+179$ dB for the
51×51 kernel ($|H|_{\min}=1.13\times10^{-9}$). With $K>0$ the gain is capped at
$|H|/(|H|^2+K)\le 1/(2\sqrt K)$, i.e. an *amplitude* ceiling the caller sets
(§14.5).

**Q: What is $K$, precisely?**
In the scalar Wiener model, the noise power over the **sharp** signal's power,
$K=S_n/S_f=\sigma^2/\operatorname{var}(f)$ — a dimensionless number, never a
decibel value, and exact only if both spectra are flat (§14.2). It is also the
reciprocal of the signal-to-noise ratio of a coefficient the blur passed
undiminished, so $K=10^{-2}$ says "this coefficient carries 20 dB of noise
against its signal". Where $|H|^2=K$ the shutter is exactly $\tfrac12$: that
is the crossover frequency.

**Q: How do you get $K$ from the BSNR the degradation grid stores?**
You cannot get the model's $K$ from it — that needs $\operatorname{var}(f)$ —
so the code uses an observed-power heuristic (§14.4, §16).
`degradation.noise_sigma` sets
$\sigma=\sqrt{\operatorname{var}(h*f)/10^{\mathrm{BSNR}/10}}$, so the BSNR is a
power ratio in dB against the **blurred** signal: $\rho=10^{-\mathrm{BSNR}/10}=
\sigma^2/\operatorname{var}(h*f)$. If noise and signal add independently,
$\operatorname{var}(g)=\operatorname{var}(h*f)(1+\rho)$ and
$\rho/(1+\rho)=\sigma^2/\operatorname{var}(g)$, the noise fraction of the
**observed** power. That is 9.999e-5 at 40 dB, 9.990e-4 at 30 dB and 9.901e-3
at 20 dB. It equals the model's $K$ only if $\operatorname{var}(g)=
\operatorname{var}(f)$; on a blurred plate it is 1.3× (7×7) to 3.4× (51×51)
too large.

**Q: Is that just $1/$BSNR?**
Numerically, yes: $K=\rho$ differs from $\rho/(1+\rho)$ only at second order —
at most 1 % at every level in `BSNR_LEVELS`, 0.2 % in RMSE over ten plates, a
factor 2 at 0 dB. Neither is noise over the sharp plate. The candidate that
*is* consistent with the flat model, $K=\rho\sum h^2$, is exactly optimal on a
white texture but restores 1.9–2.9× worse on plates, because a plate is not
white (§16).

**Q: Then why keep $\rho/(1+\rho)$?**
Because, together with $K=\rho$ (which it matches to 0.2 % in RMSE), it is the
best of the deployable rules on this dataset, and it is only
1.07–1.43× from the best scalar that the reference could buy (§16). Its
success on the 7×7 kernel is a cancellation of two errors, not a derivation:
the flat model under-asks for a plate's $K$, and the observed-power
substitution over-asks. So `bsnr_db=` stays as a convenience, and every
controlled experiment passes an explicit `K=` and reports it.

**Q: Why is $K=0$ the inverse filter?**
The shutter becomes $|H|^2/|H|^2=1$, so nothing is suppressed and the filter is
$H^*G/|H|^2=G/H$ — step 4.2, reached by a different route. Measured: the two
agree to $1.8\times10^{-7}$, one float32 ULP at 0.5. The one coefficient where
it is *not* the same is an exact null, where $0/0$ is undefined and the code
returns zero gain by convention (§14.5).

**Q: Does Wiener filtering always improve the result?**
No, and the honest answer is in the table. On clean input with a benign kernel
the inverse filter is 130× more accurate ($3.3\times10^{-5}$ against
$4.2\times10^{-3}$); over-regularising clean input makes Wiener *worse than the
blur* ($1.165\times10^{-1}$ against $1.106\times10^{-1}$); and on the 7×7 kernel
at 20 dB it removes only 22 % of the blur's error, because the remainder is the
band the blur destroyed, not noise. It does win decisively where the inverse
filter cannot choose a floor without the ground truth — 12× better at 30 dB and
7.8× at 20 dB on the 7×7 kernel, and on the clean 51×51 kernel
$1.466\times10^{-1}$ against the best floor's $1.831\times10^{-1}$ (§14.3, §14.6).

**Q: Why can it still lose the fine character detail?**
Because the shutter closes exactly where fine detail lives. A stroke edge of
width $w$ carries its energy at spatial frequencies around $1/w$; a 43 px smear
nulls a band of spacings around 1/43 cycles/px, and $T\approx0$ there means
the band is *omitted*, not recovered. The filter's contract is minimum mean
square error, not minimum error on the strokes — and the square error prefers a
slightly blurred stroke to a noisy one (§14.6).

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

`python -m pytest -q` → **218 passed, 6 skipped** at the end of the audit,
**262 passed, 6 skipped** after step 4.2 (§13), and **315 passed, 6 skipped**
after step 4.3 (§15).

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
9. **This section is the inverse filter only.** Wiener is implemented in §14
   and CLS and Richardson–Lucy are steps 4.4–4.5; none of them needs anything
   that this phase did not build (§7). The limitations above are properties of
   $1/H$, not of restoration in general.

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

**Next.** Step 4.3 (Wiener) replaced the flat floor by the shutter
$|H|^2/(|H|^2+K)$ predicted here, and *not* by a kernel-dependent $K$: §14.4
converts the BSNR with the observed-power heuristic $\rho/(1+\rho)$ (audited
in §16, where the PSF-aware white-signal rule is measured and rejected for
plates). Steps 4.4 (CLS) and 4.5
(Richardson–Lucy) reuse `psf_frequency_response`, `taper_edges` and
`clip_unit_interval` unchanged, and will be scored by step 8's runner, not by a
metric written here.

---

## 14. Wiener filtering (step 4.3)

The implementation is `wiener_filter` plus `wiener_k_from_bsnr` in
`license_plate_restoration/restoration.py`. Its defaults are the same as step
4.2's for the same measured reason: `profile="none"`, `centre="auto"`,
`require_odd=False`, stored kernels consumed as-is. What is new is a single
scalar, $K$, and a documented way to derive it.

### 14.1 What inverse filtering does to the noise

Adding the noise to the model of §1 and dividing:

$$
\hat F(u,v) \;=\; \frac{G(u,v)}{H(u,v)}
                \;=\; F(u,v) \;+\; \frac{N(u,v)}{H(u,v)},
$$

so the error is the noise multiplied by $1/|H|$, and the noise **power** at that
coefficient by $1/|H|^2$. Nothing distinguishes signal from noise: a
coefficient carrying 20 dB of noise against its signal is amplified exactly as
much as one carrying none. Measured end to end at 20 dB BSNR (§14.3): the
blurred plate is $1.148\times10^{-1}$ RMS from the reference and the inverse
filter is $7.005\times10^{-1}$ — **6.1× worse than doing nothing**.

The two kernels used below bracket the mechanism. $|H|_{\min}$ is
$2.07\times10^{-6}$ for the 7×7 smear and $1.13\times10^{-9}$ for the 51×51
one, i.e. amplitude gains of $4.8\times10^{5}$ and $8.9\times10^{8}$, or
$+114$ dB and $+179$ dB in power, at the frequency the blur destroyed.

The floor of §12.2 caps that gain, but it caps it for *every* coefficient at
once and without knowing where the noise is. The Wiener filter asks a sharper
question: which multiplier minimises the mean square error *of this
coefficient*?

### 14.2 Derivation: the multiplier that minimises the mean square error

Drop $(u,v)$ and consider one coefficient. The model is

$$
G = HF + N, \qquad E|F|^2 = S_f, \qquad E|N|^2 = S_n, \qquad E[F N^*] = 0,
$$

where $S_f(u,v)$ and $S_n(u,v)$ are the signal and noise power spectra at that
frequency. We want the multiplier $W$ that minimises

$$
J(W) \;=\; E|\hat F - F|^2 \;=\; E\left|W(HF+N) - F\right|^2 .
$$

Because $F$ and $N$ are uncorrelated, the cross terms vanish:

$$
J(W) \;=\; |WH-1|^2\,S_f \;+\; |W|^2 S_n .
$$

Setting $\partial J/\partial W^* = 0$:

$$
H^*(WH-1)\,S_f + W S_n = 0
\;\Longrightarrow\;
W\left(|H|^2 S_f + S_n\right) = H^* S_f
\;\Longrightarrow\;
W = \frac{H^* S_f}{|H|^2 S_f + S_n}
  = \frac{H^*}{|H|^2 + S_n/S_f} .
$$

This is the full Wiener filter: one ratio $S_n/S_f$ **per frequency**. The
code evaluates the scalar version,

$$
\hat F \;=\; \frac{H^*}{|H|^2 + K}\,G
\;=\; \underbrace{\frac{|H|^2}{|H|^2+K}}_{T\,\in\,[0,\,1)}
      \;\cdot\; \underbrace{\frac{G}{H}}_{\text{step 4.2}} ,
$$

which equals the full filter **only if $S_n/S_f$ is the same constant at every
frequency**. Three assumptions make it so, and each is checkable:

1. **The noise is white**: $S_n = MN\sigma^2$ at every $(u,v)$ (NumPy's
   unnormalised DFT of an $M\times N$ field of variance $\sigma^2$).
   `add_noise` draws i.i.d. Gaussian samples, so this holds before clipping.
2. **The signal spectrum is flat**: $S_f$ is the same at every non-DC
   frequency. By Parseval its average is $MN\operatorname{var}(f)$, so a flat
   $S_f$ *is* $MN\operatorname{var}(f)$. A plate violates this badly: its
   power sits in the low frequencies, where $|H|\approx1$.
3. **Signal and noise are uncorrelated**, so the cross terms vanish. Clipping in
   `add_noise` violates this mildly: the realised residual is negative on
   average and anticorrelated with the signal (§16.4).

Under 1–3 the constant is

$$
K \;=\; \frac{S_n}{S_f} \;=\; \frac{MN\sigma^2}{MN\operatorname{var}(f)}
  \;=\; \frac{\sigma^2}{\operatorname{var}(f)} ,
$$

the noise power over the **sharp** (original) signal's power — not over the
blurred one, and not over the observed one. (The DC coefficient carries the
mean, not the variance; $|H(0,0)|=1$ gives it gain $1/(1+K)$, a relative
brightness loss of $K/(1+K)$, under 1 % for every BSNR-derived $K$.)

Each factor earns its place:

* **$H^*$** carries the blur's phase back to $-\arg H$, so the estimate is
  neither shifted nor echoed, and it keeps *one* factor of $|H|$ in the
  numerator. That is what makes the passband gain
  $|H|^2/(|H|^2+K)\approx1$ instead of the inverse filter's $1/|H|$: numerator
  and denominator are scaled alike. Dropping it is catastrophic, not subtle —
  on a genuinely complex response the error is $4.8\times10^{-1}$ against
  $1.4\times10^{-2}$ (`test_complex_frequency_response_keeps_the_conjugate`).
* **$|H|^2$** is the blur's power transmittance at that frequency. It is what
  $K$ is compared against, so $|H|^2/K$ *is* that coefficient's
  signal-to-noise ratio, and $|H|^2 = K$ is the crossover, where the shutter is
  exactly $\tfrac12$.
* **$K$** is the only unknown, and the only knob. It sets a ceiling on the
  amplitude gain, $\max_{|H|}|H|/(|H|^2+K) = 1/(2\sqrt K)$ — $K=10^{-2}$ allows
  a factor 5, $K=10^{-4}$ a factor 50. §14.4 explains what the BSNR the
  degradation grid stores can and cannot supply for it.

The identity at the right-hand end is pinned as a test
(`test_estimate_is_the_inverse_filter_times_the_wiener_taper`): the Wiener
filter **is** the inverse filter of step 4.2 multiplied by a real,
frequency-dependent shutter in $[0,1)$, open where the blur passed the signal
and closed where it annihilated it.

### 14.3 Measured: degraded vs inverse vs Wiener on identical samples

Plate `generate_plates(1, seed=7)`, noise `add_noise(..., seed=3)`, error =
interior RMSE against $f$ over a 12 px margin, printed by
`python -m license_plate_restoration.restoration`. **7×7** = motion `length=3`
at 8° with no defocus ($\sum h^2 = 0.1089$); **51×51** = motion `length=21` at
8° with defocus radius 4 ($\sum h^2 = 0.0029$). Every row is the *same*
sample of the same plate.

| kernel | input | degraded | inverse ($\varepsilon=10^{-12}$) | $K=10^{-4}$ | $K=10^{-3}$ | $K=10^{-2}$ | $K=10^{-1}$ | from BSNR |
|---|---|---|---|---|---|---|---|---|
| 7×7 | clean | 1.106e-1 | **3.271e-5** | 1.459e-2 | 2.902e-2 | 5.135e-2 | 1.165e-1 | — |
| 7×7 | 30 dB | 1.110e-1 | 6.985e-1 | 1.767e-1 | 7.689e-2 | **5.689e-2** | 1.189e-1 | 7.692e-2 ($K=9.990$e-4) |
| 7×7 | 20 dB | 1.148e-1 | 7.005e-1 | 4.410e-1 | 2.205e-1 | 8.954e-2 | 1.264e-1 | 8.973e-2 ($K=9.901$e-3) |
| 51×51 | clean | 3.164e-1 | 6.994e-1 | **1.466e-1** | 1.690e-1 | 2.304e-1 | 2.990e-1 | — |
| 51×51 | 30 dB | 3.164e-1 | 7.002e-1 | **1.630e-1** | 1.691e-1 | 2.302e-1 | 2.991e-1 | 1.690e-1 ($K=9.990$e-4) |
| 51×51 | 20 dB | 3.169e-1 | 7.004e-1 | 2.631e-1 | **1.758e-1** | 2.302e-1 | 2.994e-1 | 2.299e-1 ($K=9.901$e-3) |

The sweeps that locate each optimum (same samples, printed by the demo):

```
K sweep  (7, 7)  clean (bsnr K n/a) -> 1e-06:4.243e-03 1e-05:7.464e-03 0.0001:1.459e-02 0.001:2.902e-02 0.01:5.135e-02 0.03:6.926e-02 0.1:1.165e-01 1:4.686e-01
K sweep  (7, 7)  30 dB (bsnr K 9.990e-04) -> 1e-06:4.522e-01 1e-05:3.071e-01 0.0001:1.767e-01 0.001:7.689e-02 0.01:5.689e-02 0.03:7.182e-02 0.1:1.189e-01 1:4.704e-01
K sweep  (7, 7)  20 dB (bsnr K 9.901e-03) -> 1e-06:6.229e-01 1e-05:5.568e-01 0.0001:4.410e-01 0.001:2.205e-01 0.01:8.954e-02 0.03:8.570e-02 0.1:1.264e-01 1:4.744e-01
K sweep (51, 51)  clean (bsnr K n/a) -> 1e-06:3.100e-01 1e-05:1.914e-01 0.0001:1.466e-01 0.001:1.690e-01 0.01:2.304e-01 0.03:2.620e-01 0.1:2.990e-01 1:5.356e-01
K sweep (51, 51)  30 dB (bsnr K 9.990e-04) -> 1e-06:5.840e-01 1e-05:3.382e-01 0.0001:1.630e-01 0.001:1.691e-01 0.01:2.302e-01 0.03:2.620e-01 0.1:2.991e-01 1:5.359e-01
K sweep (51, 51)  20 dB (bsnr K 9.901e-03) -> 1e-06:6.610e-01 1e-05:5.553e-01 0.0001:2.631e-01 0.001:1.758e-01 0.01:2.302e-01 0.03:2.619e-01 0.1:2.994e-01 1:5.367e-01
```

Six readings, including the ones that do not flatter the method:

1. **On noisy input it is the win the phase promises.** 7×7 at 30 dB: the
   inverse filter is $6.985\times10^{-1}$ (6.3× worse than the blur), the
   derived $K$ gives $7.692\times10^{-2}$ — 9.1× better than the inverse filter
   and 1.44× better than the blur. At 20 dB the derived $K$ gives
   $8.973\times10^{-2}$: 7.8× better than the inverse filter, 1.28× better than
   the blur. On the 51×51 kernel at 20 dB, $2.299\times10^{-1}$ against
   $7.004\times10^{-1}$: 3.0× better.
2. **On clean input with a benign kernel the inverse filter still wins, and by
   two orders of magnitude.** 7×7 clean: $3.271\times10^{-5}$ against Wiener's
   best $4.243\times10^{-3}$ (at $K=10^{-6}$). With no noise there is nothing to
   suppress, so the shutter only removes recoverable detail. Wiener's best is
   still 26× better than the blur, which is the point: it is a
   *regularised* filter, not a bad one.
3. **On clean input with an aggressive kernel the ordering reverses, because
   the inverse filter cannot choose its floor without the reference.** 51×51
   clean: $\varepsilon=10^{-12}$ gives $6.994\times10^{-1}$, *worse than the
   blur*; the best floor found in §12.3 ($\varepsilon=10^{-1}$) gives
   $1.831\times10^{-1}$; Wiener with the *smallest* explicit $K=10^{-4}$ gives
   $1.466\times10^{-1}$ — 1.25× better than the best floor and 2.16× better
   than the blur. A single $\varepsilon$ has to be guessed; $K$ has a meaning
   and therefore a way to be derived.
4. **Over-regularisation is a real failure mode, not a theoretical one.**
   7×7 clean at $K=10^{-1}$ gives $1.165\times10^{-1}$ — *worse than not
   restoring at all* ($1.106\times10^{-1}$), because the shutter has closed on
   the blur's passband. On 51×51 clean the same $K$ buys only 6 %
   ($2.990\times10^{-1}$ against $3.164\times10^{-1}$). $K$ must be justified,
   which is the whole of §14.4.
5. **On this sample the derived $K$ is within 1.04–1.35× of the swept optimum
   in RMSE.** Per row, against the eight-point sweep: 1.35× (7×7 30 dB), 1.05×
   (7×7 20 dB, optimum at $3\times10^{-2}$), 1.04× (51×51 30 dB), 1.31×
   (51×51 20 dB). It never beat the swept optimum, and it is computed from the
   noise level alone, without the reference. Over ten plates and a 25-point
   sweep the range is 1.07–1.43× (§16.2).
6. **Its bias flips sign with the kernel, and the optimum does depend on the
   noise.** The derived $K$ is too *small* for the 7×7 kernel (sample optimum
   $10^{-2}$ at 30 dB and $3\times10^{-2}$ at 20 dB, against $9.99\times10^{-4}$
   and $9.90\times10^{-3}$) and too *large* for the 51×51 one (optimum
   $10^{-4}$ and $10^{-3}$). The error curve is flat enough near its minimum
   that both land within 1.35× in RMSE. The kernel dependence is what the
   model predicts and the conversion cannot see: the model's $K$ is
   $\sigma^2/\operatorname{var}(f) = \rho\operatorname{var}(h*f)/
   \operatorname{var}(f)$, which contains the kernel, and the flat-spectrum
   misfit is itself kernel-dependent (§16.3). The noise dependence is real but
   sub-linear: over ten plates the optimum grows by about $\sqrt{10}$ (3.2×) for
   a tenfold rise in $\rho$, on both kernels (§16.2).

### 14.4 The regularisation constant from the repository's BSNR

The BSNR convention is read off the code, not off the name of the quantity.
`degradation.noise_sigma` defines it once:

```python
sigma = sqrt(var(img) / 10 ** (bsnr_db / 10))      # img = h * f
```

so the repository's BSNR is a **power** ratio in decibels ($10\log_{10}$, not
$20\log_{10}$) against the **blurred** plate:

$$
\mathrm{BSNR}_{dB} = 10 \log_{10}
  \frac{\operatorname{var}(h*f)}{\sigma^2},
\qquad
\rho \;=\; 10^{-\mathrm{BSNR}_{dB}/10}
\;\Longleftrightarrow\;
\sigma^2 = \rho \operatorname{var}(h*f).
$$

`degradation.measure_bsnr` reports the same quantity as
$10\log_{10}\left(\frac{\operatorname{var}(\text{clean})}
{\operatorname{var}(\text{noisy}-\text{clean})}\right)$, and `add_noise`'s
saturation compensation rescales the Gaussian amplitude until that is true of
the *clipped* residual $r=g-h*f$: $\operatorname{var}(r)=\sigma^2$ to $10^{-3}$
(measured 0.9993–1.0000 over ten plates, §16.4). It guarantees nothing else:
$r$ is not zero-mean and not independent of $h*f$, because clipping at the
white background only ever removes positive excursions.

**Four quantities, kept apart.** $\rho$ is linear and $\le 1$ for every level
this project uses ($10^{-4}$ at 40 dB, $10^{-2}$ at 20 dB). It is one of four
ratios that are easy to confuse:

| quantity | definition | in terms of $\rho$ | needs |
|---|---|---|---|
| BSNR ratio (noise over *blurred* power) | $\sigma^2/\operatorname{var}(h*f)$ | $\rho$ | the BSNR |
| noise over *original* power (the model's $K$, §14.2) | $\sigma^2/\operatorname{var}(f)$ | $\rho\,\operatorname{var}(h*f)/\operatorname{var}(f)$ | the sharp plate |
| noise fraction of the *observed* power | $\sigma^2/\operatorname{var}(g)$ | $\approx\rho/(1+\rho)$ | the BSNR (if the split holds) |
| **what `wiener_k_from_bsnr` returns** | $\rho/(1+\rho)$ | $\rho/(1+\rho)$ | the BSNR |

The third row follows from the variance split. If noise and signal are
uncorrelated,

$$
\operatorname{var}(g) = \operatorname{var}(h*f) + \sigma^2
                      = \operatorname{var}(h*f)\,(1+\rho)
\quad\Longrightarrow\quad
\frac{\sigma^2}{\operatorname{var}(g)}
  = \frac{\rho\,\operatorname{var}(h*f)}{\operatorname{var}(h*f)(1+\rho)}
  = \frac{\rho}{1+\rho} .
$$

So $\rho/(1+\rho)$ is the noise fraction of the **observed** power. Written
against the model's constant,

$$
\frac{\rho}{1+\rho} \;\approx\; \frac{\sigma^2}{\operatorname{var}(g)}
  \;=\; \underbrace{\frac{\sigma^2}{\operatorname{var}(f)}}_{K\text{ of }
  §14.2}\cdot\frac{\operatorname{var}(f)}{\operatorname{var}(g)} ,
$$

it is the model's $K$ with the **observed variance substituted for the sharp
one** — the only signal power available at restoration time. The two agree only
if $\operatorname{var}(g)=\operatorname{var}(f)$, i.e. if the noise puts back
exactly the variance the blur took out. On every plate measured the blur takes
out more ($|H|\le1$ for a nonnegative unit-sum kernel, so
$\operatorname{var}(h*f)\le\operatorname{var}(f)$), and $\rho/(1+\rho)$
overestimates the model's $K$: 1.3× for the 7×7 kernel and 3.4× for the
51×51 one, averaged over ten plates. The same inequality makes $K=\rho$
larger still.

Consequences:

* **This is an observed-power heuristic, not a derivation of the model's $K$.**
  The BSNR fixes $\sigma^2$ relative to $\operatorname{var}(h*f)$; it carries no
  information about $\operatorname{var}(f)$, so no function of the BSNR alone
  can return $\sigma^2/\operatorname{var}(f)$. What it can return is a
  deployable estimate, and §16 measures how good this one is: within
  1.07–1.43× of the best scalar the reference could buy, and the best of the
  deployable candidates on plates.
* **$K=\rho/(1+\rho)$ and $K=\rho$ are the same rule for this dataset.** They
  differ at second order in $\rho$ — at most 1 % at every level in
  `degradation.BSNR_LEVELS`, within 0.2 % in RMSE over ten plates, a factor 2
  at 0 dB. Neither is noise over the sharp plate.
* **Decibels and ratios are never mixed.** `bsnr_db=30` means $\rho=10^{-3}$;
  passing $30$ where $10^{-3}$ belongs would regularise 30 000× too hard and
  return a nearly black image (at $K=1$ the 7×7 row reads
  $4.704\times10^{-1}$, worse than the blur).
* **No kernel appears in the conversion, but one appears in the model's $K$.**
  $\sigma^2/\operatorname{var}(f) = \rho\,\operatorname{var}(h*f)/
  \operatorname{var}(f)$, and $\operatorname{var}(h*f)/\operatorname{var}(f)$
  is 0.757 for the 7×7 kernel and 0.290 for the 51×51 one (ten-plate means).
  The same BSNR therefore yields the same $K$ for both kernels, while the swept
  optimum differs by about 30× between them (§16.2).
* **The self-consistent flat-model rule is right for white signals and wrong
  for plates.** Under assumption 2 of §14.2, Parseval gives
  $\operatorname{var}(h*f) = \left(\sum h^2\right)\operatorname{var}(f)$, so
  $\sigma^2/\operatorname{var}(f) = \rho\sum h^2$ — deployable, and on a white
  texture it lands on the swept optimum (§16.3). A plate is not white:
  $\sum h^2 = 0.109$ against the measured 0.757 (7×7), $0.0029$ against 0.290
  (51×51), and the rule restores 1.9–2.9× worse than $\rho/(1+\rho)$ (§16.2).
* **The heuristic works on plates by cancellation.** The flat model *under*-asks
  for a plate's $K$ — even the exact $\sigma^2/\operatorname{var}(f)$ loses to
  $\rho/(1+\rho)$ on the 7×7 kernel, 0/10 plates — and substituting
  $\operatorname{var}(g)$ for $\operatorname{var}(f)$ *over*-asks. On the 7×7
  kernel the two errors roughly cancel; on the 51×51 kernel the substitution
  overshoots by 3.4× and the oracle wins on 10/10 plates.

**Use.** `bsnr_db=` stays as a convenience: deployable, reproducible, and as
good as any rule derivable from the BSNR here. Scientifically controlled
experiments — step 8's runner, any ablation, any comparison with CLS — pass an
explicit `K=` and report it. Neither the function nor its signature changed
(§16.5).

**Verified against the data.** On a texture that `add_noise` never clips, the
realised $\sigma^2/\operatorname{var}(g)$ matches $\rho/(1+\rho)$: formula over
measurement 1.0003, 1.0009, 1.0016, 1.0028 at 40/30/25/20 dB
(`test_bsnr_conversion_matches_the_realised_noise_ratio`). On a *plate* it does
not, because clipping biases the residual; the demo prints, for plate seed 7
and noise seed 3:

```
  kernel  bsnr rho/(1+rho)   observed   obs/K      split
  (7, 7)    40  9.9990e-05 1.0033e-04  1.0034 9.9920e-05
  (7, 7)    30  9.9900e-04 1.0124e-03  1.0135 9.9900e-04
  (7, 7)    25  3.1523e-03 3.2300e-03  1.0246 3.1523e-03
  (7, 7)    20  9.9010e-03 1.0358e-02  1.0461 9.9001e-03
(51, 51)    40  9.9990e-05 1.0011e-04  1.0012 9.9931e-05
(51, 51)    30  9.9900e-04 1.0048e-03  1.0058 9.9846e-04
(51, 51)    25  3.1523e-03 3.1873e-03  1.0111 3.1495e-03
(51, 51)    20  9.9010e-03 1.0129e-02  1.0230 9.9009e-03
```

"observed" is $\operatorname{var}(r)/\operatorname{var}(g)$; "split" is
$\operatorname{var}(r)/(\operatorname{var}(h*f)+\operatorname{var}(r))$, which
`add_noise` enforces and which therefore matches the formula. The observed
fraction exceeds it by 0.1–4.6 %, growing with the noise and with the clipped
fraction (about 25 % of pixels for 7×7, 9–10 % for 51×51). Restoring with the
observed fraction instead of the formula moves the RMSE by under 1 % (§16.4).

### 14.5 Numerical behaviour

* **No denominator floor, and none is needed for $K>0$.** The denominator is
  $|H|^2 + K \ge K > 0$, so the Wiener expression is protected by construction.
  `safe_divide` is deliberately *not* used: its floor changes the equation, and
  at $|H|$ just above the floor it would silently substitute a different
  operator than the one being solved. The shutter already does the job the
  floor was invented for.
* **Exact zeros of $H$.** A 2×2 bilinear kernel nulls Nyquist exactly
  ($|H|_{\min}=0.0$ on 44 coefficients of a 40×40 grid). With $K>0$ the
  coefficient is $0/K = 0$, which is the gain's limit as well — the formula
  protects itself. With $K=0$ it is $0/0$, and here the limit is **not** zero:
  $|H|/|H|^2 = 1/|H|$ *diverges*, which is the inverse filter's
  ill-conditioning in its purest form. The code returns zero gain there by an
  explicit convention written into the division (`where=denominator > 0`), and
  the docstring says so. The measured cost is honest: with $K=0$ the bilinear
  case restores at $5.2\times10^{-1}$ RMS, against $2.5\times10^{-2}$ at
  $K=10^{-12}$, because half the spectrum is simply not recoverable.
* **Nothing non-finite is ever hidden.** No value is replaced after the fact;
  `clip_unit_interval` still raises if a NaN/inf reaches it, which is why a
  NaN or inf *image* is rejected by validation rather than smeared into the
  output.
* **Complex arithmetic is preserved** end to end (`complex128` spectrum, one
  vectorised division), and the imaginary part of the inverse FFT is dropped
  only at the very end, where it is round-off for a real image and a real
  kernel.
* **Even-sized kernels still need `centre="auto"`**, and the effect measured
  through *this* filter is the same as through step 4.2: on the sub-pixel blob
  $6.5\times10^{-7}$ against $2.8\times10^{-2}$ and $0.0000$ px against
  $+0.5000$; on the stored factor-2 pair $0.089$ against $0.195$ with a drift
  of $0.006$ px against $0.151$ px. A phase error is not something a real $K$
  can fix.
* **Large and small $K$** behave as the algebra says: output spread falls
  monotonically with $K$ ($8.5\times10^{-2} \to 1.3\times10^{-2}$ from
  $K=10^{-4}$ to $1$) and $K=10^{6}$ returns an essentially black image
  (max $5.6\times10^{-7}$).

### 14.6 Why the fine character detail still goes

The shutter is a **suppression**, not a recovery. A stroke edge of width $w$
carries its energy around spatial frequency $1/w$; the 43 px smear nulls a band
of spacings near $1/43$ cycles/px, and there $T\approx0$ — the band is
*omitted*. The filter's contract is minimum mean square error, and a square
error prefers a slightly blurred stroke to a noisy one; that is the trade the
user asked for, and it is not the same as "recover the character".

The numbers show how much of the residual is detail rather than noise. For the
7×7 kernel at 20 dB the noise's own standard deviation is
$\sigma=3.31\times10^{-2}$, while the restored error is $8.973\times10^{-2}$ —
**2.7× $\sigma$**. For the 51×51 kernel at 20 dB, $\sigma=1.98\times10^{-2}$
against an error of $2.299\times10^{-1}$, **11.6× $\sigma$**. The noise has been
suppressed to a fraction of $\sigma$; what is left is the band the blur
destroyed, and no value of $K$ brings it back.

### 14.7 Limitations — say these out loud

1. **One scalar $K$ cannot be optimal at every frequency.** The flat-spectrum
   model is an average. Over ten plates (§16.2) the best swept scalar reaches
   0.70–0.94× of the derived value's RMSE, and the spectral oracle
   $H^*S_f/(|H|^2S_f+S_n)$, which no restoration can compute, goes further
   still (0.60–0.87×).
2. **It cannot invent detail**, and the error is dominated by the destroyed
   band at strong blur (§14.6). It is a better *estimate*, not a better
   *measurement*: no downstream step recovers information the blur removed.
3. **$K=0$ is not free.** It is the inverse filter, including its
   ill-conditioning; the only addition is the documented zero-gain convention
   at exact nulls, whose cost is measured in §14.5.
4. **`add_noise` guarantees only the residual's variance.** After clipping,
   $r=g-h*f$ has $\operatorname{var}(r)=\sigma^2$, but it is biased negative
   and correlated with the plate, so the variance split behind
   $\rho/(1+\rho)$ holds to 0.956–0.994, not exactly (§16.4). The cost is
   measured, not assumed: restoring with the realised fraction instead of the
   formula moves the RMSE by at most 1 %. A real sensor's noise would have to
   be measured as well, and the conversion would need that measurement.
5. **No PSNR or SSIM here.** Step 8 owns the metrics; this phase reports
   interior RMSE against $f$ only. RMSE and PSNR are monotone in each other at
   fixed peak, so the *orderings* above do not depend on the choice.
6. **Clipping is inside the measured error.** The estimate is clipped to
   $[0,1]$ before scoring, as the image convention requires.
7. **The border is meaningless under a taper.** All numbers are interior crops;
   a tapered frame must never be binarised or scored (§5.3).
8. **Two kernels and few samples**, from the generative pipeline
   (`plates.npz` + `blur_fft`), not from the 471 MB `degraded.npz`, which is
   gitignored and absent from this checkout. §14.3 uses one plate and one seed
   per condition; §16 extends the $K$ comparison to ten plates and ten noise
   draws at two levels. Enough to pin the defaults and the conversion; step
   8's runner is what turns this into a survey.
9. **$K$ from the BSNR is a heuristic.** It is the noise fraction of the
   observed power, not the model's noise-to-sharp-signal ratio, and it ignores
   the kernel (§14.4). Scientifically controlled experiments pass an explicit
   `K=` and report it; `bsnr_db=` is a convenience for deployment and demos.

---

## 15. Phase 4.3 log

| File | Change | Tests |
|---|---|---|
| `license_plate_restoration/restoration.py` | `wiener_filter`, `wiener_k_from_bsnr`, `_finite_scalar` (the numeric gate for $K$ and the BSNR); module docstring; demo extended with the Wiener table, the $K$ sweeps and the $\sum h^2$ vs $\operatorname{var}(h*f)/\operatorname{var}(f)$ line | existing 163 tests across the three suites unchanged |
| `tests/test_wiener_filter.py` | **new**, 53 tests: the fourteen required categories plus a Hypothesis property | 53 |
| `docs/restoration_foundation.md` | §9 counts and claims, §10 Wiener Q&As, §14, §15 | — |

**Decisions, and what each cost.**

| Decision | Alternative rejected | Evidence |
|---|---|---|
| `K = \rho/(1+\rho)` from `var(g) = var(h*f)(1+\rho)` | $K=\rho$ (naive $1/$BSNR); $K=\rho\sum h^2$ (white plate) | §14.4: agree to 1 % here but a factor 2 at 0 dB; the white-plate route is 1.9–2.9× worse in RMSE. *Phase 4.3.1 (§16): kept, but re-described as the noise fraction of the observed power — an observed-power heuristic, not the model's $\sigma^2/\operatorname{var}(f)$.* |
| `wiener_k_from_bsnr` takes no PSF | a PSF-aware conversion | the PSF enters only through $|H|^2$; adding it invites the rejected variant above |
| **exactly one** of `K=` / `bsnr_db=` required, neither defaulted | a default $K$ such as $10^{-3}$ | $K$ is a property of the noise; §14.3 shows a wrong $K$ can be worse than no restoration |
| default `profile="none"` (shared with step 4.2) | the module default `tukey` | §12.3 measured it; the Wiener arm inherits the same rounding model |
| **no** denominator floor; `np.divide(..., where=...)` | `safe_divide` with a small $\varepsilon$ | a floor changes the equation; for $K>0$ the denominator is already $\ge K$ |
| $0/0$ at an exact null with $K=0$ returns zero gain | leaving the NaN, or flooring by $\varepsilon$ | finite output is required and the gain diverges; the convention is stated, not hidden, and its cost is measured (§14.5) |
| `clip_unit_interval` still raises on non-finite | patching NaN with a plausible value | the spec forbids silent repair; validation rejects a NaN image up front |

**Scope held.** `degradation.py`, `psf_builder.py`, `renderer.py`,
`ccpd_loader.py`, `binarization.py` and everything under `data/` are
untouched; `degraded.npz` was not rebuilt; no stored PSF array was modified;
`DEFAULT_TAPER_PROFILE` still reads `"tukey"` for the rest of the module; no
public signature changed. `_floor_value` was briefly routed through
`_finite_scalar` and reverted when the pinned `"eps must be > 0 and finite"`
message changed — the numeric gate stays private to the Wiener parameters.

**Verification.** `python -m pytest -q` → **315 passed, 6 skipped** (the six
CCPD tests need the external `CCPD2019/` folder; no test was skipped, relaxed or
deleted to accommodate this phase). The Wiener module alone gives **53
passed**; the other module suites give 66, 58 and 39. Module import is free of
`SyntaxWarning` (`python -W error::SyntaxWarning`), which matters because the
docstrings carry LaTeX, and the 12 doctests in the module pass.

**Next.** Step 4.4 (CLS) replaces the shutter justification by a smoothness
prior, $\lambda|C|^2$ in the same denominator position; step 4.5
(Richardson–Lucy) leaves the frequency domain. Neither is started here.

---

## 16. Wiener BSNR-to-$K$ mathematical audit (Phase 4.3.1 / 4.3.2)

This section records the mathematical audit of the BSNR-to-$K$ regularisation
conversion implemented in `wiener_k_from_bsnr`, as conducted in Phase 4.3.1 and
finalised in Phase 4.3.2. For the complete numerical tables, 10-seed standard
deviations, multi-plate evaluations, and white-noise controls, refer directly to
the standalone audit report: [`docs/wiener_mathematical_audit.md`](wiener_mathematical_audit.md).

### 16.1 The repository's BSNR definition

`degradation.noise_sigma` fixes the definition of BSNR once for the entire
pipeline:

$$
\mathrm{BSNR}_{\mathrm{dB}} = 10 \log_{10} \frac{\operatorname{var}(h * f)}{\sigma_n^2}
$$

Key properties:
* **Power ratio ($10\log_{10}$)**: It compares signal variance to noise
  variance, not amplitudes ($20\log_{10}$).
* **Defined against the blurred plate**: The signal variance is
  $\operatorname{var}(h * f)$, **not** the sharp plate's variance
  $\operatorname{var}(f)$.
* **Dimensionless linear ratio**:
  $$
  \rho = 10^{-\mathrm{BSNR}_{\mathrm{dB}}/10} = \frac{\sigma_n^2}{\operatorname{var}(h * f)}
  $$

### 16.2 Observed-power heuristic vs the classical Wiener model

The MMSE derivation (§14.2) requires $K_{\mathrm{model}} = S_n / S_f$. Under
the standard flat-spectrum assumption, this is the noise power over the
**sharp** plate's power:

$$
K_{\mathrm{model}} = \frac{\sigma_n^2}{\operatorname{var}(f)} \qquad \text{(Candidate C, an oracle)}
$$

What `wiener_k_from_bsnr` evaluates instead is:

$$
K_A = \frac{\rho}{1 + \rho} = \frac{\sigma_n^2 / \operatorname{var}(h * f)}{1 + \sigma_n^2 / \operatorname{var}(h * f)}
    = \frac{\sigma_n^2}{\operatorname{var}(h * f) + \sigma_n^2} \approx \frac{\sigma_n^2}{\operatorname{var}(g)}
$$

This is the noise fraction of the **observed (degraded) power**. Writing it in
terms of the model's $K_{\mathrm{model}}$:

$$
K_A = K_{\mathrm{model}} \cdot \frac{\operatorname{var}(f)}{\operatorname{var}(g)}
$$

Because any passive low-pass blur attenuates image variance
($\operatorname{var}(h * f) < \operatorname{var}(f)$), the degraded variance
is smaller than the sharp variance ($\operatorname{var}(g) < \operatorname{var}(f)$)
whenever blur variance loss exceeds added noise power (true for every tested
plate and BSNR level). Consequently:

$$
K_A > K_{\mathrm{model}}
$$

$K_A$ is an **observed-power heuristic**, not theoretically equivalent to the
classical noise-to-original-signal power ratio.

### 16.3 Impact of clipping and saturation compensation

The forward model in `degradation.add_noise` clips pixels to $[0, 1]$ and uses
a fixed-point loop (`saturation_compensation=True`) to rescale the Gaussian
amplitude until the realised residual $r = g - h * f$ has
$\operatorname{var}(r) = \sigma_n^2$.

On license plates ($\approx 60\%$ white background at $1.0$):
1. **Residual bias**: Clipping truncates positive excursions on bright pixels.
   The residual is not zero-mean (measured $\mathbb{E}[r] \approx -2.3\times 10^{-3}$
   at 30 dB, $-7.3\times 10^{-3}$ at 20 dB).
2. **Signal-noise correlation**: Pixels with $h * f \approx 1$ clip downwards
   ($r \le 0$), inducing negative covariance $\operatorname{Cov}(h * f, r) < 0$
   (measured $-7.3\times 10^{-4}$ at 30 dB, $-2.4\times 10^{-3}$ at 20 dB).
3. **Variance split breakdown**:
   $\operatorname{var}(g) = \operatorname{var}(h * f) + \operatorname{var}(r) + 2\operatorname{Cov}(h * f, r) < \operatorname{var}(h * f) + \operatorname{var}(r)$.
   The split ratio $\operatorname{var}(g) / (\operatorname{var}(h * f) + \sigma_n^2)$
   drops to $0.987$ at 30 dB and $0.955$ at 20 dB.
4. **Practical sensitivity**: Restoring with the realised observed noise
   fraction $K_{A,\mathrm{obs}} = \operatorname{var}(r) / \operatorname{var}(g)$
   changes RMSE by under $0.9\%$ across all tested conditions.

### 16.4 Candidate $K$ comparison and experimental findings

Controlled experiments across 10 random seeds on both small (7×7) and large
(51×51) kernels at 20 dB and 30 dB BSNR yielded the following findings:

* **Candidate A ($K_A = \frac{\rho}{1+\rho}$) vs Candidate B ($K_B = \rho$)**:
  Candidate B is the direct BSNR ratio $\sigma_n^2 / \operatorname{var}(h * f)$.
  $K_B$ and $K_A$ differ by $\rho^2 / (1 + \rho)$ ($\le 0.99\%$ at $\ge 20$ dB).
  In practice, they are **nearly indistinguishable**: RMSE differs by less
  than $0.2\%$, and PSNR differs by less than $0.02$ dB across all tested
  conditions.
* **Candidate C ($K_C = \sigma_n^2 / \operatorname{var}(f)$)**:
  Candidate C is an **offline oracle** that requires the unavailable clean
  image variance $\operatorname{var}(f)$; it cannot be deployed in ordinary
  restoration. On large blur (51×51), Candidate C improved RMSE by $9.3\%$
  (30 dB) and $14.9\%$ (20 dB) because severe blur destroys high frequencies
  and over-regularising causes blurring to remain uncorrected. On small blur
  (7×7), Candidate C performed **worse** than Candidate A by $6.7\%\text{--}11.3\%$
  because the flat-spectrum model under-regularises high frequencies, and
  Candidate A's inflation helps compensate for this mismatch.
* **Candidate W ($K_W = \rho \sum h^2$)**:
  Candidate W is the Parseval white-signal estimate ($\operatorname{var}(h * f) = \sum h^2 \operatorname{var}(f)$).
  It proved completely unsuitable for structured license-plate images
  ($2.2\times\text{--}3.4\times$ worse RMSE, performing worse than doing
  nothing on large blurs) because plate glyphs have highly colored spectra.
* **Non-universality of scalar $K$**:
  No single scalar BSNR-derived $K$ was universally optimal across all tested
  blur conditions. For scientifically controlled experiments, ablations, and
  pipeline benchmarking (Stage 8), callers should pass an explicit `K=`
  parameter to explore parameter sensitivity rather than relying on the
  convenience heuristic `bsnr_db=`.

---

## 17. Constrained Least Squares filtering (step 4.4)

Constrained Least Squares (CLS) restoration — also known as regularised inverse filtering or Tikhonov–Miller regularisation — tackles the fundamental ill-posedness of image deconvolution by formulating restoration as a constrained optimisation problem.

### 17.1 Mathematical model and derivation

Let $f \in \mathbb{R}^{M \times N}$ be the sharp underlying image, $h$ the blur point spread function, and $g$ the observed degraded image:

$$
g = h * f + n
$$

where $n$ is zero-mean noise with total power $\|n\|_2^2$.

The unconstrained least squares problem $\min_f \|g - h * f\|_2^2$ leads to the unregularised inverse filter $\hat{F} = G / H$, which catastrophically amplifies noise at frequency nulls where $H(u, v) \approx 0$.

CLS resolves this by minimising a measure of image **roughness** subject to the fidelity constraint that the residual matches the expected noise power:

$$
\min_f \|c * f\|_2^2 \qquad \text{subject to} \qquad \|g - h * f\|_2^2 = \|n\|_2^2
$$

where $c$ is a high-pass constraint operator (typically the discrete Laplacian), which measures second-order spatial derivatives and thus penalises high-frequency oscillations and noise spikes.

Using the method of Lagrange multipliers, we define the objective functional:

$$
J(f) = \|c * f\|_2^2 + \alpha \Big( \|g - h * f\|_2^2 - \|n\|_2^2 \Big)
$$

where $\alpha > 0$ is the Lagrange multiplier. Dividing by $\alpha$ and setting $\lambda = 1/\alpha \ge 0$, this is equivalent to minimising:

$$
\min_f \Big( \|g - h * f\|_2^2 + \lambda \|c * f\|_2^2 \Big)
$$

By Parseval's theorem, spatial convolutions become point-wise multiplications in the 2-D discrete Fourier domain:

$$
J(F) = \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} \left( |G(u, v) - H(u, v) F(u, v)|^2 + \lambda |C(u, v)|^2 |F(u, v)|^2 \right)
$$

Taking the complex derivative with respect to $F^*(u, v)$ and setting it to zero:

$$
\frac{\partial J}{\partial F^*(u, v)} = -H^*(u, v) \big( G(u, v) - H(u, v) F(u, v) \big) + \lambda |C(u, v)|^2 F(u, v) = 0
$$

Rearranging terms:

$$
\Big( |H(u, v)|^2 + \lambda |C(u, v)|^2 \Big) F(u, v) = H^*(u, v) G(u, v)
$$

Solving for the restored estimate $\hat{F}(u, v)$:

$$
\hat{F}(u, v) = \frac{H^*(u, v) G(u, v)}{|H(u, v)|^2 + \lambda |C(u, v)|^2}
$$

### 17.2 The Laplacian constraint operator

The high-pass constraint operator $c$ is implemented using the standard 4-neighbour discrete Laplacian kernel:

$$
c = \begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0
\end{bmatrix}
$$

#### Properties of the Laplacian operator:
1. **Zero sum**: $\sum_{i, j} c_{i, j} = 0 - 1 + 0 - 1 + 4 - 1 + 0 - 1 + 0 = 0$.
2. **Negative elements**: The kernel contains negative off-diagonal entries, representing spatial second derivatives $\nabla^2 f \approx \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}$.
3. **Never passes through `validate_psf`**: Because it sums to 0 and contains negative coefficients, passing $c$ through `validate_psf()` would immediately trigger validation errors (`sum must equal 1`, `negative weights`). It is processed by a dedicated helper: `operator_frequency_response()`.
4. **Analytical frequency response**: When centred at $(0, 0)$ on an $M \times N$ grid, the DFT $C(u, v)$ is real and given by:
   $$
   C(u, v) = 4 - 2\cos\left(\frac{2\pi u}{M}\right) - 2\cos\left(\frac{2\pi v}{N}\right)
   $$
5. **DC behaviour**: At DC $(u, v) = (0, 0)$:
   $$
   C(0, 0) = 4 - 2(1) - 2(1) = 0
   $$
   This ensures that the DC component (average image brightness) incurs **zero smoothness penalty**.
6. **High-frequency behaviour**: At the 2-D Nyquist frequency $(M/2, N/2)$:
   $$
   C(M/2, N/2) = 4 - 2(-1) - 2(-1) = 8 \implies |C(M/2, N/2)|^2 = 64
   $$
   The regularisation penalty increases monotonically from $0$ at DC to $64$ at Nyquist, heavily suppressing high-frequency noise amplification.

### 17.3 Physical meaning of regularisation parameter $\lambda$

The parameter $\lambda \ge 0$ controls the trade-off between fidelity to the degraded observations and spatial smoothness of the restored image:

* **$\lambda = 0$ (Zero regularisation)**:
  The filter reduces exactly to the unregularised inverse filter:
  $$
  \hat{F}(u, v) = \frac{H^*(u, v) G(u, v)}{|H(u, v)|^2} = \frac{G(u, v)}{H(u, v)}
  $$
  At exact frequency nulls where $H(u, v) = 0$, the repository defines zero gain ($0/0 \to 0$), preventing undefined divisions.
* **Small $\lambda > 0$ (Weak regularisation)**:
  Inverse filtering dominates across the passband where $|H(u, v)|^2 \gg \lambda |C(u, v)|^2$. High-frequency noise is suppressed where $|H(u, v)|^2$ falls below $\lambda |C(u, v)|^2$. If $\lambda$ is too small, noise amplification and ringing artefacts remain visible.
* **Large $\lambda$ (Strong regularisation)**:
  Smoothness dominates. High spatial frequencies are aggressively damped, eliminating noise but blurring fine glyph edges and stroke details.
* **$\lambda \to \infty$ (Infinite regularisation)**:
  For all $(u, v) \ne (0, 0)$, $\hat{F}(u, v) \to 0$. At DC $(0, 0)$, because $C(0, 0) = 0$ and $|H(0, 0)| = 1.0$, the gain remains identically $1.0$: $\hat{F}(0, 0) = G(0, 0)$. Thus, $\lim_{\lambda \to \infty} \hat{f}$ is a uniform flat image matching the mean brightness of $g$.

### 17.4 Comparison between CLS and Wiener filtering

| Property | Wiener Filter | Constrained Least Squares (CLS) |
|---|---|---|
| **Filter Transfer Function** | $\hat{F} = \dfrac{H^*}{|H|^2 + K}\,G$ | $\hat{F} = \dfrac{H^*}{|H|^2 + \lambda |C|^2}\,G$ |
| **Regularisation Term** | Constant scalar $K$ across all frequencies | Frequency-dependent $\lambda |C(u, v)|^2$ |
| **Underlying Prior** | White signal spectrum ($S_f(u, v) = \text{const}$) | High-pass smoothness constraint ($\|c * f\|_2^2$) |
| **DC Gain at $(0, 0)$** | $\dfrac{1}{1 + K} < 1$ (attenuates mean brightness) | $\dfrac{1}{1 + \lambda(0)} = 1.0$ (strictly preserves mean brightness) |
| **Passband Preservation** | Damped uniformly by $1/(1 + K)$ | Damped proportionally to spatial frequency (near 0 damping at low frequencies) |
| **High-Frequency Damping** | Damped by $\approx |H|^2 / K$ | Damped by $\approx |H|^2 / (\lambda |C|^2)$ (up to $64\lambda$ penalty) |
| **Parameter Tuning** | $K$ (derived from BSNR or swept) | $\lambda$ (tuned/swept for optimal trade-off) |

#### Key Theoretical Advantages of CLS on License Plates:
1. **Strict DC Brightness Preservation**:
   In Wiener filtering, the DC gain is $\frac{1}{1 + K}$. For $K = 0.01$ (20 dB BSNR), DC gain is $0.9901$, introducing a slight global darkening of the bright plate background. In CLS, because $C(0, 0) = 0$, the denominator at DC is $|H(0, 0)|^2 + \lambda |C(0, 0)|^2 = 1.0^2 + 0 = 1.0$. The DC gain is identically $1.0$ for all $\lambda \ge 0$, guaranteeing that average plate brightness is preserved without bias.
2. **Frequency-Adaptive Regularisation**:
   License plate images are characterized by piecewise-smooth backgrounds and sharp character edges. Their power is concentrated at low-to-medium frequencies. Wiener's constant $K$ penalises low frequencies unnecessarily while under-penalising high-frequency noise. CLS applies near-zero regularisation at low frequencies and massive regularisation ($|C|^2 \to 64$) at high frequencies, achieving cleaner character contrast with lower residual noise.

### 17.5 Numerical stability and null handling

In the CLS denominator:

$$
D(u, v) = |H(u, v)|^2 + \lambda |C(u, v)|^2
$$

1. **For $\lambda > 0$**:
   - At DC $(0, 0)$: $|C(0, 0)| = 0$, but the PSF is normalised such that $\sum h = 1.0$, which implies $H(0, 0) = 1.0$. Therefore, $D(0, 0) = 1.0 + 0 = 1.0 > 0$.
   - At any non-zero frequency $(u, v) \ne (0, 0)$: $|C(u, v)|^2 > 0$ everywhere on the grid. Since $\lambda > 0$ and $|H(u, v)|^2 \ge 0$, $D(u, v) \ge \lambda |C(u, v)|^2 > 0$.
   - **Conclusion**: The denominator $D(u, v)$ is **strictly positive everywhere** on the frequency grid for any $\lambda > 0$. Unlike unregularised inverse filtering, CLS requires **no artificial epsilon floor** $\varepsilon$.
2. **For $\lambda = 0$**:
   - If $\lambda = 0$, $D(u, v) = |H(u, v)|^2$. At exact frequency nulls where $H(u, v) = 0$, division $0 / 0$ is handled safely using NumPy's `np.divide(..., where=denominator > 0, out=np.zeros_like(...))`, assigning zero gain to true nulls without emitting `RuntimeWarning` or generating `NaN`/`inf`.

### 17.6 Implementation and API contract

CLS is implemented in `license_plate_restoration/restoration.py` through three components:

1. **`DEFAULT_LAPLACIAN`**:
   The $3 \times 3$ standard 4-neighbour discrete Laplacian kernel array.
2. **`operator_frequency_response(operator, image_shape)`**:
   Computes the 2-D DFT of any signed spatial kernel on `image_shape`. Pads the kernel to `image_shape`, shifts its centre to $(0, 0)$ via `np.roll`, and evaluates `np.fft.fft2`. Validates that the operator is a non-empty, finite 2-D array fitting within `image_shape`.
3. **`laplacian_frequency_response(image_shape)`**:
   Returns the cached or evaluated frequency response of `DEFAULT_LAPLACIAN` on `image_shape`.
4. **`cls_filter(image, psf, regularization: float, *, width=None, fraction=DEFAULT_TAPER_FRACTION, profile=DEFAULT_INVERSE_TAPER, centre=DEFAULT_CENTRE_MODE, require_odd=False, tol=PSF_SUM_TOL) -> np.ndarray`**:
   - Validates `image` as a grayscale float32 array in $[0, 1]$.
   - Validates `regularization` as a finite scalar $\ge 0$.
   - Validates `psf` via `validate_psf()`.
   - Computes $H$ via `psf_frequency_response(psf, degraded.shape, centre=centre, require_odd=require_odd, tol=tol)`.
   - Computes $C$ via `laplacian_frequency_response(degraded.shape)`.
   - Windows the degraded image via `taper_edges()`.
   - Evaluates $\hat{F}(u, v) = \frac{H^*(u, v) G(u, v)}{|H(u, v)|^2 + \lambda |C(u, v)|^2}$.
   - Transforms back via `np.real(np.fft.ifft2(...))` and clips to $[0, 1]$ via `clip_unit_interval()`.

### 17.7 Edge tapering and boundary assumptions

Following the design established in Phase 4.1.1 and Phase 4.2:
- The default edge-taper profile is `profile="none"` (`DEFAULT_INVERSE_TAPER`).
- Standard plate images in this repository feature an 8 px margin, ensuring that circular convolution artefacts are negligible across the plate interior.
- Callers may select alternative taper profiles (`tukey`, `hann`, `linear`) via keyword arguments if restoring cropped or unpadded images.

### 17.8 Controlled experimental comparison

Restoration performance was evaluated on a rendered synthetic plate (`seed=7`, interior margin 12 px) degraded by the smallest and largest motion-defocus kernels in the benchmark suite across clean, 30 dB, and 20 dB BSNR conditions.

#### Experiment Setup:
* **Small Blur (7×7)**: Motion length 3, angle 8°, defocus radius 0.
* **Large Blur (51×51)**: Motion length 21, angle 8°, defocus radius 4.
* **Noise levels**: Clean ($\infty$ dB), 30 dB BSNR, 20 dB BSNR (additive Gaussian noise with saturation compensation).
* **Metrics**: Root Mean Square Error (RMSE) against ground truth plate interior; Peak Signal-to-Noise Ratio (PSNR in dB).

#### Experimental Results (RMSE):

| Kernel | Noise Level | Degraded Input | Inverse Filter | Wiener Filter | CLS ($\lambda=0$) | CLS ($\lambda=10^{-4}$) | CLS ($\lambda=10^{-3}$) | CLS ($\lambda=10^{-2}$) | CLS ($\lambda=10^{-1}$) | Best CLS ($\lambda^*$) |
|---|---|---|---|---|---|---|---|---|---|---|
| **7×7** | **Clean** | 1.106e-01 | 3.271e-05 | 1.459e-02 ($K=10^{-4}$) | **3.271e-05** | 2.922e-02 | 4.491e-02 | 6.403e-02 | 8.608e-02 | **3.271e-05** ($\lambda=0$) |
| **7×7** | **30 dB** | 1.110e-01 | 6.985e-01 | 7.692e-02 ($K=10^{-3}$) | 6.985e-01 | 6.160e-02 | **5.070e-02** | 6.496e-02 | 8.629e-02 | **5.070e-02** ($\lambda=10^{-3}$) |
| **7×7** | **20 dB** | 1.148e-01 | 7.005e-01 | 8.973e-02 ($K=10^{-2}$) | 7.005e-01 | 1.718e-01 | 8.628e-02 | **7.245e-02** | 8.822e-02 | **7.245e-02** ($\lambda=10^{-2}$) |
| **51×51** | **Clean** | 3.164e-01 | 6.994e-01 | 1.466e-01 ($K=10^{-4}$) | 6.994e-01 | 1.716e-01 | **1.542e-01** | 1.697e-01 | 2.088e-01 | **1.542e-01** ($\lambda=10^{-3}$) |
| **51×51** | **30 dB** | 3.164e-01 | 7.002e-01 | 1.690e-01 ($K=10^{-3}$) | 7.002e-01 | 1.864e-01 | **1.566e-01** | 1.692e-01 | 2.087e-01 | **1.566e-01** ($\lambda=10^{-3}$) |
| **51×51** | **20 dB** | 3.169e-01 | 7.004e-01 | 2.299e-01 ($K=10^{-2}$) | 7.004e-01 | 2.727e-01 | 1.811e-01 | **1.729e-01** | 2.090e-01 | **1.729e-01** ($\lambda=10^{-2}$) |

#### Key Experimental Observations:
1. **Clean Restoration**:
   On noise-free blur with the 7×7 kernel, unregularised CLS ($\lambda=0$) recovers the sharp plate to within machine precision ($3.271\times 10^{-5}$ RMSE), exactly matching the inverse filter. Wiener filtering with $K=10^{-4}$ incurs $1.459\times 10^{-2}$ RMSE due to unnecessary flat damping across the passband.
2. **Noise Robustness at 30 dB**:
   On the 7×7 blur at 30 dB BSNR, inverse filtering explodes to $0.6985$ RMSE. Wiener filter achieves $7.692\times 10^{-2}$ RMSE ($22.28$ dB PSNR). CLS with $\lambda=10^{-3}$ achieves **$5.070\times 10^{-2}$ RMSE ($25.90$ dB PSNR)**, a **+3.62 dB improvement** over Wiener.
3. **Severe Noise at 20 dB**:
   On the 7×7 blur at 20 dB BSNR, CLS with $\lambda=10^{-2}$ outperforms Wiener ($7.245\times 10^{-2}$ vs $8.973\times 10^{-2}$ RMSE), delivering a **+1.86 dB PSNR improvement**.
4. **Large Blur (51×51) at 20 dB**:
   Under severe blur and noise (51×51 at 20 dB), Wiener filtering yields $0.2299$ RMSE ($12.77$ dB PSNR). CLS with $\lambda=10^{-2}$ achieves **$0.1729$ RMSE ($15.25$ dB PSNR)** — a **24.8% reduction in error** and a **+2.48 dB PSNR improvement**.
5. **Optimal Parameter Scaling**:
   The optimal $\lambda^*$ scales consistently with noise power: $\lambda^* \approx 10^{-3}$ at 30 dB BSNR, and $\lambda^* \approx 10^{-2}$ at 20 dB BSNR, mirroring the decade step in noise variance.

### 17.9 Limitations and failure modes

While CLS significantly outperforms Wiener filtering in visual sharpness and quantitative metrics on license plates, it has three inherent limitations:

1. **Ringing artefacts around high-contrast edges**:
   Because CLS is a linear frequency-domain filter, severe blur truncation and sharp transition steps at plate borders can induce Gibbs ringing oscillations if $\lambda$ is chosen too small.
2. **Over-smoothing of character strokes**:
   If $\lambda$ is chosen too large ($\lambda \ge 0.1$), the Laplacian penalty excessively damps high-frequency character strokes, causing thin characters (e.g. '1', 'I', 'T') to lose contrast against the background.
3. **No interior non-negativity constraint**:
   Like inverse and Wiener filtering, CLS assumes an unconstrained linear quadratic objective. Negative excursions in dark characters are clipped to $0.0$ only post-hoc via `clip_unit_interval()`. Enforcing non-negativity inside the restoration model itself requires iterative non-linear methods, specifically Richardson–Lucy (step 4.5).

---

## 18. Phase 4.4 implementation log

* **Date**: November 2026
* **Commit**: `feat(restoration): implement constrained least squares filtering`
* **Author**: Ashish (2026909001)

### Decisions made and rationale:

1. **Separate operator response helper**:
   `validate_psf()` enforces $\sum h = 1$ and non-negative weights. Because the discrete Laplacian contains negative entries and sums to $0$, passing it to `validate_psf()` would violate the API contract. Implemented `operator_frequency_response(operator, image_shape)` as an unconstrained operator response helper with identical grid alignment, phase conventions, and circular shifts as `psf_frequency_response()`.
2. **Analytical and empirical Laplacian validation**:
   Verified that the sampled DFT of `DEFAULT_LAPLACIAN` matches the analytical closed-form expression $4 - 2\cos(2\pi u / M) - 2\cos(2\pi v / N)$ to machine precision ($< 10^{-15}$).
3. **Strict DC gain of 1.0**:
   Ensured that $C(0, 0) = 0$, guaranteeing that $|H(0,0)|^2 + \lambda |C(0,0)|^2 = 1.0$ for all $\lambda \ge 0$. Constant images are recovered bit-identically without attenuation.
4. **Zero-gain convention at exact nulls**:
   For $\lambda = 0$ at true frequency zeros ($H(u, v) = 0$), division $0 / 0$ is defined as zero gain ($0 / 0 \to 0$), matching `wiener_filter()`.
5. **Denominator positivity**:
   Proved and verified that for all $\lambda > 0$, the denominator is strictly positive across the entire 2-D grid without requiring an arbitrary epsilon clamp.
6. **Hypothesis property tests**:
   Added property-based tests in `tests/test_cls_filter.py` verifying that `cls_filter` always returns valid `float32` arrays bounded within $[0, 1]$ across arbitrary image sizes and non-negative regularisation values.
7. **Regression and test suite status**:
   Added 42 unit tests in `tests/test_cls_filter.py`. Full test suite: **376 passed, 6 skipped** (zero regressions across foundation, audit, inverse filter, Wiener filter, degradation, or renderer tests).

---

## 19. Richardson–Lucy deconvolution (step 4.5)

Richardson–Lucy (RL) deconvolution — independently derived by William Richardson (1972) and Leon Lucy (1974) — is the classical iterative, non-linear restoration algorithm based on maximum likelihood estimation under a Poisson observation model.

Unlike the linear frequency-domain filters implemented in earlier phases (inverse, Wiener, and CLS), which restore images in a single direct FFT step, Richardson–Lucy solves the deconvolution problem through iterative expectation-maximization updates that inherently enforce a **physical non-negativity constraint** ($f(x, y) \ge 0$).

### 19.1 Problem formulation and Poisson likelihood

Let $f \in \mathbb{R}^{M \times N}$ be the true sharp image (representing latent photon emission intensities), $h$ the blur PSF, and $g$ the observed degraded image:

$$g = h * f + n$$

In optical imaging systems (cameras, microscopes, telescopes), photon arrivals at detector pixel $i$ follow a **Poisson distribution** with rate parameter given by the forward blurred intensity $(Hf)_i$:

$$g_i \sim \operatorname{Poisson}((Hf)_i)$$

The likelihood of observing image $g$ given latent image $f$ is:

$$p(g \mid f) = \prod_{i=1}^{MN} \frac{((Hf)_i)^{g_i} e^{-(Hf)_i}}{g_i!}$$

Taking the natural logarithm yields the log-likelihood objective:

$$\ln p(g \mid f) = \sum_{i=1}^{MN} \Big[ g_i \ln\big((Hf)_i\big) - (Hf)_i - \ln(g_i!) \Big]$$

### 19.2 Mathematical derivation of the multiplicative update

To find the maximum likelihood estimate, compute the partial derivative of the log-likelihood with respect to each pixel $f_j$:

$$\frac{\partial \ln p(g \mid f)}{\partial f_j} = \sum_{i=1}^{MN} \left[ \frac{g_i}{(Hf)_i} \frac{\partial (Hf)_i}{\partial f_j} - \frac{\partial (Hf)_i}{\partial f_j} \right]$$

Since $(Hf)_i = \sum_k H_{i, k} f_k$, the derivative is $\frac{\partial (Hf)_i}{\partial f_j} = H_{i, j}$. Substituting:

$$\frac{\partial \ln p(g \mid f)}{\partial f_j} = \sum_{i=1}^{MN} H_{i, j} \frac{g_i}{(Hf)_i} - \sum_{i=1}^{MN} H_{i, j}$$

Define the detector sensitivity at pixel $j$ as $s_j \triangleq \sum_i H_{i, j} = (H^T \mathbf{1})_j$. At a critical point, the gradient must vanish:

$$\sum_{i=1}^{MN} H_{i, j} \frac{g_i}{(Hf)_i} = s_j$$

Multiplying both sides by $f_j$ and dividing by $s_j$ yields the fixed-point iterative update equation:

$$f_j^{(k+1)} = \frac{f_j^{(k)}}{s_j} \sum_{i=1}^{MN} H_{i, j} \frac{g_i}{(Hf^{(k)})_i}$$

In vector-matrix notation:

$$f^{(k+1)} = \frac{f^{(k)}}{H^T \mathbf{1}} \odot H^T \left[ \frac{g}{H f^{(k)}} \right]$$

where $\odot$ denotes element-wise multiplication.

### 19.3 Forward operator $H$ and exact adjoint $H^T$

Under circular boundary conditions, 2-D spatial convolution with blur kernel $h$ corresponds to point-wise multiplication in the discrete Fourier domain by the transfer function $H(u, v)$ computed via `psf_frequency_response`:

$$\mathcal{F}\{H x\} = H(u, v) \cdot \mathcal{F}\{x\} \implies H x = \operatorname{Re}\Big(\mathcal{F}^{-1}\{H(u, v) \cdot \mathcal{F}\{x\}\}\Big)$$

In real inner-product space $\mathbb{R}^{M \times N}$, the adjoint operator $H^T$ is defined by the duality identity:

$$\langle Hx, y \rangle = \langle x, H^T y \rangle \qquad \forall x, y$$

By Parseval's theorem:

$$\sum_{m, n} (Hx)_{m, n} y_{m, n} = \frac{1}{MN} \sum_{u, v} \big(H(u, v) X(u, v)\big) Y^*(u, v) = \frac{1}{MN} \sum_{u, v} X(u, v) \big(H^*(u, v) Y(u, v)\big)^*$$

Therefore, the adjoint operator in the Fourier domain is multiplication by the **complex conjugate transfer function** $H^*(u, v)$:

$$\mathcal{F}\{H^T y\} = H^*(u, v) \cdot \mathcal{F}\{y\} \implies H^T y = \operatorname{Re}\Big(\mathcal{F}^{-1}\{H^*(u, v) \cdot \mathcal{F}\{y\}\}\Big)$$

In the spatial domain, for real $h$, multiplying by $H^*$ corresponds to convolution with the spatially reflected kernel $h^T(x, y) = h(-x, -y)$ (i.e. cross-correlation).

#### Sensitivity Normalisation ($s = H^T \mathbf{1}$) and Stored PSF Tolerance:
In classical RL theory assuming continuous energy-normalized convolution, $H^T \mathbf{1} = \mathbf{1}$. However, in real digital systems, stored PSFs are validated under tolerance $| \sum h - 1 | \le 10^{-5}$ (`PSF_SUM_TOL`), and project rules strictly forbid renormalizing stored PSFs.

Under circular boundary conditions without spatial windowing (`profile="none"`), the sensitivity map is:
$$s(x, y) = (H^T \mathbf{1})[x, y] = \operatorname{Re}\Big(\mathcal{F}^{-1}\{H^*(u, v) \cdot \mathcal{F}\{\mathbf{1}\}\}\Big)$$
Because $\mathcal{F}\{\mathbf{1}\}[0, 0] = MN$ and zero elsewhere, multiplication by $H^*(u, v)$ isolates the DC bin $H^*(0, 0) = \sum h$. The inverse transform yields an identically constant spatial field:
$$s(x, y) \equiv \operatorname{Re}(H^*(0, 0)) = \sum h \qquad \forall (x, y)$$

*The Danger of Omitting Sensitivity Normalisation:*
If $\sum h = 1.000008 \ne 1.0$, omitting division by $s$ causes the true scene $f^*$ ($g = H f^*$) to update as:
$$f^{(1)} = f^* \odot H^T \left[\frac{g}{Hf^*}\right] = f^* \cdot (H^T \mathbf{1}) = f^* \cdot 1.000008$$
Over $k$ iterations, this produces an uncontrolled geometric photometric drift of $(1.000008)^k$ (or decay if $\sum h < 1.0$).
By explicitly dividing by $\max(H^T \mathbf{1}, \varepsilon) = \sum h$:
$$\frac{H^T [ g / Hf^* ]}{H^T \mathbf{1}} = \frac{1.000008}{1.000008} \equiv 1.0000000000000000$$
Scale invariance and flux conservation are maintained exactly to double precision without altering the stored PSF.

*Condition Under Which Normalisation Can Be Omitted:*
Division by $H^T \mathbf{1}$ can be omitted if and only if $\sum h = 1.000000000$ exactly AND no spatial edge weighting/windowing is applied.

### 19.4 Initialization and numerical safeguards

1. **Initial Estimate $f^{(0)}$**:
   The default initialisation is $f^{(0)} = \max(g, \varepsilon)$ (the observed degraded image, with lower floor $\varepsilon$). In our experiments, initializing from the observation converges significantly faster than initializing with a flat mean image. An explicit custom initial estimate can also be supplied via `initial_estimate=`.
2. **Division by Zero Protection**:
   In the ratio $g / Hf^{(k)}$, if predicted observation $(Hf^{(k)})_i$ approaches zero, division would diverge. A lower floor $\varepsilon = 10^{-12}$ (`eps`) is applied:
   $$p_{\text{safe}}^{(k)} = \max(H f^{(k)}, \varepsilon)$$
3. **FFT Round-Off Protection**:
   Due to floating-point precision, spatial IFFT operations can produce tiny negative numbers (e.g. $-10^{-16}$). Clamping the correction factor $\text{corr}_{\text{safe}} = \max(H^T [g / p_{\text{safe}}], 0.0)$ guarantees that the multiplicative multiplier is strictly non-negative.
4. **Flux Conservation**:
   For any normalized PSF where $\sum h = 1.0$, the total image flux is mathematically conserved at every iteration:
   $$\sum_{j} f_j^{(k+1)} = \sum_{j} f_j^{(k)} \left( H^T \left[ \frac{g}{Hf^{(k)}} \right] \right)_j = \left\langle f^{(k)}, H^T \left[ \frac{g}{Hf^{(k)}} \right] \right\rangle = \left\langle Hf^{(k)}, \frac{g}{Hf^{(k)}} \right\rangle = \sum_{i} g_i$$

### 19.4.1 Edge Tapering Restriction (Option B vs Option A)

In linear frequency-domain filters (Inverse, Wiener, CLS), spatial edge tapering multiplies the observation by a window $w$ to eliminate boundary jump discontinuities before FFT division. In Stage 4.5, `taper_edges` was invoked on $g$ when `profile != "none"`, but the forward model $Hf$ remained unwindowed.

We investigated two engineering remedies:

* **Option A (Consistent Weighted Observation Model):**
  If $g_w = W g$ where $W = \operatorname{diag}(w)$, the forward operator is $A = WH$, the adjoint is $A^T = H^T W$, and the sensitivity map is $s = A^T \mathbf{1} = H^T w$.
  *Defect:* Near image borders, $w(x, y) \to 0$, causing $s(x, y) = (H^T w)(x, y)$ to fall towards zero ($s_{\min} \sim 1.6 \times 10^{-4}$ for Tukey, $2.7 \times 10^{-8}$ for Hann). Dividing by near-zero sensitivity in the update $f \leftarrow f \odot (A^T r / s)$ produces explosive boundary division instability, blowing reconstructed pixel values up to $15.5$ and propagating ringing into the interior (measured interior RMSE: $0.4916$).
* **Option B (Explicit Restriction to `profile="none"`):**
  Synthetic license plates in `plates.npz` already possess an 8 px uniform grey margin (`RGB = 0.50196`). Measured across the entire dataset, opposite edges match with a wrap step of $0.0000$, making circular boundary convolution exact. Spatial tapering is physically unnecessary and fundamentally incompatible with stationary Poisson likelihood modeling.
  *Decision:* `richardson_lucy_filter` explicitly restricts `profile="none"`. Any non-none profile (`"tukey"`, `"linear"`, `"hann"`) raises a descriptive `ValueError`.
   Numerical tests confirm this holds to float32 precision ($7.52 \times 10^{-9}$ relative error).
5. **Physical Range Clipping**:
   The final returned image is strictly clipped to $[0, 1]$ via `clip_unit_interval()`. Optional intermediate clipping (`clip_intermediate=True`) constrains overshoots on bright backgrounds during iterations.

### 19.5 Semi-convergence and noise-model mismatch

A defining characteristic of Richardson–Lucy is **semi-convergence** when applied to noisy data:
* **Early iterations ($k \le 15\text{--}30$)**: The algorithm rapidly recovers lost high-frequency edge information and sharpens character strokes.
* **Late iterations ($k > 50$)**: Because RL has no explicit regularisation penalty on spatial roughness (unlike CLS's $\lambda |C|^2$ or Wiener's $K$), it attempts to fit every noisy fluctuation in $g$. High-frequency noise begins to amplify exponentially, forming speckled "checkerboard" artefacts.

#### Noise-Model Mismatch:
Richardson–Lucy assumes Poisson photon-counting noise. Our project's synthetic degradation pipeline generates additive Gaussian noise with saturation compensation.
* On **Poisson noise**, noise variance is proportional to signal intensity ($\sigma_i^2 = \mu_i$), perfectly matching RL's logarithmic gradient.
* On **additive Gaussian noise**, noise variance is constant across all pixels ($\sigma_n^2 = \text{const}$). RL treats dark background ink with the same relative noise scaling as bright foreground pixels, causing dark stroke regions to develop grain if iterated too far.

### 19.6 Even-sized PSFs and sub-pixel centring

Factor-2 decimated images in the benchmark grid frequently use even-sized blur kernels (e.g. $18 \times 18$). By default, `centre="auto"` applies a phase ramp $R(u, v) = e^{2\pi i (\delta_y u / M + \delta_x v / N)}$ to compensate for the half-pixel offset of the kernel mass.
* In RL, forward projection uses $H(u, v)$ and back-projection uses $H^*(u, v)$.
* Centroid drift measurements:
  * With `centre="auto"`: centroid drift is **$< 0.0001$ px**.
  * With `centre="pixel"`: centroid drift is **$+0.5011$ px** vertically and **$+0.5040$ px** horizontally.

### 19.7 Controlled experimental evaluation

#### 1. Iteration Study Across Blur Kernels and Noise Levels (`seed=7`):

| Kernel | Noise Level | Degraded Input | Iteration 1 | Iteration 5 | Iteration 15 | Iteration 30 | Iteration 50 | Iteration 100 | Best Iteration ($\text{Iter}^*$) |
|---|---|---|---|---|---|---|---|---|---|
| **7×7** | **Clean** | 1.106e-01 (19.1 dB) | 9.500e-02 | 7.061e-02 | 4.986e-02 | 3.876e-02 | 3.340e-02 | **2.981e-02** (30.5 dB) | **100** (2.981e-02) |
| **7×7** | **30 dB** | 1.110e-01 (19.1 dB) | 9.538e-02 | 7.162e-02 | 5.292e-02 | 4.557e-02 | **4.551e-02** (26.8 dB) | 5.253e-02 | **50** (4.551e-02) |
| **7×7** | **20 dB** | 1.148e-01 (18.8 dB) | 9.943e-02 | 7.894e-02 | **7.026e-02** (23.1 dB) | 7.702e-02 | 9.095e-02 | 1.201e-01 | **15** (7.026e-02) |
| **51×51** | **Clean** | 3.164e-01 (10.0 dB) | 3.111e-01 | 2.982e-01 | 2.797e-01 | 2.629e-01 | 2.494e-01 | **2.313e-01** (12.7 dB) | **100** (2.313e-01) |
| **51×51** | **30 dB** | 3.164e-01 (10.0 dB) | 3.111e-01 | 2.982e-01 | 2.796e-01 | 2.629e-01 | 2.493e-01 | **2.312e-01** (12.7 dB) | **100** (2.312e-01) |
| **51×51** | **20 dB** | 3.169e-01 (10.0 dB) | 3.114e-01 | 2.985e-01 | 2.799e-01 | 2.632e-01 | 2.497e-01 | **2.319e-01** (12.7 dB) | **100** (2.319e-01) |

*Key Findings:*
1. **Semi-convergence illustrated:** On the 7×7 kernel at 20 dB BSNR, RMSE drops from $0.1148$ to a minimum of $0.07026$ at 15 iterations, then climbs to $0.1201$ at 100 iterations as noise is amplified.
2. **Slow convergence on large blurs:** On the 51×51 kernel, the first-order multiplicative update converges slowly. 100 iterations reduces RMSE to $0.2312$, whereas CLS resolves the global deconvolution in a single pass ($0.1566$).

#### 2. Multi-Seed 4-Way Method Comparison (10 Random Seeds):

| Kernel | Noise Level | Degraded Input | Inverse Filter | Wiener Filter | CLS Filter | RL (Fixed Deployable) | RL (Oracle Swept) |
|---|---|---|---|---|---|---|---|
| **7×7** | **30 dB** | $1.183\times 10^{-1}$ | $6.963\times 10^{-1}$ | $7.678\times 10^{-2}$ | **$4.897\times 10^{-2}$** | $5.273\times 10^{-2}$ (it=30) | $5.139\times 10^{-2}$ |
| **7×7** | **20 dB** | $1.220\times 10^{-1}$ | $6.990\times 10^{-1}$ | $8.869\times 10^{-2}$ | **$7.161\times 10^{-2}$** | $7.585\times 10^{-2}$ (it=20) | $7.535\times 10^{-2}$ |
| **51×51** | **30 dB** | $3.118\times 10^{-1}$ | $7.008\times 10^{-1}$ | $1.697\times 10^{-1}$ | **$1.512\times 10^{-1}$** | $2.603\times 10^{-1}$ (it=30) | $2.476\times 10^{-1}$ |
| **51×51** | **20 dB** | $3.124\times 10^{-1}$ | $7.005\times 10^{-1}$ | $2.299\times 10^{-1}$ | **$1.727\times 10^{-1}$** | $2.706\times 10^{-1}$ (it=20) | $2.483\times 10^{-1}$ |

#### 3. Poisson Noise Model Study:

##### 3.1 Mathematical Audit of Stage 4.5 Upper-Tail Clipping Defect:
In Stage 4.5, Poisson noise was simulated by drawing integer photon counts $C \sim \operatorname{Poisson}(N \cdot b)$ and normalizing via $\hat{g} = \operatorname{clip}(C / N, 0.0, 1.0)$.

*The Mathematical Flaw:*
A Poisson random variable is unbounded above ($\operatorname{supp}(C) = \mathbb{N}_0$). For pixels where the blurred intensity is at or near the plate background maximum ($b(x, y) \approx 1.0$), the rate parameter is $\lambda = N$ with standard deviation $\sigma = \sqrt{N}$. Approximately $50\%$ of white pixels naturally fluctuate above $N$ ($C > N \implies C / N > 1.0$).
* Quantitative Measurements (Synthetic Plate, $96 \times 336 = 32,256$ pixels):
  * **Exceeding Pixels:** Exactly **$7,851$ pixels (24.34%)** exceeded $1.0$ at $N=1,000$ photons, reaching values up to $1.1390$.
  * **Photometric Bias:** On white background pixels ($n=15,400$), clipping introduced a systematic negative bias of **$-0.0125$** (mean reduced from $1.00025$ to $0.98751$).
  * **Severe Variance Truncation:** On white pixels, clipping eliminated the upper half of the Poisson distribution, reducing variance from theoretical $1.000\times 10^{-3}$ to $3.321\times 10^{-4}$ — a **$66.79\%$ variance reduction**!
  * **Artifact:** Clipping acted as an artificial non-linear noise suppressor, artificially flattering Stage 4.5 restoration scores. Stage 4.5 single-seed results are therefore designated as **SUPERSEDED**.

##### 3.2 Corrected Methodology 1: Calibrated Photometric Headroom Scaling (10 Seeds, Public API):
To satisfy the project's strict $[0, 1]$ image contract (`validate_image`) without silent clipping, the latent scene is exposed with calibrated photometric headroom $s_{\text{scale}} = 0.70$.

*Mathematical Derivation of Effective Photon Budget:*
Sensor count draws are $C(x, y) \sim \operatorname{Poisson}(N \cdot s_{\text{scale}} \cdot b(x, y))$ where $b(x, y) \in [0, 1]$ is the blurred plate and $N$ is the nominal photon parameter.
The normalized observation passed to the public filter is $g = C / N$. Its expectation and variance are:
$$\mathbb{E}[g] = s_{\text{scale}} \cdot b(x, y), \qquad \operatorname{Var}(g) = \frac{s_{\text{scale}} \cdot b(x, y)}{N}$$
The restored estimate $\hat{f}_{\text{scaled}}$ resides in the scaled intensity domain $[0, s_{\text{scale}}]$ and is calibrated back to unit scale via $\hat{f} = \operatorname{clip}(\hat{f}_{\text{scaled}} / s_{\text{scale}}, 0.0, 1.0)$.
Expressed on the original unit intensity scale, the effective observation is $g_{\text{unit}} = g / s_{\text{scale}} = \frac{C}{N s_{\text{scale}}}$, whose variance is:
$$\operatorname{Var}(g_{\text{unit}}) = \operatorname{Var}\left(\frac{C}{N s_{\text{scale}}}\right) = \frac{N s_{\text{scale}} b(x, y)}{(N s_{\text{scale}})^2} = \frac{b(x, y)}{N s_{\text{scale}}}$$
Comparing this to an unscaled Poisson observation model where $\operatorname{Var}(g_0) = \frac{b(x, y)}{N_{\text{eff}}}$, we have:
$$N_{\text{eff}} = s_{\text{scale}} \cdot N$$
Therefore, **headroom scaling changes the effective photon budget to $N_{\text{eff}} = s_{\text{scale}} N$ rather than preserving the nominal budget $N$**:
* Nominal $N = 1,000 \implies N_{\text{eff}} = 700$ photons ($30\%$ flux reduction).
* Nominal $N = 5,000 \implies N_{\text{eff}} = 3,500$ photons.
* Nominal $N = 20,000 \implies N_{\text{eff}} = 14,000$ photons.
Headroom scaling preserves the scale invariance of the Richardson–Lucy multiplicative ratio ($\frac{s g}{s H f} = \frac{g}{H f}$), but the noise variance corresponds to effective flux $N_{\text{eff}}$.

*Rare-Event Handling and Qualified Exceedance Bounds:*
Because the Poisson distribution has unbounded support over all $\mathbb{N}_0$, no finite headroom factor can provide a mathematical guarantee of zero exceedance ($P(C > N) \equiv 0$ is mathematically impossible).
However, setting $s_{\text{scale}} = 0.70$ provides an $11.33\sigma$ headroom margin below the $1.0$ saturation threshold at nominal $N=1,000$ (where $\mu = 700$, $\sigma = \sqrt{700} \approx 26.46$, and the margin is $(1000 - 700) / 26.46 \approx 11.34$).
By the Chernoff / Bennett bound for Poisson distributions:
$$P(C \ge N) \le \exp\left(-\mu \left((1+\delta)\ln(1+\delta) - \delta\right)\right) \le 2.3 \times 10^{-25} \quad \text{per pixel}$$
By the union bound across all $32,256$ pixels of a $96 \times 336$ license plate frame:
$$P(\text{frame exceedance}) \le 32256 \times 2.3 \times 10^{-25} \le 7.4 \times 10^{-21}$$
At $N=5,000$, the margin is $25.35\sigma$ ($P < 10^{-130}$); at $N=20,000$, the margin is $50.70\sigma$ ($P < 10^{-500}$).
*Explicit Failure on Exceedance:* Raw Poisson observations are **never silently clipped**. If an observation array ever has $\max(g) > 1.0$ or $\min(g) < 0.0$, the experiment raises an explicit `RuntimeError` and halts the trial. Across all 10 evaluation seeds (30 trials total), $0$ trials exceeded $[0, 1]$.
*Distinction in Clipping Policy:* Raw observations are strictly unclipped. Final unit-interval clipping (`clip_unit_interval`) applies conceptually and strictly only to the post-restoration rescaled estimate ($\hat{f}_{\text{scaled}} / s_{\text{scale}}$).
*Practical API Limitation:* A bounded $[0, 1]$ public image processing API cannot represent every possible realization of an unbounded Poisson process without either headroom scaling (which reduces effective flux) or silent truncation (which introduces bias and cuts variance).

*Audit of Wiener and CLS Parameter Scaling:*
* **CLS Filter Regularization ($\lambda$):** The unconstrained CLS objective is $\min_f \|g - Hf\|_2^2 + \lambda \|Cf\|_2^2$. Under uniform image scaling $g \to s g$, both the residual term $\|s g - H (s f)\|^2 = s^2 \|g - Hf\|^2$ and the regularization term $\lambda \|C (s f)\|^2 = s^2 \lambda \|Cf\|^2$ scale by $s^2$. The minimizer scales as $s f^*$. Dividing by $s$ post-hoc recovers $f^*$ identically. Hence, **CLS regularization $\lambda$ is strictly invariant to uniform amplitude scaling $s$ and requires no adjustment**.
* **Wiener Filter Parameter ($K$):** The scalar Wiener regularizer represents the noise-to-signal power ratio $K = \sigma_n^2 / \sigma_f^2$. Under headroom scaling, noise variance scales as $s$ ($\sigma_n^2 = s \bar{b} / N$), while signal power scales as $s^2$ ($\sigma_{f_{\text{scaled}}}^2 = s^2 \sigma_f^2$). The ratio scales as $s / s^2 = 1/s$:
  $$K(N_{\text{eff}}) = \frac{\bar{b}}{s N \sigma_f^2} = \frac{\bar{b}}{N_{\text{eff}} \sigma_f^2} = \frac{1}{s} K_{\text{nominal}}$$
  Because $N_{\text{eff}} = s N$, $K$ is inversely proportional to the effective photon budget.
* **Why Deployable Parameters Were Kept Fixed:** In our experimental methodology, deployable configurations represent fixed, $a\ priori$ parameter choices made without ground-truth clean plate access ($K = 10^{-3}$, $\lambda = 10^{-3}$, $k = 20$). Adjusting $K$ post-hoc solely to make Wiener score better at low flux would violate the deployable evaluation protocol. At $N=1,000$ ($N_{\text{eff}}=700$), the true BSNR is only $\approx 17.5$ dB, so $K=10^{-3}$ severely under-regularizes (RMSE $0.2521$). The oracle sweep ($K \in \{10^{-4}, 10^{-3}, 10^{-2}\}$) demonstrates that selecting an appropriate $K=10^{-2}$ restores Wiener performance to $0.0928$ RMSE.

Multi-seed evaluation across 10 independent random seeds (`seed=0..9`):

| Nominal Flux $N$ | Effective Flux $N_{\text{eff}}$ | Degraded Input | Wiener (Deployable, $K=10^{-3}$) | CLS (Deployable, $\lambda=10^{-3}$) | RL (Deployable, $k=20$) | Wiener (Oracle) | CLS (Oracle) | RL (Oracle) | Relative Performance |
|---|---|---|---|---|---|---|---|---|---|
| **1,000 photons** | **700 photons** | $0.1207 \pm 0.0045$ | $0.2521 \pm 0.0019$ | $0.0875 \pm 0.0011$ | **$0.0668 \pm 0.0037$** | $0.0928 \pm 0.0010$ | $0.0707 \pm 0.0019$ | **$0.0668 \pm 0.0037$** | **RL Deployable wins (+2.3 dB over CLS)** |
| **5,000 photons** | **3,500 photons** | $0.1184 \pm 0.0047$ | $0.1174 \pm 0.0008$ | **$0.0554 \pm 0.0011$** | $0.0560 \pm 0.0045$ | $0.0627 \pm 0.0011$ | $0.0554 \pm 0.0011$ | **$0.0504 \pm 0.0040$** | **CLS Deployable / RL Oracle** |
| **20,000 photons** | **14,000 photons** | $0.1180 \pm 0.0047$ | $0.0641 \pm 0.0006$ | **$0.0470 \pm 0.0014$** | $0.0537 \pm 0.0047$ | $0.0546 \pm 0.0013$ | $0.0470 \pm 0.0014$ | **$0.0441 \pm 0.0047$** | **CLS Deployable wins (+1.1 dB over RL)** |

##### 3.3 Methodology 2: Unconstrained Photon-Count Domain RL (10 Seeds, Theoretical Reference):
As a secondary theoretical benchmark, RL was executed directly on unbounded integer counts $C \sim \operatorname{Poisson}(N \cdot b)$ without headroom scaling:
$$f_{\text{count}}^{(k+1)} = f_{\text{count}}^{(k)} \odot \frac{H^T \left[ \frac{C}{\max(Hf_{\text{count}}^{(k)}, \varepsilon)} \right]}{H^T \mathbf{1}}, \qquad \hat{f} = \operatorname{clip}(f_{\text{count}} / N, 0.0, 1.0)$$

*Effective Budget Note & Non-Like-for-Like Comparison:*
This reference experiment operates directly with rate $\lambda = N \cdot b$, so its effective photon budget is $N$ ($1,000, 5,000, 20,000$).
Because the headroom experiment has effective budget $N_{\text{eff}} = s_{\text{scale}} N$ ($700, 3,500, 14,000$), this unconstrained count benchmark possesses a $1 / s_{\text{scale}} \approx 1.43\times$ higher photon count. It is a theoretical reference ceiling, **not a like-for-like comparison at the same nominal label**.

Results across 10 random seeds (`seed=0..9`):
* **1,000 nominal photons (Effective flux 1,000):** Degraded: $0.1199 \pm 0.0047$ | Deployable RL ($k=20$): **$0.0630 \pm 0.0039$** | Oracle RL ($k=20$): **$0.0623 \pm 0.0037$**
* **5,000 nominal photons (Effective flux 5,000):** Degraded: $0.1183 \pm 0.0047$ | Deployable RL ($k=20$): **$0.0551 \pm 0.0046$** | Oracle RL ($k=30$): **$0.0479 \pm 0.0043$**
* **20,000 nominal photons (Effective flux 20,000):** Degraded: $0.1179 \pm 0.0047$ | Deployable RL ($k=20$): **$0.0535 \pm 0.0047$** | Oracle RL ($k=50$): **$0.0435 \pm 0.0048$**

*API Limitation Note:* Raw normalized counts $C / N$ exceed $1.0$ (up to $1.1390$) and cannot be consumed by the public filter APIs (`richardson_lucy_filter`, `wiener_filter`, `cls_filter`) due to the $[0, 1]$ range contract enforced by `validate_image`.

##### 3.4 Summary of Poisson Findings:
The Poisson noise experiments demonstrate that algorithm superiority is **conditional on the photon flux regime**:
1. **Low Photon Flux ($N=1,000, N_{\text{eff}}=700$):**
   Richardson–Lucy decisively outperforms both CLS and Wiener filtering (RL Deployable RMSE $0.0668$ vs CLS Deployable $0.0875$, a $+2.3$ dB gain). At low photon counts, Poisson shot noise deviates strongly from Gaussian statistics, and RL's Poisson log-likelihood model with natural non-negativity constraint provides superior restoration.
2. **Moderate to High Photon Flux ($N=5,000$ and $20,000, N_{\text{eff}}=3,500$ and $14,000$):**
   By the Central Limit Theorem, high-count Poisson noise approaches an additive Gaussian distribution ($\operatorname{Poisson}(\lambda) \approx \mathcal{N}(\lambda, \lambda)$). In this regime, Constrained Least Squares (CLS) deployable achieves lower RMSE than fixed-iteration Richardson–Lucy ($0.0554$ vs $0.0560$ at $N=5,000$; $0.0470$ vs $0.0537$ at $N=20,000$).
3. **Semi-Convergence Sensitivity of Fixed-Iteration RL:**
   Unregularised RL requires tuning iteration count to the noise level: $k=20$ is optimal at low flux ($N=1,000$), but under-iterates at high flux ($N=20,000$), where RL Oracle requires $50$ iterations to reach $0.0441$. CLS's explicit Laplacian smoothness prior ($\lambda |C|^2$) suppresses high-frequency noise across plate interiors in a single direct pass without iteration tuning.

---

### 19.8 Four-way algorithm comparison matrix

| Dimension | Inverse Filter | Wiener Filter | CLS Filter | Richardson–Lucy |
|---|---|---|---|---|
| **Type** | Linear, direct | Linear, direct | Linear, direct | Non-linear, iterative |
| **Optimization Criterion** | $\min \|g - h*f\|^2$ | $\min \mathbb{E}[\|f - \hat{f}\|^2]$ | $\min \|c*f\|^2$ s.t. $\|g - h*f\|^2 = \|n\|^2$ | $\max \ln p(g \mid f)$ (Poisson ML) |
| **Noise Assumption** | None (noise ignored) | Additive Gaussian (flat prior) | Additive Gaussian (smooth prior) | Poisson photon counting |
| **Regularisation Parameter** | Denominator floor $\varepsilon$ | Constant scalar $K$ | High-pass penalty $\lambda |C|^2$ | Iteration count $k$ |
| **DC Gain at $(0, 0)$** | $1.0$ | $1 / (1 + K) < 1.0$ | $1.0$ identically | $1.0$ (flux conserving) |
| **Non-Negativity ($f \ge 0$)** | Post-hoc clip only | Post-hoc clip only | Post-hoc clip only | **Inherent in formulation** |
| **Computational Complexity** | 2 FFTs (1 pass, $\sim 2$ ms) | 2 FFTs (1 pass, $\sim 2$ ms) | 2 FFTs (1 pass, $\sim 2$ ms) | 4 FFTs per iteration ($\sim 80$ ms for 30 iters) |
| **Best Used For** | Noise-free images | Moderate Gaussian noise | Structured plates with Gaussian noise | Low-light photon noise / localized blurs |

---

### 19.9 DIP Viva Preparation: Core Questions & Answers

**Q: What is the fundamental difference between Richardson–Lucy and the other three methods?**
*Answer:* Inverse, Wiener, and CLS are direct, linear frequency-domain filters derived under $L_2$ Euclidean norm minimization. They assume unconstrained pixel values ($-\infty, +\infty$) and enforce $[0, 1]$ range only post-hoc. Richardson–Lucy is an iterative, non-linear maximum likelihood method derived under a Poisson statistics model that inherently enforces a strict non-negativity constraint ($f \ge 0$).

**Q: Why is the Richardson–Lucy update multiplicative rather than additive?**
*Answer:* Because it is derived from the Poisson log-likelihood gradient $\frac{\partial \ln p}{\partial f_j} = \sum_i H_{i, j} \frac{g_i}{(Hf)_i} - s_j$. Setting this to zero and solving as a fixed-point equation yields $f_j^{(k+1)} = \frac{f_j^{(k)}}{s_j} \cdot \left[ H^T (g / Hf) \right]_j$. The multiplicative form guarantees that if $f^{(0)} \ge 0$ and $g \ge 0$, then $f^{(k)} \ge 0$ for all iterations: non-negativity is preserved naturally without projection.

**Q: What does the adjoint operator $H^T$ do, and how is it implemented?**
*Answer:* The adjoint operator back-projects the observation-to-prediction discrepancy ratio $g / Hf$ back into the image space. In the frequency domain, $H^T$ is implemented by multiplying by the complex conjugate $H^*(u, v)$ followed by an inverse FFT. In spatial terms, this is cross-correlation with the PSF (or convolution with the spatially flipped PSF $h(-x, -y)$).

**Q: Why does the sensitivity map $H^T \mathbf{1}$ appear in the general Richardson–Lucy update, and when can it be omitted?**
*Answer:* The sensitivity map $s_j = (H^T \mathbf{1})_j = \sum_i H_{i, j}$ represents the total probability that a photon emitted from pixel $j$ is detected anywhere on the sensor grid. It arises directly from setting the Poisson log-likelihood gradient $\frac{\partial \ln p}{\partial f_j} = \sum_i H_{i, j} \frac{g_i}{(Hf)_i} - s_j$ to zero.
Under circular convolution without spatial windowing, $s(x, y) \equiv \sum h$. If the PSF is normalized so that $\sum h = 1.000000000$ exactly, then $s \equiv 1.0$ and sensitivity division is identity. However, stored PSFs in digital image processing are validated within tolerance $|\sum h - 1| \le 10^{-5}$ and must not be renormalised. Dividing by $s = \sum h$ eliminates cumulative scaling drift of $(\sum h)^k$ across iterations. It can only be omitted when $\sum h = 1.0$ exactly and no spatial mask is applied.

**Q: Why is spatial edge tapering (Tukey/Hann) prohibited in our Richardson–Lucy implementation?**
*Answer:* For two fundamental reasons:
1. *Mathematical inconsistency & boundary explosion:* If an observation is windowed by $W$, the consistent sensitivity map is $s = H^T w$. As $w \to 0$ near image borders, $s \to 0$. Dividing by near-zero sensitivity causes severe boundary division instability (reconstructed pixels spike to 15.5+ and noise explodes inwards).
2. *Dataset physics:* Synthetic license plates feature an 8 px grey margin where opposite borders match with zero jump (measured wrap step 0.0000), making circular convolution exact. Applying a taper attenuates character strokes near borders while offering zero spectral leakage benefit.

**Q: What is the difference between oracle parameter selection and a deployable heuristic?**
*Answer:* An **oracle** benchmark sweeps hyper-parameters (e.g. iteration count $k$, Wiener $K$, CLS $\lambda$) against the uncorrupted ground-truth clean image $f_{\text{true}}$, choosing the parameter that minimizes RMSE. In real deployment, the clean image is unknown, so a **deployable** heuristic selects parameters based solely on observable quantities (such as estimated BSNR or fixed noise-scaled iteration stops). Oracle reporting is useful to measure theoretical ceiling capacity, but deployable benchmarks must always be reported to reflect practical performance.

**Q: What is semi-convergence, and why does it occur in Richardson–Lucy?**
*Answer:* Semi-convergence is the property where restoration quality improves during early iterations but deteriorates during later iterations. Because unregularised Richardson–Lucy has no roughness penalty, maximum likelihood attempts to fit every pixel fluctuation. On noise-free data, likelihood increases monotonically toward the true scene. On noisy data, early iterations recover large-scale glyph features, while late iterations amplify high-frequency noise spikes to fit the noisy observations.

**Q: Why does CLS outperform Richardson–Lucy on our project's license plate dataset?**
*Answer:* For two mathematical reasons: (1) Our degradation pipeline adds Gaussian noise, whereas RL is formulated for Poisson noise. (2) License plates have structured glyph edges where CLS's Laplacian smoothness prior ($\lambda |C|^2$) suppresses high-frequency noise while leaving character boundaries sharp. Furthermore, on large blurs (51×51), CLS inverts global frequencies in a single pass, whereas RL's first-order gradient updates converge very slowly.

**Q: Why does naively clipping Poisson observations to $[0, 1]$ corrupt statistical evaluations?**
*Answer:* Poisson observations $C \sim \operatorname{Poisson}(\lambda)$ have support on all non-negative integers $\mathbb{N}_0$. At pixels with maximum plate intensity ($b=1.0$), $\lambda = N$. Because Poisson fluctuations are symmetric around $N$ for large flux, roughly $50\%$ of white pixels have $C > N$, meaning normalized counts $C / N > 1.0$. On synthetic plates where white background covers $48.7\%$ of pixels, $24.34\%$ of all image pixels exceed $1.0$ (up to $1.1390$). Forcing them into $[0, 1]$ via `clip_unit_interval` introduces a negative photometric bias ($-0.0125$) and truncates $66.79\%$ of the white-pixel noise variance, acting as an artificial non-linear noise filter that misrepresents algorithm performance.

**Q: How does calibrated headroom scaling reconcile the $[0, 1]$ API contract with unbounded Poisson noise?**
*Answer:* Real optical sensors avoid saturation by exposing scenes with headroom below full-well capacity. By scaling the latent scene by $s_{\text{scale}} = 0.70$, the maximum expected count at nominal flux $N=1,000$ is $\mu = 700$, providing an $11.33\sigma$ headroom margin below the $1.0$ saturation threshold. By the Chernoff bound, the per-pixel exceedance probability is $P(C > N) \le 2.3 \times 10^{-25}$, and the frame exceedance probability is $\le 7.4 \times 10^{-21}$ for a $96 \times 336$ license plate.
Unlike naive clipping, raw observations are strictly unclipped: if any pixel exceeds $[0, 1]$, the trial is failed explicitly via a `RuntimeError` rather than silently truncated. Final unit-interval clipping (`clip_unit_interval`) is applied only to the post-restoration rescaled estimate. Because the Richardson–Lucy multiplicative ratio $\frac{s g}{s Hf} = \frac{g}{Hf}$ is scale invariant, headroom scaling preserves exact Poisson deconvolution dynamics while allowing direct, compliant invocation of public filter APIs.

**Q: How does headroom scaling affect the effective photon budget, and why is direct count RL not a like-for-like comparison?**
*Answer:* For observation $g = C / N$ with $C \sim \operatorname{Poisson}(N s b)$, the expected value is $\mathbb{E}[g] = s b$ and variance is $\operatorname{Var}(g) = s b / N$. Dividing the reconstructed estimate by $s$ maps the variance to $\operatorname{Var}(g / s) = b / (N s) = b / N_{\text{eff}}$. Therefore, headroom scaling changes the effective photon budget to $N_{\text{eff}} = s \cdot N$ (700, 3,500, and 14,000 photons for nominal budgets 1,000, 5,000, and 20,000).
The unconstrained count-domain RL benchmark directly draws $C \sim \operatorname{Poisson}(N b)$ where the effective budget is full nominal $N$ ($1,000, 5,000, 20,000$). Consequently, the direct count benchmark operates at a $1 / s = 1.43\times$ higher photon count than the headroom-scaled experiment at the same nominal label, serving as an unconstrained theoretical reference ceiling rather than a direct like-for-like comparison.

**Q: Does CLS regularization or Wiener $K$ require recalibration under headroom scaling?**
*Answer:*
1. *CLS Filter:* In the Lagrangian objective $\min_f \|g - Hf\|^2 + \lambda \|Cf\|^2$, uniform scaling $g \to s g$ scales both the residual and roughness penalty by $s^2$, causing the minimizer to scale linearly as $s f^*$. Dividing by $s$ post-hoc exactly recovers $f^*$. Therefore, CLS regularization $\lambda$ is strictly scale-invariant and requires no recalibration.
2. *Wiener Filter:* The regularizer $K$ represents the noise-to-signal ratio $\sigma_n^2 / \sigma_f^2$. Under headroom scaling, $\sigma_n^2$ scales as $s$ while signal variance scales as $s^2$, so their ratio scales as $1/s$: $K(N_{\text{eff}}) = \frac{\bar{b}}{N_{\text{eff}} \sigma_f^2} = \frac{1}{s} K_{\text{nominal}}$.
In our benchmark, deployable parameters are chosen $a\ priori$ and kept static across noise sweeps ($K = 10^{-3}$, $\lambda = 10^{-3}$, $k = 20$). Recalibrating $K$ post-hoc solely to boost Wiener's score at low flux would violate the deployable evaluation discipline. The oracle sweep ($K \in \{10^{-4}, 10^{-3}, 10^{-2}\}$) demonstrates how tuning $K$ to the noise level recovers performance (reducing Wiener RMSE from $0.2521$ to $0.0928$ at 1,000 photons).

---

## 20. Implementation and Audit Logs

### 20.1 Phase 4.5 Implementation Log

* **Date**: November 2026
* **Commit**: `feat(restoration): implement Richardson-Lucy filtering`
* **Author**: Ashish (2026909001)

#### Decisions made and rationale:

1. **Exact Adjoint Formulation**:
   Verified mathematically and empirically that $H^T y = \operatorname{Re}(\mathcal{F}^{-1}\{H^*(u, v) \mathcal{F}\{y\}\})$ is the exact adjoint of circular convolution $Hx = \operatorname{Re}(\mathcal{F}^{-1}\{H(u, v) \mathcal{F}\{x\}\})$, satisfying $\langle Hx, y \rangle = \langle x, H^T y \rangle$ to $< 10^{-14}$.
2. **Sub-Pixel Centring via `centre="auto"`**:
   Forwarded `centre="auto"` to `psf_frequency_response`, ensuring that even-sized kernels (e.g. $18 \times 18$ from factor-2 decimation) restore without half-pixel centroid displacement ($< 0.0001$ px drift).
3. **Robust Numerical Safeguards**:
   Implemented lower floor $\varepsilon = 10^{-12}$ on predicted observation $Hf$ to prevent division by zero, non-negative clamping on back-projected corrections to guard against FFT numerical round-off artifacts, and validated non-negative integer iteration counts.
4. **Comprehensive Test Suite**:
   Created `tests/test_richardson_lucy.py` containing 59 unit, regression, and property-based tests covering all contracts, mathematical properties, and image resolutions. Full test suite: **435 passed, 6 skipped** (zero regressions across foundation, audit, inverse, Wiener, or CLS tests).
5. **Experimental Benchmarking**:
   Authored `experiments/run_richardson_lucy_experiment.py` and saved full results in `experiments/richardson_lucy_experiment_results.json`. Replicated iteration trajectories, 10-seed comparisons, and Poisson noise experiments.
6. **Independent Audit Report**:
   Authored comprehensive audit report in `docs/richardson_lucy_audit.md`.

### 20.2 Phase 4.5.1 Corrective Hardening Log

* **Date**: November 2026
* **Commit**: `fix(restoration): harden Richardson-Lucy sensitivity model`
* **Author**: Ashish (2026909001)

#### Corrective engineering actions:

1. **Explicit Sensitivity Normalisation ($s = H^T \mathbf{1} = \sum h$):**
   Corrected the assumption that $H^T \mathbf{1} \equiv 1.0$. Because `validate_psf()` permits PSFs within $|\sum h - 1| \le 10^{-5}$ without renormalisation, omitting sensitivity division produced an uncontrolled geometric drift of $(\sum h)^k$. Implemented division by $s = \operatorname{Re}(H^*(0, 0)) = \sum h$ with floor $\varepsilon = 10^{-12}$, guaranteeing exact scale invariance and flux conservation without modifying stored PSFs.
2. **Edge Tapering Restriction (Option B):**
   Enforced strict restriction `profile="none"`. Proved mathematically and empirically that Option A (weighted adjoint $A^T = H^T W$ and sensitivity $H^T w$) causes catastrophic boundary division explosion (reconstructed pixel intensities up to $15.5$, interior RMSE $0.4916$) because sensitivity falls to 0 at borders. In contrast, `profile="none"` exploits the true 8 px grey margin of the dataset where circular wrap error is $0.0000$ (interior RMSE $2.37 \times 10^{-8}$). Unsupported profiles raise clear `ValueError`.
3. **Transparent Oracle vs Deployable Benchmarking:**
   Updated `experiments/run_richardson_lucy_experiment.py` to explicitly distinguish a priori fixed deployable parameters from ground-truth-swept oracle benchmarks in both Gaussian and Poisson noise studies.
4. **Test Suite Expansion:**
   Added 9 comprehensive tests in `tests/test_richardson_lucy.py` covering sensitivity map constancy, photometric drift prevention with PSF sums $1.0 \pm 8 \times 10^{-6}$, exact unity normalisation, even-sized PSF DC preservation, flux conservation under approximate PSF sums, and taper profile rejection.
   **Full test suite: 444 passed, 6 skipped (0 regressions).**

### 20.3 Phase 4.5.2 Poisson Methodology Audit & Correction Log

* **Date**: November 2026
* **Commit**: `fix(experiments): correct Poisson observation handling`
* **Author**: Ashish (2026909001)

#### Corrective engineering actions:

1. **Audit and Quantification of Clipping Defect:**
   Quantified that Stage 4.5's `clip_unit_interval(counts / photons)` clipped 24.34% of image pixels (up to $1.1390$), introducing a $-0.0125$ negative bias and truncating 66.79% of white-pixel variance. Retained superseded results under `superseded_clipped_stage4_5` with explicit status markings.
2. **Corrected Headroom Scaling Methodology (10 Seeds, Public API):**
   Implemented calibrated photometric headroom scaling ($s_{\text{scale}} = 0.70$), providing $> 11\sigma$ margin below sensor saturation ($P(\text{frame exceedance}) \le 7.4 \times 10^{-21}$) and eliminating clipping while remaining compliant with public API filters (`richardson_lucy_filter`, `wiener_filter`, `cls_filter`).
3. **Unconstrained Count-Domain Reference Benchmark (10 Seeds):**
   Implemented direct photon-count domain RL reference benchmark on raw integer counts $C \sim \operatorname{Poisson}(N b)$. Explicitly documented the API limitation that raw counts cannot be consumed by public filters due to `validate_image`'s $[0, 1]$ contract.
4. **Multi-Seed Statistical Reporting:**
   Evaluated all experiments across 10 random seeds (`seed=0..9`), reporting mean, std, min, and max for both deployable configurations and oracle benchmarks.

### 20.4 Phase 4.5.3 Poisson Experiment Final Consistency Audit Log

* **Date**: November 2026
* **Commit**: `fix(experiments): clarify Poisson photon budget and results`
* **Author**: Ashish (2026909001)

#### Corrective engineering and reporting actions:

1. **Effective Photon-Budget Correction:**
   Proved mathematically that scaling by $s_{\text{scale}} = 0.70$ results in an observation variance of $\operatorname{Var}(g / s) = b / (N s)$, establishing that the effective photon budget is $N_{\text{eff}} = s \cdot N$ (700, 3,500, and 14,000 photons for nominal budgets 1,000, 5,000, and 20,000). Corrected all documentation claiming headroom scaling preserves the original photon budget. Clarified that the unconstrained count benchmark operates at full nominal flux $N$ ($1.43\times$ higher flux), making it a theoretical reference rather than a like-for-like comparison at the same nominal label.
2. **Qualified Rare-Event & Clipping Handling:**
   Replaced absolute claims ("guaranteed zero clipping", "100% zero clipping") with mathematically sound probability bounds ($P(C \ge N) \le 2.3 \times 10^{-25}$ per pixel, $P(\text{frame exceedance}) \le 7.4 \times 10^{-21}$ at $N=1,000$ under an $11.33\sigma$ margin). Replaced bare assertions in `run_poisson_noise_experiment` with explicit trial verification that raises `RuntimeError` on exceedance without silent truncation. Maintained conceptual separation between raw observation validation and post-restoration unit-interval clipping (`clip_unit_interval`).
3. **Wiener & CLS Parameter Audit:**
   Derived that CLS regularization $\lambda$ is strictly scale-invariant under uniform amplitude scaling, requiring no adjustment. Proved that Wiener $K$ scales inversely with $N_{\text{eff}}$ ($K(N_{\text{eff}}) = \frac{1}{s} K_{\text{nominal}}$). Confirmed that deployable parameters must remain $a\ priori$ fixed across flux levels ($K=10^{-3}, \lambda=10^{-3}, k=20$) rather than post-hoc tuned, while oracle sweeps reflect the noise-adapted performance ceiling.
4. **Authoritative Results Reconciliation:**
   Reconciled discrepancies between earlier handover text and committed JSON: confirmed authoritative 1,000-photon degraded RMSE is $0.1207 \pm 0.0045$ (not $0.2312$). Corrected broad claims of universal RL superiority: documented that while RL wins decisively at low flux ($N=1,000, N_{\text{eff}}=700$), CLS Deployable outperforms fixed-iteration ($k=20$) RL at moderate and high flux ($N=5,000$ and $20,000$).
5. **Full Repository Verification:**
   Confirmed zero modifications to restoration algorithms or tests. Full test suite: **444 passed, 6 skipped (0 regressions)**. Zero whitespace warnings under `git diff --check`.
