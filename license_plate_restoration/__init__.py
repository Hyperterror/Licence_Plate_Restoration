"""Classical (non-deep-learning) license-plate restoration pipeline -- CS7.404.

Stages owned by this module set
-------------------------------
Step 1  Clean plate acquisition : renderer, ccpd_loader, dataset_writer
Step 2  Degradation            : psf_builder, degradation
Step 5  Binarization           : binarizer

Forward model implemented by steps 1-2::

    g(x, y) = D( h(x, y) * f(x, y) ) + n(x, y)

``f`` is the sharp plate produced here, ``h`` the PSF, ``D`` block-mean
downsampling and ``n`` additive Gaussian noise.

Conventions shared with the other pipeline stages:

* images are grayscale ``float32`` in ``[0, 1]`` (0 = black, 1 = white);
* PSF kernels are odd-sized, centred and sum to 1;
* angles are in degrees, counter-clockwise from the x-axis;
* every plate carries an 8 px grey (128/255) border.

Hand-off contract (settled -- consume as-is, do not re-derive)
-------------------------------------------------------------
* ``binarize`` defaults to ``polarity="auto"`` (minority class of the plate
  interior).  Requirement 8.3's literal "lower class mean" rule remains
  available as ``polarity="dark"``, but it inverts every CCPD plate, which
  has white text on a dark field.
* ``<key>_psf`` in ``degraded.npz`` is already resolution-matched to its
  ``<key>_img``; a ``downsample_factor == 2`` image comes with a roughly
  half-width kernel.
* ``make_motion_psf(length, angle)`` takes a *half*-length, so ``length=15``
  is a 31 px smear.
* ``add_noise`` calibrates its amplitude after clipping
  (``saturation_compensation=False`` disables it), because the raw formula
  reads about +1 dB high once the noise tail is clipped.
* Plates are ``(96, 336)`` synthetic and ``(92, 256)`` CCPD -- the 320x80
  drawing area plus the 8 px margin -- not ``320x80``.
* ``segment_characters`` drops the largest component only when it spans or
  touches the plate edge (the border); a large glyph is kept.
"""

__all__ = [
    "renderer",
    "ccpd_loader",
    "dataset_writer",
    "psf_builder",
    "degradation",
    "binarizer",
]
