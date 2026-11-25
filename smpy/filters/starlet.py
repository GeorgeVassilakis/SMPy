"""Starlet Transform implementation.

This module implements the isotropic undecimated wavelet transform (starlet
transform), which is used in the KS+ mass-inversion method to apply power
spectrum constraints when reconstructing convergence maps from shear data.

The implementation is based on the algorithm described in J.-L. Starck,
J. Fadili, and F. Murtagh, "The Undecimated Wavelet Decomposition and its
Reconstruction," IEEE Transactions on Image Processing, vol. 16, no. 2,
pp. 297-309, 2007.

This code draws from the CosmoStat implementation:
    https://github.com/CosmoStat/cosmostat/blob/master/pycs/sparsity/sparse2d/starlet.py
    (MIT Licensed)
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from smpy._precision import numpy_precision

B3_SPLINE = np.array([1, 4, 6, 4, 1]) / 16
"""Taps of the B3-spline filter of the starlet transform (`numpy.ndarray`)."""

def _filter_axis(data, step):
    """Apply the dilated B3-spline to the first axis."""
    size = data.shape[0]
    padded = jnp.pad(data, ((2 * step, 2 * step), (0, 0)), mode="reflect")
    return sum(float(tap) * padded[i * step:i * step + size] for i, tap in enumerate(B3_SPLINE))

@numpy_precision
@partial(jax.jit, static_argnames=("step",))
def apply_filter(data, step):
    """Smooth an image with B3-spline taps spaced ``step`` pixels apart.

    Parameters
    ----------
    data : `jax.Array` or `numpy.ndarray`
        Input 2D image.
    step : `int`
        Static spacing of the taps, 2^j at à trous scale j.

    Returns
    -------
    smoothed : `numpy.ndarray` or `jax.Array`
        Smoothed 2D image.

    Notes
    -----
    The filter is applied along columns, then rows, with the image mirrored
    at its edges without repeating the edge pixel (``mode='reflect'`` in
    `numpy.pad`).
    """
    return _filter_axis(_filter_axis(data, step).T, step).T

@numpy_precision
def starlet_transform_2d(data, nscales):
    """Compute the isotropic undecimated wavelet transform of an image.

    Perform the starlet transform decomposition using the à trous algorithm
    to generate wavelet coefficients at multiple scales.

    Parameters
    ----------
    data : `jax.Array` or `numpy.ndarray`
        Input 2D image.
    nscales : `int`
        Number of wavelet scales to compute.

    Returns
    -------
    wavelet_bands : `numpy.ndarray` or `jax.Array`
        3D array containing wavelet coefficients with shape (nscales, ny, nx)
        where the first nscales-1 bands are wavelet coefficients and the
        last band is the coarse approximation.

    Notes
    -----
    The starlet transform is an isotropic undecimated wavelet transform
    that preserves translation invariance. Each wavelet band represents
    the detail information between consecutive smoothing scales.
    """
    if isinstance(data, np.ndarray):
        data = np.asarray(data, dtype=np.float64)
    return _starlet_transform_2d(data, nscales)

@partial(jax.jit, static_argnames=("nscales",))
def _starlet_transform_2d(data, nscales):
    """Compute the detail bands and coarse image on the device."""
    data = jnp.asarray(data, dtype=jnp.result_type(data, 1.0))
    details = []
    for scale in range(nscales - 1):
        smoothed = apply_filter(data, 1 << scale)
        details.append(data - smoothed)
        data = smoothed
    return jnp.stack((*details, data))

@numpy_precision
@jax.jit
def inverse_starlet_transform_2d(wavelet_bands):
    """Reconstruct an image from its starlet transform coefficients.

    Parameters
    ----------
    wavelet_bands : `jax.Array` or `numpy.ndarray`
        Wavelet coefficients from starlet_transform_2d with shape
        (nscales, ny, nx).

    Returns
    -------
    reconstructed : `numpy.ndarray` or `jax.Array`
        Reconstructed 2D image.

    Notes
    -----
    Each detail band is the difference between two successive smoothings of
    the image, so the detail bands and the coarse residual sum to the image.
    """
    return jnp.sum(wavelet_bands, axis=0)


def compute_starlet_nscales_max(height: int, width: int) -> int:
    """Compute the safe maximum ``nscales`` for a starlet transform.

    The starlet (isotropic undecimated, à trous) transform produces ``J``
    detail bands and one coarse residual. The number of scales is defined as
    ``nscales = J + 1``. The B3–spline à trous kernel support at detail level
    ``j`` (0-indexed) is ``L_j = 4 * 2^j + 1`` pixels. To avoid border-
    dominated coefficients, the coarsest detail (``j = J - 1``) must satisfy
    ``L_{J-1} <= N`` where ``N = min(height, width)``.

    Parameters
    ----------
    height : `int`
        Image height in pixels.
    width : `int`
        Image width in pixels.

    Returns
    -------
    nscales_max : `int`
        Maximum safe number of starlet scales (detail bands + coarse).
    """
    max_step = max(int(min(height, width)) - 1, 0) // 4
    return 1 + max(1, max_step.bit_length())


def starlet_nscales_support_aware(
    height: int, width: int, cfg_nscales: Optional[int] = None
) -> int:
    """Return a safe ``nscales`` for the starlet transform.

    This function enforces the kernel-support constraint for the B3–spline
    starlet and applies an optional user override clipped to the safe range.

    Parameters
    ----------
    height : `int`
        Image height in pixels.
    width : `int`
        Image width in pixels.
    cfg_nscales : `int`, optional
        User-requested number of scales. If provided, it is clipped to the
        inclusive range ``[2, nscales_max]``.

    Returns
    -------
    nscales : `int`
        Number of scales to use (detail bands + coarse residual).
    """
    nscales_max = compute_starlet_nscales_max(height, width)

    if cfg_nscales is None:
        return nscales_max

    return max(2, min(int(cfg_nscales), nscales_max))
