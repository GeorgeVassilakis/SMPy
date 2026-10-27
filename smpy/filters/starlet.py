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

from typing import Optional

import numpy as np

B3_SPLINE = np.array([1, 4, 6, 4, 1]) / 16
"""Taps of the B3-spline filter of the starlet transform (`numpy.ndarray`)."""

def apply_filter(data, step):
    """Smooth an image with the B3-spline filter, its taps ``step`` pixels
    apart.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input 2D image.
    step : `int`
        Spacing of the filter taps; 2^j at scale j of the à trous algorithm.

    Returns
    -------
    smoothed : `numpy.ndarray`
        Smoothed 2D image.

    Notes
    -----
    The filter is applied along columns, then rows, with the image mirrored
    at its edges without repeating the edge pixel (``mode='reflect'`` in
    `numpy.pad`). Only the five taps are applied, so the cost does not grow
    with ``step``.
    """
    # Filter down the columns, then transpose to filter along the rows
    for _ in range(2):
        n = data.shape[0]
        padded = np.pad(data, ((2 * step, 2 * step), (0, 0)), mode="reflect")
        data = sum(tap * padded[k * step:k * step + n] for k, tap in enumerate(B3_SPLINE)).T
    return data

def starlet_transform_2d(data, nscales):
    """Compute the isotropic undecimated wavelet transform of an image.

    Perform the starlet transform decomposition using the à trous algorithm
    to generate wavelet coefficients at multiple scales.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input 2D image.
    nscales : `int`
        Number of wavelet scales to compute.

    Returns
    -------
    wavelet_bands : `numpy.ndarray`
        3D array containing wavelet coefficients with shape (nscales, ny, nx)
        where the first nscales-1 bands are wavelet coefficients and the
        last band is the coarse approximation.

    Notes
    -----
    The starlet transform is an isotropic undecimated wavelet transform
    that preserves translation invariance. Each wavelet band represents
    the detail information between consecutive smoothing scales.
    """
    # Make sure input is float
    data = np.float64(data)

    # Initialize arrays
    ny, nx = data.shape
    wavelet_bands = np.zeros((nscales, ny, nx))

    # Initialize smoothed data
    smoothed_data = data.copy()

    # Compute wavelet bands
    for j in range(nscales-1):
        # Smooth with the filter taps 2^j pixels apart
        smoothed_j_plus_1 = apply_filter(smoothed_data, 2**j)

        # Wavelet coefficients = detail signal between two scales
        wavelet_bands[j] = smoothed_data - smoothed_j_plus_1

        # Update smoothed data for next scale
        smoothed_data = smoothed_j_plus_1

    # Last scale contains the coarse approximation
    wavelet_bands[nscales-1] = smoothed_data

    return wavelet_bands

def inverse_starlet_transform_2d(wavelet_bands):
    """Reconstruct an image from its starlet transform coefficients.

    Parameters
    ----------
    wavelet_bands : `numpy.ndarray`
        Wavelet coefficients from starlet_transform_2d with shape
        (nscales, ny, nx).

    Returns
    -------
    reconstructed : `numpy.ndarray`
        Reconstructed 2D image.

    Notes
    -----
    Each detail band is the difference between two successive smoothings of
    the image, so the detail bands and the coarse residual sum to the image.
    """
    return np.sum(wavelet_bands, axis=0)


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
    N = int(min(height, width))

    # Guard against very small images. Ensure at least one detail band.
    if N <= 1:
        return 2

    value = (N - 1) / 4.0
    # If value < 1, log2 is negative; clamp to at least 1 detail band
    J_max = max(1, int(np.floor(np.log2(value))) + 1)
    return J_max + 1


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
