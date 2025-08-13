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

import numpy as np
from scipy.ndimage import convolve1d
from typing import Optional

def b3spline_filter(step=1):
    """Create a B3-spline filter for the starlet transform.

    Generate the 1D B3-spline filter kernel used in the starlet wavelet
    transform with appropriate dilation for the à trous algorithm.

    Parameters
    ----------
    step : `int`, optional
        The dilation step for the à trous algorithm.

    Returns
    -------
    kernel : `numpy.ndarray`
        The 1D B3-spline filter with appropriate spacing.

    Notes
    -----
    The B3-spline filter coefficients are [1/16, 1/4, 3/8, 1/4, 1/16].
    For step > 1, zeros are inserted between coefficients according to
    the à trous algorithm.
    """
    # B3-spline filter coefficients
    h = np.array([1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16])
    
    # For step=1, return the basic filter
    if step == 1:
        return h
    
    # For larger steps, add zeros between coefficients (à trous algorithm)
    kernel = np.zeros(len(h) + (len(h)-1)*(step-1))
    kernel[::step] = h
    
    return kernel

def apply_filter(data, kernel):
    """Apply separable convolution with the given kernel.

    Perform 2D convolution by applying the 1D kernel separately along
    rows and columns using mirror boundary conditions.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input 2D image.
    kernel : `numpy.ndarray`
        1D convolution kernel.

    Returns
    -------
    smoothed : `numpy.ndarray`
        Smoothed 2D image.

    Notes
    -----
    Uses mirror boundary conditions to handle edges appropriately
    for wavelet transforms.
    """
    # Apply filter along rows, then columns
    temp = convolve1d(data, kernel, axis=0, mode='mirror')
    return convolve1d(temp, kernel, axis=1, mode='mirror')

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
        # Get filter for current scale
        step = 2**j
        kernel = b3spline_filter(step)
        
        # Smooth data using convolution with the filter
        smoothed_j_plus_1 = apply_filter(smoothed_data, kernel)
        
        # Wavelet coefficients = detail signal between two scales
        wavelet_bands[j] = smoothed_data - smoothed_j_plus_1
        
        # Update smoothed data for next scale
        smoothed_data = smoothed_j_plus_1
    
    # Last scale contains the coarse approximation
    wavelet_bands[nscales-1] = smoothed_data
    
    return wavelet_bands

def inverse_starlet_transform_2d(wavelet_bands):
    """Reconstruct an image from its starlet transform coefficients.

    Perform the inverse starlet transform using the second generation
    reconstruction algorithm to recover the original image.

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
    The reconstruction starts with the coarsest scale and progressively
    adds detail bands from coarse to fine scales, applying appropriate
    smoothing at each step to ensure perfect reconstruction.
    """
    nscales, _, _ = wavelet_bands.shape
    
    # Start with the coarsest scale
    reconstructed = np.copy(wavelet_bands[nscales-1])
    
    # Process each detail band from coarse to fine
    for j in range(nscales-2, -1, -1):
        # Calculate step size for the filter
        step = 2**(j)
        
        # Get filter for current scale
        kernel = b3spline_filter(step)
        
        # Apply filter to current reconstruction
        filtered = apply_filter(reconstructed, kernel)
        
        # Add the wavelet coefficients
        reconstructed = filtered + wavelet_bands[j]
    
    return reconstructed


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
