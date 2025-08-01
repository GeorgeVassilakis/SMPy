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