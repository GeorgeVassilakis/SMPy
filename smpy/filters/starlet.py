"""
Starlet Transform implementation

This module implements the isotropic undecimated wavelet transform (starlet transform), 
which is used in the KS+ mass-inversion method to apply power spectrum constraints 
when reconstructing convergence maps from shear data.

The implementation is based on the algorithm described in:
    J.-L. Starck, J. Fadili, and F. Murtagh, "The Undecimated Wavelet Decomposition 
    and its Reconstruction," IEEE Transactions on Image Processing, vol. 16, no. 2, 
    pp. 297-309, 2007.

This code draws from the CosmoStat implementation:
    https://github.com/CosmoStat/cosmostat/blob/master/pycs/sparsity/sparse2d/starlet.py
    (MIT Licensed)
"""

import numpy as np

def b3spline_filter(step=1):
    """
    Create a B3-spline filter for the starlet transform.
    
    Parameters
    ----------
    step : int
        The dilation step (à trous algorithm)
        
    Returns
    -------
    kernel : ndarray
        The 1D B3-spline filter with appropriate spacing
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

def starlet_transform_2d(data, nscales):
    """
    Compute the isotropic undecimated wavelet transform (starlet transform) of an image.
    
    Parameters
    ----------
    data : ndarray
        Input 2D image
    nscales : int
        Number of wavelet scales to compute
    
    Returns
    -------
    wavelet_bands : ndarray
        3D array containing wavelet coefficients. Shape is (nscales, ny, nx)
        where the first nscales-1 bands are wavelet coefficients and the last band is the 
        coarse approximation.
    """
    # Make sure input is float
    data = np.float64(data)
    
    # Initialize arrays
    nx, ny = data.shape
    wavelet_bands = np.zeros((nscales, nx, ny))
    
    # Initialize smoothed data
    smoothed_data = data.copy()
    
    # Compute wavelet bands
    for j in range(nscales-1):
        # Get filter for current scale
        step = 2**j
        kernel = b3spline_filter(step)
        
        # Smooth data using convolution with the filter
        smoothed_j_plus_1 = smooth_with_filter(smoothed_data, kernel)
        
        # Wavelet coefficients = detail signal between two scales
        wavelet_bands[j] = smoothed_data - smoothed_j_plus_1
        
        # Update smoothed data for next scale
        smoothed_data = smoothed_j_plus_1
    
    # Last scale contains the coarse approximation
    wavelet_bands[nscales-1] = smoothed_data
    
    return wavelet_bands

def smooth_with_filter(data, kernel):
    """
    Apply separable convolution with the given kernel.
    
    Parameters
    ----------
    data : ndarray
        Input 2D image
    kernel : ndarray
        1D convolution kernel
    
    Returns
    -------
    smoothed : ndarray
        Smoothed 2D image
    """
    # Pad for border handling (mirror padding)
    pad_width = len(kernel) // 2
    padded = np.pad(data, pad_width, mode='reflect')
    
    # Apply filter along rows
    temp = np.zeros_like(padded)
    for i in range(padded.shape[0]):
        temp[i] = np.convolve(padded[i], kernel, mode='same')
    
    # Apply filter along columns
    smoothed = np.zeros_like(padded)
    for j in range(padded.shape[1]):
        smoothed[:, j] = np.convolve(temp[:, j], kernel, mode='same')
    
    # Remove padding
    return smoothed[pad_width:-pad_width, pad_width:-pad_width]

def inverse_starlet_transform_2d(wavelet_bands):
    """
    Reconstruct an image from its starlet transform coefficients.
    
    Parameters
    ----------
    wavelet_bands : ndarray
        Wavelet coefficients from starlet_transform_2d
    
    Returns
    -------
    reconstructed : ndarray
        Reconstructed 2D image
    """
    # Simple reconstruction by adding all bands
    return np.sum(wavelet_bands, axis=0)

def get_wavelet_variance(wavelet_bands, mask=None):
    """
    Calculate variance at each wavelet scale.
    
    Parameters
    ----------
    wavelet_bands : ndarray
        Wavelet coefficients from starlet_transform_2d
    mask : ndarray, optional
        Binary mask (1 for valid data, 0 for masked data)
    
    Returns
    -------
    variances : ndarray
        Variance at each scale
    """
    nscales = wavelet_bands.shape[0]
    variances = np.zeros(nscales)
    
    for j in range(nscales):
        if mask is not None:
            # Calculate variance of unmasked regions
            values = wavelet_bands[j][mask > 0]
            if len(values) > 0:
                variances[j] = np.var(values)
        else:
            # Calculate variance of entire band
            variances[j] = np.var(wavelet_bands[j])
    
    return variances