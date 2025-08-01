"""Filter functions for aperture mass mapping and related processing.

This module provides implementations of various filter functions used in
aperture mass mapping including Schirmer and Schneider filters, along with
utility functions for creating 2D convolution kernels.
"""

from scipy.ndimage import convolve
import numpy as np

def schirmer_filter(scaled_radii, scale):
    """Compute Schirmer filter values for given scaled radii.

    Calculate the Schirmer filter function values used in aperture mass
    mapping for optimal mass reconstruction with suppressed noise.

    Parameters
    ----------
    scaled_radii : `numpy.ndarray`
        Radius values scaled by the aperture radius.
    scale : `float`
        Scale radius of the filter in pixels.

    Returns
    -------
    filter_values : `numpy.ndarray`
        Filter values at the given radii positions.

    Notes
    -----
    Implements the Q function filter as defined in McCleary et al. 2020:
    Q(x) = [1/(1 + e^(a-bx) + e^(dx-c))] * [tanh(x/x_c)/(pi*Rs^2*(x/x_c))]

    Parameters are set to:
    a = 6, b = 150, c = 47, d = 50, x_c = 0.12 (Hetterscheidt et al. 2005)

    References
    ----------
    .. [1] McCleary et al. 2020
    .. [2] Hetterscheidt et al. 2005
    """
    # Filter parameters
    a = 6
    b = 150
    c = 47
    d = 50
    x_c = 0.12
    
    # Calculate the first part: 1/(1 + e^(a-bx) + e^(dx-c))
    part1 = 1.0 / (1.0 + np.exp(a - b*scaled_radii) + np.exp(d*scaled_radii - c))
    
    # Calculate the second part: tanh(x/x_c)/(pi*Rs^2*(x/x_c))
    # Avoid division by zero when x = 0
    x_ratio = np.divide(scaled_radii, x_c, out=np.ones_like(scaled_radii)*1e-10, where=scaled_radii!=0)
    part2 = np.tanh(x_ratio) / (np.pi * scale**2 * x_ratio)
    
    # Combine parts
    return part1 * part2

def schneider_filter(scaled_radii, scale, l=3):
    """Compute Schneider filter values for given scaled radii.

    Calculate the Schneider filter function values with polynomial weighting
    for aperture mass mapping applications.

    Parameters
    ----------
    scaled_radii : `numpy.ndarray`
        Radius values scaled by the aperture radius.
    scale : `float`
        Scale radius of the filter in pixels.
    l : `int`, optional
        Polynomial order parameter.

    Returns
    -------
    filter_values : `numpy.ndarray`
        Q filter values at the given radii positions.

    Notes
    -----
    Implements the Schneider et al. (1998) Q function:
    Q(ϑ) = [(1+ℓ)(2+ℓ)/(πθ²)] * (ϑ²/θ²) * (1-ϑ²/θ²)^ℓ * H(θ-ϑ)
    where H is the Heaviside function ensuring compact support.

    References
    ----------
    .. [1] Schneider et al. 1998
    """
    # Initialize filter values
    filter_vals = np.zeros_like(scaled_radii)
    
    # Apply filter only within the aperture radius (Heaviside function)
    mask = scaled_radii <= 1.0
    
    # Q function filter
    filter_vals[mask] = ((1 + l) * (2 + l) / (np.pi * scale**2)) * \
                        (scaled_radii[mask]**2) * \
                        (1 - scaled_radii[mask]**2)**l
    
    return filter_vals

def create_filter_kernel(filter_func, scale, truncation=1.0, l=None):
    """Create a 2D kernel for any aperture mass filter.

    Generate a 2D convolution kernel from a radial filter function for
    use in aperture mass mapping operations.

    Parameters
    ----------
    filter_func : callable
        Filter function to apply to scaled radii.
    scale : `float`
        Scale radius of the filter in pixels.
    truncation : `float`, optional
        Truncation radius in units of scale radius.
    l : `int`, optional
        Polynomial order parameter for Schneider filter.

    Returns
    -------
    kernel : `numpy.ndarray`
        2D filter kernel for convolution operations.

    Notes
    -----
    The kernel size is determined by the truncation radius and is always
    made odd to ensure a well-defined center pixel.
    """
    # Create grid for kernel
    size = int(np.ceil(2 * truncation * scale))
    size = size + 1 if size % 2 == 0 else size  # Ensure odd size
    
    x = np.linspace(-truncation * scale, truncation * scale, size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Calculate scaled radii and apply filter
    scaled_radius = R / scale
    
    # Call the filter function with appropriate arguments
    if filter_func == schneider_filter and l is not None:
        kernel = filter_func(scaled_radius, scale, l)
    else:
        kernel = filter_func(scaled_radius, scale)
    
    return kernel

def apply_filter_convolution(data, kernel):
    """Apply a filter kernel via convolution.

    Perform 2D convolution of data with the specified kernel using
    constant boundary conditions.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D input data array.
    kernel : `numpy.ndarray`
        2D convolution kernel.

    Returns
    -------
    filtered_data : `numpy.ndarray`
        Convolved data array.
    """
    return convolve(data, kernel, mode='constant', cval=0.0)

def apply_aperture_filter(data, filter_config):
    """Apply aperture filter based on configuration.

    Apply the specified aperture mass filter to input data according
    to the provided configuration parameters.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D input data array.
    filter_config : `dict` or None
        Filter configuration dictionary containing 'type', 'scale',
        'truncation', and optional 'l' parameters.

    Returns
    -------
    filtered_data : `numpy.ndarray`
        Filtered data array.

    Raises
    ------
    ValueError
        If unknown aperture filter type is specified.

    Notes
    -----
    Supported filter types are 'schirmer' and 'schneider'. If filter_config
    is None, returns the input data unchanged.
    """
    if filter_config is None:
        return data
        
    filter_type = filter_config.get('type', '').lower()
    scale = filter_config.get('scale', 1.0)
    truncation = filter_config.get('truncation', 1.0)
    
    if filter_type == 'schirmer':
        kernel = create_filter_kernel(schirmer_filter, scale, truncation)
    elif filter_type == 'schneider':
        l = filter_config.get('l', 3)
        kernel = create_filter_kernel(schneider_filter, scale, truncation, l=l)
    else:
        raise ValueError(f"Unknown aperture filter type: {filter_type}")
        
    return apply_filter_convolution(data, kernel)