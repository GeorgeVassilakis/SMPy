"""Filter functions for aperture mass mapping and related processing."""

from scipy.ndimage import convolve
import numpy as np

def s98_aperture_filter(x, scale):
    """Schneider+98 (S98) aperture mass isotropic filter function U(theta).
    
    Parameters
    ----------
    x : `numpy.ndarray`
        Radial distances from the center of the filter
    scale : `float`
        Scale radius in pixels
        
    Returns
    -------
    numpy.ndarray
        Filter values at input radii
        
    Notes
    -----
    Implements the S98 filter from Schneider et al. 1998, MNRAS 296, 873.
    The filter includes a normalization factor to match Giocoli et al. 2015.
    
    References
    ----------
    .. [1] Schneider et al. 1998, MNRAS 296, 873
    .. [2] Giocoli et al. 2015
    """
    x = np.atleast_1d(x).astype(float)
    y = x / scale
    
    # S98 filter parameters
    l = 1  # polynomial order
    prefactor = np.sqrt(276) / 24  # normalization
    A = (l + 2) / np.pi / scale**2
    
    # Compute filter
    result = A * np.power(1. - y**2, l) * (1. - (l + 2.) * y**2)
    result = prefactor * result * np.heaviside(scale - np.abs(x), 0.5)
    
    return result

def create_s98_kernel(scale, truncation=1.0):
    """Create a 2D kernel for the S98 aperture mass filter.
    
    Parameters
    ----------
    scale : float
        Scale radius of the filter in pixels
    truncation : float, optional
        Truncation radius in units of scale radius, by default 1.0
        
    Returns
    -------
    numpy.ndarray
        2D filter kernel
    """
    # Create grid for kernel
    size = int(np.ceil(2 * truncation * scale))
    size = size + 1 if size % 2 == 0 else size  # Ensure odd size
    
    x = np.linspace(-truncation * scale, truncation * scale, size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Compute filter values
    kernel = s98_aperture_filter(R, scale)
    
    # Normalize
    kernel = kernel / np.sum(np.abs(kernel))
    
    return kernel

def s98_aperture_filter_convolution(data, filter_config):
    """Apply S98 aperture filter to input data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input shear data to be filtered
    filter_config : dict
        Filter configuration containing:
        - type: filter type ('s98')
        - scale: characteristic scale of the filter
        - truncation: scale factor for truncation (optional)
        
    Returns
    -------
    numpy.ndarray
        Filtered data
    """
    if not filter_config or filter_config.get('type', '').lower() != 's98':
        raise ValueError("Invalid or missing filter configuration")
    
    scale = filter_config.get('scale', 1.0)
    truncation = filter_config.get('truncation', 1.0)
    
    kernel = create_s98_kernel(scale, truncation)
    return convolve(data, kernel, mode='constant', cval=0.0)

def apply_aperture_filter(data, filter_config):
    """Apply aperture filter based on configuration.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data array to be filtered
    filter_config : dict
        Filter configuration
        
    Returns
    -------
    numpy.ndarray
        Filtered data array
    """
    if filter_config is None:
        return data
        
    filter_type = filter_config.get('type', '').lower()
    
    if filter_type == 's98':
        return s98_aperture_filter_convolution(data, filter_config)
    else:
        raise ValueError(f"Unknown aperture filter type: {filter_type}")