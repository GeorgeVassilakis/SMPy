"""Filter functions for aperture mass mapping and related processing."""

from scipy.ndimage import convolve
import numpy as np

def create_schirmer_kernel(scale, truncation=1.0):
    """Create a 2D kernel for the Schirmer aperture mass filter.
    
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
    # Create grid for kernel
    size = int(np.ceil(2 * truncation * scale))
    size = size + 1 if size % 2 == 0 else size  # Ensure odd size
    
    x = np.linspace(-truncation * scale, truncation * scale, size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Calculate scaled radii
    x = R / scale
    
    # Filter parameters
    a = 6
    b = 150
    c = 47
    d = 50
    x_c = 0.12
    
    # Calculate the first part: 1/(1 + e^(a-bx) + e^(dx-c))
    part1 = 1.0 / (1.0 + np.exp(a - b*x) + np.exp(d*x - c))
    
    # Calculate the second part: tanh(x/x_c)/(pi*Rs^2*(x/x_c))
    # Avoid division by zero when x = 0
    x_ratio = np.divide(x, x_c, out=np.ones_like(x)*1e-10, where=x!=0)
    part2 = np.tanh(x_ratio) / (np.pi * scale**2 * x_ratio)
    
    # Combine parts
    kernel = part1 * part2
    
    # Normalize
    kernel = kernel / np.sum(np.abs(kernel))
    
    return kernel

def apply_filter_convolution(data, kernel):
    """Apply a filter kernel via convolution.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data to be filtered
    kernel : numpy.ndarray
        Filter kernel to apply
        
    Returns
    -------
    numpy.ndarray
        Filtered data
    """
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
    
    if filter_type == 'schirmer':
        scale = filter_config.get('scale', 1.0)
        truncation = filter_config.get('truncation', 1.0)
        kernel = create_schirmer_kernel(scale, truncation)
        return apply_filter_convolution(data, kernel)
    else:
        raise ValueError(f"Unknown aperture filter type: {filter_type}")