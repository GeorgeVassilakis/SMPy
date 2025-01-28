"""Filter functions for aperture mass mapping and related processing."""

from scipy.ndimage import gaussian_filter, convolve
import numpy as np

def gaussian_aperture_filter(data, filter_config):
    """Apply Gaussian aperture filter to input data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input shear data to be filtered
    filter_config : dict
        Filter configuration containing:
        - type: filter type ('gaussian')
        - scale: characteristic scale of the filter
        - truncation: number of scale lengths at which to truncate (optional)
        
    Returns
    -------
    numpy.ndarray
        Filtered data
    """
    if not filter_config or filter_config.get('type', '').lower() != 'gaussian':
        raise ValueError("Invalid or missing filter configuration")
    
    scale = filter_config.get('scale', 1.0)
    truncation = filter_config.get('truncation', 5.0)
    
    # Create filter kernel
    size = int(np.ceil(2 * truncation * scale))
    size = size + 1 if size % 2 == 0 else size  # Ensure odd size
    
    x = np.linspace(-truncation * scale, truncation * scale, size)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    
    # Compute Q(r) filter that relates to U(r) = exp(-r²/2σ²)
    # Q(r) = (1/r²)[exp(-r²/2σ²) - (1 - r²/2σ²)exp(-r²/2σ²)]
    sigma2 = scale**2
    R2 = R**2
    exponent = np.exp(-R2/(2*sigma2))
    
    # Handle r=0 case to avoid division by zero
    Q = np.zeros_like(R)
    nonzero = R > 0
    Q[nonzero] = (1/R2[nonzero]) * (1 - (1 - R2[nonzero]/(2*sigma2))) * exponent[nonzero]
    Q[~nonzero] = 1/(2*sigma2)  # Limit as r→0
    
    # Truncate and normalize
    Q[R > truncation * scale] = 0
    Q = Q / np.sum(np.abs(Q))  # Ensure proper normalization
    
    # Apply filter
    return convolve(data, Q, mode='constant', cval=0.0)

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
    
    if filter_type == 'gaussian':
        return gaussian_aperture_filter(data, filter_config)
    else:
        raise ValueError(f"Unknown aperture filter type: {filter_type}")