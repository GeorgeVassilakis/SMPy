from scipy.ndimage import gaussian_filter

def apply_filter(data, filter_config):
    """Apply filtering to input data.
    
    Parameters
    ----------
    data : `numpy.ndarray`
        Input data array to be filtered
    filter_config : `dict`
        Filter configuration containing:
        - type: type of filter ('gaussian' or None)
        - sigma: smoothing scale (for gaussian)
        
    Returns
    -------
    numpy.ndarray
        Filtered data array
        
    Raises
    ------
    ValueError
        If unknown filter type specified
    """
    if filter_config is None or filter_config.get('type') is None:
        return data
        
    filter_type = filter_config['type'].lower()
    
    if filter_type == 'gaussian':
        sigma = filter_config.get('sigma', 1.0)
        return gaussian_filter(data, sigma=sigma)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")