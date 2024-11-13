from scipy.ndimage import gaussian_filter

def apply_filter(data, filter_config):
    """
    Apply the specified filter to the input data.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input data to be filtered
    filter_config : dict
        Configuration dictionary containing:
        - 'type': str, type of filter to apply
        - Additional parameters specific to each filter type
        
    Returns
    -------
    numpy.ndarray
        Filtered data
    """
    if filter_config is None or filter_config.get('type') is None:
        return data
        
    filter_type = filter_config['type'].lower()
    
    if filter_type == 'gaussian':
        sigma = filter_config.get('sigma', 1.0)
        return gaussian_filter(data, sigma=sigma)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")