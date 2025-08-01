"""Filtering utilities for plotting and visualization.

This module provides filtering functions used in mass map visualization
and post-processing, primarily focused on Gaussian smoothing operations.
"""

from scipy.ndimage import gaussian_filter

def apply_filter(data, filter_config):
    """Apply filtering to input data.

    Apply the specified filter to input data according to the provided
    configuration parameters. Currently supports Gaussian smoothing.

    Parameters
    ----------
    data : `numpy.ndarray`
        Input data array to be filtered.
    filter_config : `dict` or None
        Filter configuration dictionary containing:
        - type: type of filter ('gaussian' or None)
        - sigma: smoothing scale (for gaussian)

    Returns
    -------
    filtered_data : `numpy.ndarray`
        Filtered data array.

    Raises
    ------
    ValueError
        If unknown filter type is specified.

    Notes
    -----
    If filter_config is None or filter_config['type'] is None,
    returns the input data unchanged.
    """
    if filter_config is None or filter_config.get('type') is None:
        return data
        
    filter_type = filter_config['type'].lower()
    
    if filter_type == 'gaussian':
        sigma = filter_config.get('sigma', 1.0)
        return gaussian_filter(data, sigma=sigma)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")