"""Filter functions for aperture mass mapping.

This module provides implementations of various filter functions used in
aperture mass mapping including Schirmer and Schneider filters.
"""

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
