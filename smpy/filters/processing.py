"""Filter functions for aperture mass mapping.

This module provides implementations of various filter functions used in
aperture mass mapping including Schirmer and Schneider filters.
"""

import jax.numpy as jnp

from smpy._precision import numpy_precision

@numpy_precision
def schirmer_filter(scaled_radii, scale):
    """Compute Schirmer filter values for given scaled radii.

    Calculate the Schirmer filter function values used in aperture mass
    mapping for optimal mass reconstruction with suppressed noise.

    Parameters
    ----------
    scaled_radii : `array_like`
        Radius values scaled by the aperture radius.
    scale : `float` or `jax.Array`
        Scale radius of the filter in pixels.

    Returns
    -------
    filter_values : `numpy.ndarray` or `jax.Array`
        Filter values at the given radii positions.

    Notes
    -----
    Implements the Q function filter as defined in McCleary et al. 2020 [1]_:
    Q(x) = [1/(1 + e^(a-bx) + e^(dx-c))] * [tanh(x/x_c)/(pi*Rs^2*(x/x_c))]

    Parameters are set to a = 6, b = 150, c = 47, d = 50 (Hetterscheidt et
    al. 2005 [2]_) and x_c = 0.12 (McCleary et al. 2020 [1]_); Hetterscheidt
    et al. use x_c = 0.15. The ratio tanh(x/x_c)/(x/x_c) takes its limiting
    value of one at the origin.

    References
    ----------
    .. [1] McCleary et al. 2020
    .. [2] Hetterscheidt et al. 2005
    """
    scaled_radii = jnp.asarray(scaled_radii)
    a = 6
    b = 150
    c = 47
    d = 50
    x_c = 0.12

    log_taper_denominator = jnp.logaddexp(0.0, jnp.logaddexp(a - b * scaled_radii, d * scaled_radii - c))
    x_ratio = scaled_radii / x_c
    safe_ratio = jnp.where(x_ratio == 0, 1.0, x_ratio)
    profile = jnp.where(x_ratio == 0, 1.0, jnp.tanh(safe_ratio) / safe_ratio)
    return jnp.exp(-log_taper_denominator) * profile / (jnp.pi * scale**2)

@numpy_precision
def schneider_filter(scaled_radii, scale, ell=3):
    """Compute Schneider filter values for given scaled radii.

    Calculate the Schneider filter function values with polynomial weighting
    for aperture mass mapping applications.

    Parameters
    ----------
    scaled_radii : `array_like`
        Radius values scaled by the aperture radius.
    scale : `float` or `jax.Array`
        Scale radius of the filter in pixels.
    ell : `int`, optional
        Polynomial order parameter.

    Returns
    -------
    filter_values : `numpy.ndarray` or `jax.Array`
        Q filter values at the given radii positions.

    Notes
    -----
    Implements the Schneider et al. (1998) Q function [1]_:
    Q(ϑ) = [(1+ℓ)(2+ℓ)/(πθ²)] * (ϑ²/θ²) * (1-ϑ²/θ²)^ℓ * H(θ-ϑ)
    where H is the Heaviside function ensuring compact support.

    References
    ----------
    .. [1] Schneider et al. 1998
    """
    scaled_radii = jnp.asarray(scaled_radii)
    mask = scaled_radii <= 1.0
    radius_squared = jnp.where(mask, scaled_radii, 0.0)**2
    profile = (1 + ell) * (2 + ell) / (jnp.pi * scale**2) * radius_squared * (1 - radius_squared)**ell
    return jnp.where(mask, profile, 0.0)
