"""JAX Gaussian, starlet and aperture filters."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from smpy.filters.starlet import B3_SPLINE

@partial(jax.jit, static_argnames=("radius",))
def gaussian_filter(image, sigma, *, radius):
    """Smooth a two-dimensional map with a separable Gaussian.

    Parameters
    ----------
    image : `jax.Array`
        Real two-dimensional map to smooth.
    sigma : `float` or `jax.Array`
        Scalar Gaussian standard deviation in pixels. A scalar tracer is
        supported. A value of zero leaves the input unchanged.
    radius : `int`
        Static maximum kernel radius in pixels, at least zero.

    Returns
    -------
    smoothed : `jax.Array`
        Smoothed map with the input shape.

    Notes
    -----
    The normalized kernel uses the nearest integer to four times sigma,
    rounding ties up and capping at ``radius``. The filter repeats the
    edge pixel on reflection, matching SciPy's ``mode='reflect'``. Its
    derivative with respect to sigma is computed within each fixed active
    radius.
    """
    if radius == 0:
        return image

    sigma = jnp.asarray(sigma, dtype=image.dtype)
    active_radius = jnp.floor(4 * sigma + 0.5)
    should_smooth = active_radius >= 1
    # A batched conditional can evaluate both branches, including at zero.
    safe_sigma = jnp.where(should_smooth, sigma, 1.0)

    def smooth(data):
        offsets = jnp.arange(-radius, radius + 1, dtype=data.dtype)
        weights = jnp.where(jnp.abs(offsets) <= active_radius, jnp.exp(-0.5 * (offsets / safe_sigma)**2), 0)
        kernel = weights / weights.sum()

        data = data[None, None, :, :]
        data = jnp.pad(data, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="symmetric")
        data = lax.conv_general_dilated(data, kernel[None, None, :, None], (1, 1), "VALID")
        data = jnp.pad(data, ((0, 0), (0, 0), (0, 0), (radius, radius)), mode="symmetric")
        data = lax.conv_general_dilated(data, kernel[None, None, None, :], (1, 1), "VALID")
        return data[0, 0]

    return lax.cond(should_smooth, smooth, lambda data: data, image)

def _filter_axis(data, step):
    """Apply the dilated B3-spline to the first axis."""
    size = data.shape[0]
    padded = jnp.pad(data, ((2 * step, 2 * step), (0, 0)), mode="reflect")
    return sum(float(tap) * padded[i * step:i * step + size] for i, tap in enumerate(B3_SPLINE))

@partial(jax.jit, static_argnames=("step",))
def apply_filter(data, step):
    """Smooth an image with B3-spline taps spaced ``step`` pixels apart.

    Parameters
    ----------
    data : `jax.Array`
        Input 2D image.
    step : `int`
        Static spacing of the taps, 2^j at à trous scale j.

    Returns
    -------
    smoothed : `jax.Array`
        Smoothed 2D image.

    Notes
    -----
    The filter is applied along columns, then rows, with the image mirrored
    at its edges without repeating the edge pixel (``mode='reflect'`` in
    `numpy.pad`).
    """
    return _filter_axis(_filter_axis(data, step).T, step).T

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
    filter_values : `jax.Array`
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
    filter_values : `jax.Array`
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
