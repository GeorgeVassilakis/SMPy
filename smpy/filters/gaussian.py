"""Gaussian smoothing for JAX mass maps."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

@partial(jax.jit, static_argnames=("radius",))
def gaussian_filter(image, sigma, *, radius):
    """Smooth a two-dimensional map with a separable Gaussian.

    Parameters
    ----------
    image : `numpy.ndarray` or `jax.Array`
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
    image = jnp.asarray(image)
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
