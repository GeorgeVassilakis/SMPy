"""Implementation of aperture mass mapping using filter functions.

This module implements the aperture mass mapping technique using various
optimal filter functions (Schirmer or Schneider filters) for localized
mass reconstruction from weak lensing shear measurements.
"""

from functools import partial
from math import ceil
from sys import float_info

import jax
import jax.numpy as jnp

from smpy.mapping_methods import ApertureMassMapper as NumpyApertureMassMapper

from .filters import schirmer_filter, schneider_filter

def _fft_length(minimum):
    """Return the smallest FFT length with prime factors 2, 3 and 5."""
    length = minimum
    while True:
        remainder = length
        for factor in (2, 3, 5):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return length
        length += 1

@partial(jax.jit, static_argnames=("filter_type", "ell", "max_radius"))
def _aperture_mass_maps(g1, g2, filter_scale, truncation, *, filter_type, ell, max_radius):
    """Convolve binned shear with a compact spin-2 filter."""
    ny, nx = g1.shape
    dtype = jnp.result_type(g1, g2, 1.0)
    if ny == 0 or nx == 0:
        return jnp.zeros_like(g1, dtype=dtype), jnp.zeros_like(g2, dtype=dtype)

    # Offsets longer than a map axis cannot contribute to its same-mode crop.
    half_y = min(max_radius, ny - 1)
    half_x = min(max_radius, nx - 1)
    y = jnp.arange(-half_y, half_y + 1, dtype=dtype)[:, None]
    x = jnp.arange(-half_x, half_x + 1, dtype=dtype)[None, :]
    x_squared = x**2
    y_squared = y**2
    radius_squared = x_squared + y_squared
    radius = jnp.sqrt(radius_squared)
    scale = jnp.asarray(filter_scale, dtype=dtype)
    limit = jnp.asarray(truncation, dtype=dtype) * scale
    inside = (radius <= limit) & (radius_squared > 0)
    scaled_radius = jnp.where(inside, radius, 0.0) / scale
    if filter_type == "schneider":
        profile = schneider_filter(scaled_radius, scale, ell)
    else:
        profile = schirmer_filter(scaled_radius, scale)
    spin_weight = jnp.where(inside, profile, 0.0) / jnp.where(radius_squared > 0, radius_squared, 1.0)
    kernel_cos = spin_weight * (x_squared - y_squared)
    kernel_sin = spin_weight * (2 * x * y)
    # `scipy.ndimage.convolve`, which the NumPy backend uses, skips kernel
    # weights this small whatever the precision of the map.
    kernel_cos = jnp.where(jnp.abs(kernel_cos) > float_info.epsilon, kernel_cos, 0.0)
    kernel_sin = jnp.where(jnp.abs(kernel_sin) > float_info.epsilon, kernel_sin, 0.0)

    # Any wraparound in this padding falls before the same-mode crop.
    fft_shape = (_fft_length(ny + half_y), _fft_length(nx + half_x))
    fields = jnp.stack((g1, g2, g1 != 0, g2 != 0))
    shear1_fft, shear2_fft, some1_fft, some2_fft = jnp.fft.rfft2(fields, s=fft_shape)
    kernels = jnp.stack((kernel_cos, kernel_sin, kernel_cos != 0, kernel_sin != 0))
    cos_fft, sin_fft, some_cos_fft, some_sin_fft = jnp.fft.rfft2(kernels, s=fft_shape)
    maps_fft = jnp.stack((
        -shear1_fft * cos_fft - shear2_fft * sin_fft,
        shear1_fft * sin_fft - shear2_fft * cos_fft,
        some1_fft * some_cos_fft + some2_fft * some_sin_fft,
        some1_fft * some_sin_fft + some2_fft * some_cos_fft,
    ))
    maps_and_terms = jnp.fft.irfft2(maps_fft, s=fft_shape)[:, half_y:half_y + ny, half_x:half_x + nx]
    maps, terms = maps_and_terms[:2], maps_and_terms[2:]

    # A mode is exactly zero where none of its products of a shear
    # component with a kernel is nonzero, as in a direct sum. The linear
    # convolution still has a derivative at those shear values.
    maps += jax.lax.stop_gradient(jnp.where(terms > 0.5, 0.0, -maps))
    return maps[0], maps[1]

class ApertureMassMapper(NumpyApertureMassMapper):
    """Aperture mass mapping with JAX.

    Takes the settings of `smpy.mapping_methods.ApertureMassMapper`.
    """

    def create_maps(self, grid, *, filter_scale=None, truncation=None):
        """Create aperture mass maps by convolving with the selected filter.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear with empty pixels set to zero.
        filter_scale : `float` or `jax.Array`, optional
            Positive aperture radius in map pixels. Defaults to the
            configured ``filter_scale``.
        truncation : `float` or `jax.Array`, optional
            Positive kernel radius in units of ``filter_scale``. Defaults
            to the configured ``truncation``.

        Returns
        -------
        map_e : `jax.Array`
            E-mode aperture mass map.
        map_b : `jax.Array`
            B-mode aperture mass map.

        Notes
        -----
        The convolution uses zero shear beyond the grid edges. The product
        of the supplied scale and truncation must be at most the configured
        product, which determines the static kernel size. Scale derivatives
        hold the discrete aperture footprint fixed. The truncation
        derivative is zero between changes in that footprint.
        """
        scale = self.filter_scale if filter_scale is None else filter_scale
        cutoff = self.truncation if truncation is None else truncation
        return _aperture_mass_maps(
            grid.g1,
            grid.g2,
            scale,
            cutoff,
            filter_type=self.filter_type,
            ell=self.ell,
            max_radius=ceil(self.filter_scale * self.truncation),
        )
