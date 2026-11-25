"""Implementation of aperture mass mapping using filter functions.

This module implements the aperture mass mapping technique using various
optimal filter functions (Schirmer or Schneider filters) for localized
mass reconstruction from weak lensing shear measurements.
"""

from functools import partial
from math import ceil

import jax
import jax.numpy as jnp

from smpy._precision import numpy_precision
from smpy.filters.processing import schirmer_filter, schneider_filter
from smpy.mapping_methods.base import MassMapper

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

    # Any wraparound in this padding falls before the same-mode crop.
    fft_shape = (_fft_length(ny + half_y), _fft_length(nx + half_x))
    occupied = (g1 != 0) | (g2 != 0)
    footprint = (kernel_cos != 0) | (kernel_sin != 0)
    shear1_fft, shear2_fft, occupied_fft = jnp.fft.rfft2(jnp.stack((g1, g2, occupied)), s=fft_shape)
    kernels = jnp.stack((kernel_cos, kernel_sin, footprint))
    cos_fft, sin_fft, footprint_fft = jnp.fft.rfft2(kernels, s=fft_shape)
    maps_fft = jnp.stack((
        -shear1_fft * cos_fft - shear2_fft * sin_fft,
        shear1_fft * sin_fft - shear2_fft * cos_fft,
        occupied_fft * footprint_fft,
    ))
    maps_and_overlap = jnp.fft.irfft2(maps_fft, s=fft_shape)[:, half_y:half_y + ny, half_x:half_x + nx]
    maps = maps_and_overlap[:2]

    # Restore exact zeros when the filter overlaps no input shear. The
    # linear convolution still has a derivative at those shear values.
    overlap = maps_and_overlap[2]
    maps += jax.lax.stop_gradient(jnp.where(overlap > 0.5, 0.0, -maps))
    return maps[0], maps[1]

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping using configurable filters.

    This class implements the aperture mass mapping technique using various
    optimal filter functions. The method applies convolution with compact
    support filters to create localized mass reconstructions from shear data.

    Parameters
    ----------
    filter_type : `str`
        Aperture filter: 'schirmer' or 'schneider'.
    filter_scale : `float`
        Aperture radius in map pixels.
    truncation : `float`
        Kernel radius in units of ``filter_scale``.
    ell : `int`
        Polynomial order of the Schneider filter; unused for Schirmer.

    Raises
    ------
    ValueError
        Raised if ``filter_type`` is unknown or a scale is not positive.

    Notes
    -----
    The aperture mass method uses filter functions Q(r) to compute:
    Map_E = conv(-g1, Q*cos(2φ)) + conv(-g2, Q*sin(2φ))
    Map_B = conv(g1, Q*sin(2φ)) + conv(-g2, Q*cos(2φ))
    where φ is the polar angle and Q(r) is the chosen filter function.
    This is the aperture-mass integral of Schneider (1996) over the binned
    shear, with map pixels as the unit of area. Q is zero beyond
    ``truncation * filter_scale`` and at the center pixel, which has no
    tangential direction.
    """

    def __init__(self, filter_type, filter_scale, truncation, ell):
        if filter_type not in ("schirmer", "schneider"):
            raise ValueError(f"filter_type must be 'schirmer' or 'schneider', got {filter_type!r}")
        if filter_scale <= 0 or truncation <= 0:
            raise ValueError("filter_scale and truncation must be positive")
        self.filter_type = filter_type
        self.filter_scale = filter_scale
        self.truncation = truncation
        self.ell = ell

    @numpy_precision
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
        map_e : `numpy.ndarray` or `jax.Array`
            E-mode aperture mass map.
        map_b : `numpy.ndarray` or `jax.Array`
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
