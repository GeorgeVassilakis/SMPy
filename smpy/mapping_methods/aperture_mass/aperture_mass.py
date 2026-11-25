"""Implementation of aperture mass mapping using filter functions.

This module implements the aperture mass mapping technique using various
optimal filter functions (Schirmer or Schneider filters) for localized
mass reconstruction from weak lensing shear measurements.
"""

import numpy as np
from scipy.ndimage import convolve

from smpy.filters.processing import schirmer_filter, schneider_filter
from smpy.mapping_methods.base import MassMapper

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

    def _create_aperture_mass_kernels(self):
        """Create the aperture mass convolution kernels.

        Returns
        -------
        kernel_1 : `numpy.ndarray`
            Convolution kernel Q(r/rs) * cos(2*phi) for first component.
        kernel_2 : `numpy.ndarray`
            Convolution kernel Q(r/rs) * sin(2*phi) for second component.

        Notes
        -----
        The kernels implement the angular dependence needed for E/B mode
        decomposition in the aperture mass formalism.
        """
        rs = self.filter_scale

        # Pixel offsets out to the kernel radius
        radius = self.truncation * rs
        offsets = np.arange(-np.ceil(radius), np.ceil(radius) + 1)
        X_k, Y_k = np.meshgrid(offsets, offsets)

        # Calculate radial distance and polar angle
        R_k = np.sqrt(X_k**2 + Y_k**2)
        Phi_k = np.arctan2(Y_k, X_k)

        # Get scaled radii for filter calculation
        scaled_radii_k = R_k / rs

        # Get Q filter values based on filter type
        if self.filter_type == 'schneider':
            q_vals_k = schneider_filter(scaled_radii_k, rs, self.ell)
        else:
            q_vals_k = schirmer_filter(scaled_radii_k, rs)

        # Cut at the radius; the center has no tangential direction
        q_vals_k[(R_k > radius) | (R_k == 0)] = 0

        # Create the kernels
        kernel_1 = q_vals_k * np.cos(2 * Phi_k)
        kernel_2 = q_vals_k * np.sin(2 * Phi_k)

        return kernel_1, kernel_2

    def create_maps(self, grid):
        """Create aperture mass maps by convolving with the selected filter.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear.

        Returns
        -------
        map_e : `numpy.ndarray`
            E-mode aperture mass map.
        map_b : `numpy.ndarray`
            B-mode aperture mass map.

        Notes
        -----
        Implements the aperture mass convolution:
        Map_E = conv(-g1, K1) + conv(-g2, K2)
        Map_B = conv(g1, K2) + conv(-g2, K1)
        where K1, K2 are the filter kernels.
        """
        K1, K2 = self._create_aperture_mass_kernels()

        # E-mode: Map_E = conv(-g1, K1) + conv(-g2, K2)
        map_e = (convolve(-grid.g1, K1, mode='constant', cval=0.0)
                 + convolve(-grid.g2, K2, mode='constant', cval=0.0))

        # B-mode: Map_B = conv(g1, K2) + conv(-g2, K1)
        map_b = (convolve(grid.g1, K2, mode='constant', cval=0.0)
                 + convolve(-grid.g2, K1, mode='constant', cval=0.0))

        return map_e, map_b
