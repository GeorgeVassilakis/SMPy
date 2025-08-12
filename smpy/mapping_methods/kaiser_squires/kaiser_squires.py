"""Kaiser-Squires mass mapping implementation.

This module implements the Kaiser-Squires direct inversion algorithm for
reconstructing convergence maps from shear measurements using Fourier
transforms. The method provides both E-mode and B-mode reconstruction
with optional Gaussian smoothing.
"""

import numpy as np
from ..base import MassMapper
from smpy.filters import apply_filter

class KaiserSquiresMapper(MassMapper):
    """Implementation of Kaiser-Squires mass mapping.

    This class implements the Kaiser-Squires direct inversion method for
    reconstructing convergence (mass) maps from weak lensing shear data.
    The algorithm uses Fourier transforms to perform the inversion and
    supports optional smoothing for noise reduction.

    Notes
    -----
    The Kaiser-Squires method directly inverts the shear-convergence relation:
    kappa = D^(-1) * gamma, where D is the differential operator relating
    convergence to shear in Fourier space.
    """
    
    @property
    def name(self):
        """Name identifier for the Kaiser-Squires method.

        Returns
        -------
        method_name : `str`
            String identifier 'kaiser_squires'.
        """
        return "kaiser_squires"
    
    def create_maps(self, g1_grid, g2_grid):
        """Create convergence maps using Kaiser-Squires inversion.

        Perform direct inversion of shear components to reconstruct both
        E-mode and B-mode convergence maps using Fourier transforms.
        Applies optional smoothing if configured.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.

        Returns
        -------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Notes
        -----
        The inversion is performed in Fourier space using the relations:
        kappa_E = ((k1^2 - k2^2) * g1 + 2 * k1 * k2 * g2) / k^2
        kappa_B = ((k1^2 - k2^2) * g2 - 2 * k1 * k2 * g1) / k^2
        """
        # Get grid dimensions
        npix_dec, npix_ra = g1_grid.shape

        # Fourier transform the shear components
        g1_hat = np.fft.fft2(g1_grid)
        g2_hat = np.fft.fft2(g2_grid)

        # Create wavenumber grids with correct ordering (dec=y, ra=x)
        k2, k1 = np.meshgrid(np.fft.fftfreq(npix_dec), np.fft.fftfreq(npix_ra), indexing='ij')
        k_squared = k1**2 + k2**2

        # Avoid division by zero
        k_squared = np.where(k_squared == 0, np.finfo(float).eps, k_squared)

        # Kaiser-Squires inversion in Fourier space
        kappa_e_hat = (1 / k_squared) * ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat)
        kappa_b_hat = (1 / k_squared) * ((k1**2 - k2**2) * g2_hat - 2 * k1 * k2 * g1_hat)

        # Inverse Fourier transform
        kappa_e = np.real(np.fft.ifft2(kappa_e_hat))
        kappa_b = np.real(np.fft.ifft2(kappa_b_hat))

        # Apply smoothing if configured
        smoothing_config = self.method_config.get('smoothing')
        if smoothing_config and smoothing_config.get('type'):
            kappa_e = apply_filter(kappa_e, smoothing_config)
            kappa_b = apply_filter(kappa_b, smoothing_config)

        return kappa_e, kappa_b