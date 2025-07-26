"""Kaiser-Squires mass mapping implementation."""

import numpy as np
from ..base import MassMapper
from smpy.filters import apply_filter

class KaiserSquiresMapper(MassMapper):
    """Implementation of Kaiser-Squires mass mapping."""
    
    @property
    def name(self):
        return "kaiser_squires"
    
    def create_maps(self, g1_grid, g2_grid):
        """Create convergence maps using Kaiser-Squires inversion.
        
        Parameters
        ----------
        g1_grid : numpy.ndarray
            First shear component grid
        g2_grid : numpy.ndarray
            Second shear component grid
            
        Returns
        -------
        kappa_e, kappa_b : numpy.ndarray
            E-mode and B-mode convergence maps
        """
        # Get grid dimensions
        npix_dec, npix_ra = g1_grid.shape

        # Fourier transform the shear components
        g1_hat = np.fft.fft2(g1_grid)
        g2_hat = np.fft.fft2(g2_grid)

        # Create wavenumber grids
        k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
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