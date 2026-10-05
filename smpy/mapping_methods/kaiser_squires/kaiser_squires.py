"""Kaiser-Squires mass mapping implementation.

This module implements the Kaiser-Squires direct inversion algorithm for
reconstructing convergence maps from shear measurements using Fourier
transforms. The method provides both E-mode and B-mode reconstruction
with optional Gaussian smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..base import MassMapper

def shear_from_convergence(kappa_e, kappa_b):
    """Compute the shear produced by E- and B-mode convergence maps.

    Parameters
    ----------
    kappa_e : `numpy.ndarray`
        E-mode convergence map.
    kappa_b : `numpy.ndarray`
        B-mode convergence map, with the same shape.

    Returns
    -------
    gamma1 : `numpy.ndarray`
        First shear component.
    gamma2 : `numpy.ndarray`
        Second shear component.

    Notes
    -----
    In Fourier space gamma1 + i gamma2 = D (kappa_E + i kappa_B), with
    D = (k1^2 - k2^2 + 2i k1 k2) / k^2. This is the relation that
    `KaiserSquiresMapper` inverts. A uniform convergence produces no shear.
    """
    npix_dec, npix_ra = kappa_e.shape
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_squared = k1**2 + k2**2
    k_squared[0, 0] = 1.0
    gamma_hat = (k1**2 - k2**2 + 2j * k1 * k2) / k_squared * np.fft.fft2(kappa_e + 1j * kappa_b)
    gamma = np.fft.ifft2(gamma_hat)
    return gamma.real, gamma.imag

class KaiserSquiresMapper(MassMapper):
    """Implementation of Kaiser-Squires mass mapping.

    This class implements the Kaiser-Squires direct inversion method for
    reconstructing convergence (mass) maps from weak lensing shear data.
    The algorithm uses Fourier transforms to perform the inversion and
    supports optional smoothing for noise reduction.

    Parameters
    ----------
    smoothing : `float`
        Gaussian smoothing sigma in map pixels; 0 for no smoothing.

    Raises
    ------
    ValueError
        Raised if ``smoothing`` is negative.

    Notes
    -----
    The Kaiser-Squires method directly inverts the shear-convergence relation:
    kappa = D^(-1) * gamma, where D is the differential operator relating
    convergence to shear in Fourier space.
    """

    def __init__(self, smoothing):
        if smoothing < 0:
            raise ValueError(f"smoothing must be at least 0, got {smoothing}")
        self.smoothing = smoothing

    def create_maps(self, grid):
        """Create convergence maps using Kaiser-Squires inversion.

        Perform direct inversion of shear components to reconstruct both
        E-mode and B-mode convergence maps using Fourier transforms, then
        apply the configured smoothing.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear.

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
        npix_dec, npix_ra = grid.g1.shape

        # Fourier transform the shear components
        g1_hat = np.fft.fft2(grid.g1)
        g2_hat = np.fft.fft2(grid.g2)

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

        return gaussian_filter(kappa_e, self.smoothing), gaussian_filter(kappa_b, self.smoothing)
