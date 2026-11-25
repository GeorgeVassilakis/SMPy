"""Kaiser-Squires mass mapping implementation.

This module implements the Kaiser-Squires direct inversion algorithm for
reconstructing convergence maps from shear measurements using Fourier
transforms. The method provides both E-mode and B-mode reconstruction
with optional Gaussian smoothing.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..base import MassMapper

def _shear_kernel(shape):
    """Return the Fourier-space operator D taking convergence to shear.

    D = (k1^2 - k2^2 + 2i k1 k2) / k^2. It is zero for the uniform mode,
    which produces no shear, and for modes at the highest frequency of an
    even side that also vary along the other axis. That frequency stands for
    both +1/2 and -1/2 cycles per pixel, so the sign of the 2 k1 k2 term is
    undefined there.
    """
    npix_dec, npix_ra = shape
    k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
    k_squared = k1**2 + k2**2
    k_squared[0, 0] = 1.0
    kernel = (k1**2 - k2**2 + 2j * k1 * k2) / k_squared
    kernel[((k1 == -0.5) & (k2 != 0)) | ((k2 == -0.5) & (k1 != 0))] = 0
    return kernel

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
    D = (k1^2 - k2^2 + 2i k1 k2) / k^2. A uniform convergence produces no
    shear, and modes at the highest frequency of an even side that also vary
    along the other axis are dropped.
    """
    gamma = np.fft.ifft2(_shear_kernel(kappa_e.shape) * np.fft.fft2(kappa_e + 1j * kappa_b))
    return gamma.real, gamma.imag

def convergence_from_shear(gamma1, gamma2):
    """Invert a shear field into E- and B-mode convergence maps
    (Kaiser & Squires 1993).

    Parameters
    ----------
    gamma1 : `numpy.ndarray`
        First shear component.
    gamma2 : `numpy.ndarray`
        Second shear component, with the same shape.

    Returns
    -------
    kappa_e : `numpy.ndarray`
        E-mode convergence map, with zero mean.
    kappa_b : `numpy.ndarray`
        B-mode convergence map, with zero mean.

    Notes
    -----
    kappa_E + i kappa_B = D* (gamma1 + i gamma2) in Fourier space, the
    inverse of `shear_from_convergence` for every mode it keeps.
    """
    kappa = np.fft.ifft2(np.conj(_shear_kernel(gamma1.shape)) * np.fft.fft2(gamma1 + 1j * gamma2))
    return kappa.real, kappa.imag

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
        Raised if ``smoothing`` is not a number of at least 0.

    Notes
    -----
    The Kaiser-Squires method directly inverts the shear-convergence relation:
    kappa = D^(-1) * gamma, where D is the differential operator relating
    convergence to shear in Fourier space.
    """

    def __init__(self, smoothing):
        if not isinstance(smoothing, (int, float)) or smoothing < 0:
            raise ValueError(f"smoothing must be a number of at least 0, got {smoothing!r}")
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
        kappa_e, kappa_b = convergence_from_shear(grid.g1, grid.g2)
        return gaussian_filter(kappa_e, self.smoothing), gaussian_filter(kappa_b, self.smoothing)
