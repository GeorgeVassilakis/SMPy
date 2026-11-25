"""Kaiser-Squires mass mapping implementation.

This module implements the Kaiser-Squires direct inversion algorithm for
reconstructing convergence maps from shear measurements using Fourier
transforms. The method provides both E-mode and B-mode reconstruction
with optional Gaussian smoothing.
"""

import jax
import jax.numpy as jnp

from smpy.mapping_methods import KaiserSquiresMapper as NumpyKaiserSquiresMapper

from .filters import gaussian_filter

def _shear_kernel(shape, dtype):
    """Return the Fourier-space operator D taking convergence to shear.

    D = (k1^2 - k2^2 + 2i k1 k2) / k^2. It is zero for the uniform mode,
    which produces no shear, and for modes at the highest frequency of an
    even side that also vary along the other axis. That frequency stands for
    both +1/2 and -1/2 cycles per pixel, so the sign of the 2 k1 k2 term is
    undefined there.
    """
    npix_dec, npix_ra = shape
    k1 = jnp.fft.fftfreq(npix_ra, dtype=dtype)[None, :]
    k2 = jnp.fft.fftfreq(npix_dec, dtype=dtype)[:, None]
    k_squared = k1**2 + k2**2
    k_squared = jnp.where(k_squared == 0, 1.0, k_squared)
    kernel = (k1**2 - k2**2 + 2j * k1 * k2) / k_squared
    nyquist_ra = (npix_ra % 2 == 0) & (jnp.arange(npix_ra)[None, :] == npix_ra // 2)
    nyquist_dec = (npix_dec % 2 == 0) & (jnp.arange(npix_dec)[:, None] == npix_dec // 2)
    mixed_nyquist = (nyquist_ra & (k2 != 0)) | (nyquist_dec & (k1 != 0))
    return jnp.where(mixed_nyquist, 0, kernel)

@jax.jit
def shear_from_convergence(kappa_e, kappa_b):
    """Compute the shear produced by E- and B-mode convergence maps.

    Parameters
    ----------
    kappa_e : `jax.Array`
        E-mode convergence map.
    kappa_b : `jax.Array`
        B-mode convergence map, with the same shape.

    Returns
    -------
    gamma1 : `jax.Array`
        First shear component.
    gamma2 : `jax.Array`
        Second shear component.

    Notes
    -----
    In Fourier space gamma1 + i gamma2 = D (kappa_E + i kappa_B), with
    D = (k1^2 - k2^2 + 2i k1 k2) / k^2. A uniform convergence produces no
    shear, and modes at the highest frequency of an even side that also vary
    along the other axis are dropped.
    """
    dtype = jnp.result_type(kappa_e, kappa_b, 1.0)
    kappa = jnp.asarray(kappa_e, dtype=dtype) + 1j * jnp.asarray(kappa_b, dtype=dtype)
    gamma = jnp.fft.ifft2(_shear_kernel(kappa.shape, dtype) * jnp.fft.fft2(kappa))
    return gamma.real, gamma.imag

@jax.jit
def convergence_from_shear(gamma1, gamma2):
    """Invert a shear field into E- and B-mode convergence maps.

    Parameters
    ----------
    gamma1 : `jax.Array`
        First shear component.
    gamma2 : `jax.Array`
        Second shear component, with the same shape.

    Returns
    -------
    kappa_e : `jax.Array`
        E-mode convergence map, with zero mean.
    kappa_b : `jax.Array`
        B-mode convergence map, with zero mean.

    Notes
    -----
    Kaiser & Squires (1993) use kappa_E + i kappa_B = D* (gamma1 + i gamma2)
    in Fourier space, the inverse of `shear_from_convergence` for every
    mode it keeps.
    """
    dtype = jnp.result_type(gamma1, gamma2, 1.0)
    gamma = jnp.asarray(gamma1, dtype=dtype) + 1j * jnp.asarray(gamma2, dtype=dtype)
    kappa = jnp.fft.ifft2(jnp.conj(_shear_kernel(gamma.shape, dtype)) * jnp.fft.fft2(gamma))
    return kappa.real, kappa.imag

class KaiserSquiresMapper(NumpyKaiserSquiresMapper):
    """Kaiser-Squires mass mapping with JAX.

    Takes the settings of `smpy.mapping_methods.KaiserSquiresMapper`.
    """

    def create_maps(self, grid, *, smoothing=None):
        """Create convergence maps using Kaiser-Squires inversion.

        Perform direct inversion of shear components to reconstruct both
        E-mode and B-mode convergence maps using Fourier transforms, then
        apply Gaussian smoothing.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear.
        smoothing : `float` or `jax.Array`, optional
            Gaussian sigma in map pixels. A scalar tracer is supported.
            Defaults to the mapper's smoothing setting, which also caps the
            filter radius at ``int(4 * self.smoothing + 0.5)``.

        Returns
        -------
        kappa_e : `jax.Array`
            E-mode convergence map.
        kappa_b : `jax.Array`
            B-mode convergence map.

        Notes
        -----
        The inversion is performed in Fourier space using the relations:
        kappa_E = ((k1^2 - k2^2) * g1 + 2 * k1 * k2 * g2) / k^2
        kappa_B = ((k1^2 - k2^2) * g2 - 2 * k1 * k2 * g1) / k^2
        """
        sigma = self.smoothing if smoothing is None else smoothing
        radius = int(4 * self.smoothing + 0.5)
        kappa_e, kappa_b = convergence_from_shear(grid.g1, grid.g2)
        return gaussian_filter(kappa_e, sigma, radius=radius), gaussian_filter(kappa_b, sigma, radius=radius)
