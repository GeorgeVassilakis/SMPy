"""Implementation of Kaiser-Squires Plus (KS+) mass mapping method.

This enhanced Kaiser-Squires method corrects for systematic effects including
missing data, field borders, and reduced shear approximation using sparsity
priors in the DCT domain and wavelet-based power spectrum constraints.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import fft
from jax.scipy.special import erf

from smpy.mapping_methods import KSPlusMapper as NumpyKSPlusMapper

from .filters import apply_filter, gaussian_filter
from .kaiser_squires import convergence_from_shear, shear_from_convergence

class KSPlusMapper(NumpyKSPlusMapper):
    """KS+ mass mapping with JAX.

    Takes the settings of `smpy.mapping_methods.KSPlusMapper`.
    """

    def create_maps(self, grid, *, smoothing=None):
        """Create convergence maps using Kaiser-Squires Plus inversion.

        Perform enhanced mass mapping reconstruction with iterative
        inpainting, reduced shear corrections, and wavelet-based
        power spectrum preservation.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned reduced shear and weights. Finite shear with a finite,
            positive weight marks an observed pixel.
        smoothing : `float` or `jax.Array`, optional
            Gaussian sigma in pixels, up to the configured smoothing. By
            default, use the configured value, which sets the maximum
            Gaussian kernel radius.

        Returns
        -------
        kappa_e : `jax.Array`
            E-mode convergence map.
        kappa_b : `jax.Array`
            B-mode convergence map.

        Notes
        -----
        The reduced-shear loop uses the unsmoothed, extended convergence.
        Gaussian smoothing is applied to the cropped outputs after the last
        iteration. Derivatives hold the data mask and the hard DCT threshold
        decisions fixed.

        The wavelet constraint rescales a band only where its spread inside
        the gaps exceeds four rounding units of the largest map value. A
        smaller spread is rounding error.
        """
        height, width = grid.g1.shape
        if self.extension_size == 'double':
            padding = height // 2, width // 2
        else:
            padding = (int(self.extension_size),) * 2
        pad_y, pad_x = padding
        nscales = self._starlet_nscales(height + 2 * pad_y, width + 2 * pad_x)
        direct = None
        if padding == (0, 0) and self.reduced_shear_iterations == 1:
            direct = convergence_from_shear(grid.g1, grid.g2)
        kappa_e, kappa_b = _reconstruct(
            grid.g1, grid.g2, grid.weights, direct,
            padding=padding,
            nscales=nscales,
            inpainting_iterations=int(self.inpainting_iterations),
            reduced_shear_iterations=int(self.reduced_shear_iterations),
            use_wavelet_constraints=bool(self.use_wavelet_constraints),
        )
        sigma = self.smoothing if smoothing is None else smoothing
        radius = int(4 * self.smoothing + 0.5)
        return gaussian_filter(kappa_e, sigma, radius=radius), gaussian_filter(kappa_b, sigma, radius=radius)

def _masked_std(band, region):
    """Return the population standard deviation of a band in a region."""
    count = jnp.maximum(jnp.sum(region), 1)
    mean = jnp.sum(jnp.where(region, band, 0)) / count
    centered = jnp.where(region, band - mean, 0)
    variance = jnp.sum(centered**2) / count
    positive = variance > 0
    # The derivative of sqrt at zero is singular, even on an unused branch.
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, variance, 1)), 0)

def _apply_wavelet_constraints(kappa, mask, nscales):
    """Match starlet detail power inside the gaps to that outside."""
    gaps = ~mask
    coarse = kappa
    correction = jnp.zeros_like(kappa)
    # A band that is constant in the gaps keeps a spread of rounding
    # error there, up to about one rounding unit of the map. Dividing by
    # it would turn that error into power.
    rounding = 4 * jnp.finfo(kappa.dtype).eps * jnp.max(jnp.abs(kappa))
    # The detail bands and final coarse image sum to kappa. Accumulate
    # only their corrections so each band can be used as it is computed.
    for scale in range(nscales - 1):
        smoothed = apply_filter(coarse, 1 << scale)
        detail = coarse - smoothed
        sigma_in = _masked_std(detail, gaps)
        sigma_out = _masked_std(detail, mask)
        has_power = sigma_in > rounding
        ratio = sigma_out / jnp.where(has_power, sigma_in, 1)
        correction += jnp.where(gaps & has_power, detail * (ratio - 1), 0)
        coarse = smoothed
    return kappa + correction

def _dct_coefficients(kappa_e, kappa_b):
    """Compute the orthonormal DCT-II of E + iB."""
    return lax.complex(fft.dctn(kappa_e, type=2, norm='ortho'), fft.dctn(kappa_b, type=2, norm='ortho'))

def _inpainting_reconstruction(g1, g2, mask, nscales, fractions, lambda_max, use_wavelet_constraints):
    """Fill a shear field's gaps by sparse inpainting."""
    match_power = jnp.any(mask) & jnp.any(~mask)

    def match_wavelet_power(maps):
        return tuple(_apply_wavelet_constraints(kappa, mask, nscales) for kappa in maps)

    def inpaint(state, fraction):
        maps, lambda_max = state
        coefficients = _dct_coefficients(*maps)
        magnitudes = jnp.abs(lax.stop_gradient(coefficients))
        # The thresholds start at the largest coefficient of the first
        # reconstruction. Measured on the coefficients it thresholds, it
        # keeps that coefficient whatever the rounding.
        lambda_max = jnp.where(lambda_max < 0, jnp.max(magnitudes), lambda_max)
        coefficients = jnp.where(magnitudes < lambda_max * fraction, 0, coefficients)
        maps = (fft.idctn(coefficients.real, type=2, norm='ortho'),
                fft.idctn(coefficients.imag, type=2, norm='ortho'))
        if use_wavelet_constraints:
            maps = lax.cond(match_power, match_wavelet_power, _keep_maps, maps)

        gamma1, gamma2 = shear_from_convergence(*maps)
        maps = convergence_from_shear(jnp.where(mask, g1, gamma1), jnp.where(mask, g2, gamma2))
        return (maps, lambda_max), None

    state, _ = lax.scan(inpaint, (convergence_from_shear(g1, g2), lambda_max), fractions)
    return state

def _keep_maps(maps):
    """Return maps unchanged when there are no two regions to match."""
    return maps

@partial(jax.jit, static_argnames=(
    "padding", "nscales", "inpainting_iterations", "reduced_shear_iterations", "use_wavelet_constraints",
))
def _reconstruct(g1, g2, weights, direct, *, padding, nscales, inpainting_iterations,
                 reduced_shear_iterations, use_wavelet_constraints):
    """Reconstruct convergence with field extension and reduced shear."""
    height, width = g1.shape
    pad_y, pad_x = padding
    pad_width = ((pad_y, pad_y), (pad_x, pad_x))
    finite_data = jnp.isfinite(g1) & jnp.isfinite(g2)
    mask = jnp.pad(finite_data & jnp.isfinite(weights) & (weights > 0), pad_width)
    # Retain finite gap values in the seed. The mask controls replacement.
    dtype = jnp.result_type(g1, g2, 1.0)
    g1 = jnp.pad(jnp.where(finite_data, g1, 0).astype(dtype), pad_width)
    g2 = jnp.pad(jnp.where(finite_data, g2, 0).astype(dtype), pad_width)

    def solve():
        iteration = jnp.arange(inpainting_iterations, dtype=g1.dtype)
        fractions = 1 - erf(2.8 * iteration / inpainting_iterations)

        def correct_reduced_shear(_, state):
            maps, lambda_max = state
            factor = 1 - jnp.where(mask, maps[0], 0)
            return _inpainting_reconstruction(g1 * factor, g2 * factor, mask, nscales,
                                              fractions, lambda_max, use_wavelet_constraints)

        zeros = jnp.zeros_like(g1)
        # A negative threshold scale marks the uncorrected inversion, whose
        # first inpainting step measures it.
        unmeasured = jnp.asarray(-1, dtype=g1.dtype)
        (kappa_e, kappa_b), _ = lax.fori_loop(0, reduced_shear_iterations, correct_reduced_shear,
                                              ((zeros, zeros), unmeasured))
        crop = (slice(pad_y, pad_y + height), slice(pad_x, pad_x + width))
        return kappa_e[crop], kappa_b[crop]

    if direct is not None:
        return lax.cond(jnp.all(mask), lambda: direct, solve)
    return solve()
