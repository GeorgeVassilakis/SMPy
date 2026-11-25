"""Implementation of Kaiser-Squires Plus (KS+) mass mapping method.

This enhanced Kaiser-Squires method corrects for systematic effects including
missing data, field borders, and reduced shear approximation using sparsity
priors in the DCT domain and wavelet-based power spectrum constraints.
"""

import logging
import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy import fft
from jax.scipy.special import erf

from smpy._precision import numpy_double
from smpy.filters.gaussian import gaussian_filter
from smpy.filters.starlet import apply_filter, starlet_nscales_support_aware

from ..base import MassMapper
from ..kaiser_squires.kaiser_squires import convergence_from_shear, shear_from_convergence

logger = logging.getLogger(__name__)

class KSPlusMapper(MassMapper):
    """Implementation of Kaiser-Squires Plus mass mapping.

    The KS+ method extends the standard Kaiser-Squires approach with
    several enhancements for improved mass reconstruction accuracy:

    1. Correcting for missing data using DCT-domain sparsity
    2. Reducing field border effects through field extension
    3. Iteratively correcting for reduced shear approximation
    4. Preserving proper statistical properties using wavelet constraints

    Parameters
    ----------
    smoothing : `float`
        Gaussian smoothing sigma in map pixels; 0 for no smoothing.
    inpainting_iterations : `int`
        DCT thresholding iterations per reduced-shear step.
    reduced_shear_iterations : `int`
        Reconstructions in the reduced-shear loop; 1 reconstructs once,
        without reduced-shear correction.
    nscales : `int` or `None`
        Starlet scales for the power-spectrum constraint, at least 2, or
        `None` for the largest safe number. Larger values are clipped to it.
    extension_size : `str` or `int`
        'double' to pad each side by half the field, or pixels per side.
    use_wavelet_constraints : `bool`
        Whether to match the wavelet power of E and B inside the gaps to that
        outside.

    Raises
    ------
    ValueError
        Raised if a setting is invalid.

    Notes
    -----
    This follows Algorithm 1 of Pires et al. (2020, A&A 638, A141): sparse
    inpainting of E + iB in the DCT domain with a threshold that falls from
    its largest coefficient towards zero, a wavelet power-spectrum constraint
    in the gaps, and an outer loop that corrects for reduced shear.

    The binary mask marks valid measurements as data and missing or invalid
    measurements as gaps. A measured shear of zero is valid data.
    """

    def __init__(self, smoothing, inpainting_iterations, reduced_shear_iterations, nscales,
                 extension_size, use_wavelet_constraints):
        if not isinstance(smoothing, (int, float)) or smoothing < 0:
            raise ValueError(f"smoothing must be a number of at least 0, got {smoothing!r}")
        for name, value in (("inpainting_iterations", inpainting_iterations),
                            ("reduced_shear_iterations", reduced_shear_iterations)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")
        if nscales is not None and not (isinstance(nscales, int) and nscales >= 2):
            raise ValueError(f"nscales must be null or an integer of at least 2, got {nscales!r}")
        if extension_size != 'double' and not (isinstance(extension_size, int) and extension_size >= 0):
            raise ValueError(f"extension_size must be 'double' or a pixel count, got {extension_size!r}")
        self.smoothing = smoothing
        self.inpainting_iterations = inpainting_iterations
        self.reduced_shear_iterations = reduced_shear_iterations
        self.nscales = nscales
        self.extension_size = extension_size
        self.use_wavelet_constraints = use_wavelet_constraints

    @numpy_double
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
        kappa_e : `numpy.ndarray` or `jax.Array`
            E-mode convergence map.
        kappa_b : `numpy.ndarray` or `jax.Array`
            B-mode convergence map.

        Notes
        -----
        The reduced-shear loop uses the unsmoothed, extended convergence.
        Gaussian smoothing is applied to the cropped outputs after the last
        iteration. Derivatives hold the data mask and the hard DCT threshold
        decisions fixed.
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

    def _starlet_nscales(self, height, width):
        """Choose the number of starlet scales for an image.

        Parameters
        ----------
        height : `int`
            Image height in pixels.
        width : `int`
            Image width in pixels.

        Returns
        -------
        nscales : `int`
            The configured number of scales, clipped to the safe maximum for
            this image size, or the safe maximum if none is configured.
        """
        nscales = starlet_nscales_support_aware(height, width, self.nscales)
        if self.nscales is not None and self.nscales != nscales:
            warnings.warn(f"KS+ nscales={self.nscales} was clipped to {nscales} for a {height}x{width} image")
        logger.info("KS+ uses %d starlet scales for a %dx%d image", nscales, height, width)
        return nscales

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
    # The detail bands and final coarse image sum to kappa. Accumulate
    # only their corrections so each band can be used as it is computed.
    for scale in range(nscales - 1):
        smoothed = apply_filter(coarse, 1 << scale)
        detail = coarse - smoothed
        sigma_in = _masked_std(detail, gaps)
        sigma_out = _masked_std(detail, mask)
        has_power = sigma_in > 0
        ratio = sigma_out / jnp.where(has_power, sigma_in, 1)
        correction += jnp.where(gaps & has_power, detail * (ratio - 1), 0)
        coarse = smoothed
    return kappa + correction

def _dct_coefficients(kappa_e, kappa_b):
    """Compute the orthonormal DCT-II of E + iB."""
    return lax.complex(fft.dctn(kappa_e, type=2, norm='ortho'), fft.dctn(kappa_b, type=2, norm='ortho'))

def _inpainting_reconstruction(g1, g2, mask, nscales, thresholds, use_wavelet_constraints):
    """Fill a shear field's gaps by sparse inpainting."""
    match_power = jnp.any(mask) & jnp.any(~mask)

    def match_wavelet_power(maps):
        return tuple(_apply_wavelet_constraints(kappa, mask, nscales) for kappa in maps)

    def inpaint(maps, threshold):
        coefficients = _dct_coefficients(*maps)
        below_threshold = jnp.abs(lax.stop_gradient(coefficients)) < threshold
        coefficients = jnp.where(below_threshold, 0, coefficients)
        maps = (fft.idctn(coefficients.real, type=2, norm='ortho'),
                fft.idctn(coefficients.imag, type=2, norm='ortho'))
        if use_wavelet_constraints:
            maps = lax.cond(match_power, match_wavelet_power, _keep_maps, maps)

        gamma1, gamma2 = shear_from_convergence(*maps)
        maps = convergence_from_shear(jnp.where(mask, g1, gamma1), jnp.where(mask, g2, gamma2))
        return maps, None

    maps, _ = lax.scan(inpaint, convergence_from_shear(g1, g2), thresholds)
    return maps

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
        # Start the threshold schedule from the uncorrected inversion.
        initial_dct = _dct_coefficients(*convergence_from_shear(g1, g2))
        lambda_max = jnp.max(jnp.abs(lax.stop_gradient(initial_dct)))
        iteration = jnp.arange(inpainting_iterations, dtype=g1.dtype)
        thresholds = lambda_max * (1 - erf(2.8 * iteration / inpainting_iterations))

        def correct_reduced_shear(_, maps):
            factor = 1 - jnp.where(mask, maps[0], 0)
            return _inpainting_reconstruction(g1 * factor, g2 * factor, mask, nscales,
                                              thresholds, use_wavelet_constraints)

        zeros = jnp.zeros_like(g1)
        kappa_e, kappa_b = lax.fori_loop(0, reduced_shear_iterations, correct_reduced_shear, (zeros, zeros))
        crop = (slice(pad_y, pad_y + height), slice(pad_x, pad_x + width))
        return kappa_e[crop], kappa_b[crop]

    if direct is not None:
        return lax.cond(jnp.all(mask), lambda: direct, solve)
    return solve()
