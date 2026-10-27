"""Implementation of Kaiser-Squires Plus (KS+) mass mapping method.

This enhanced Kaiser-Squires method corrects for systematic effects including
missing data, field borders, and reduced shear approximation using sparsity
priors in the DCT domain and wavelet-based power spectrum constraints.
"""

import logging
import warnings

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from smpy.filters.starlet import (
    inverse_starlet_transform_2d,
    starlet_nscales_support_aware,
    starlet_transform_2d,
)

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

    The binary mask encodes data availability only. Pixels are marked as
    data (1) if measurements exist and are valid; gaps (0) are used solely
    where data are missing or invalid. Zero shear values are valid data and
    must not be masked.
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

    def create_maps(self, grid):
        """Create convergence maps using Kaiser-Squires Plus inversion.

        Perform enhanced mass mapping reconstruction with iterative
        inpainting, reduced shear corrections, and wavelet-based
        power spectrum preservation.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned reduced shear and weights. Pixels with zero weight are
            gaps to be inpainted.

        Returns
        -------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Notes
        -----
        The algorithm performs the following steps:
        1. Field extension to reduce border effects
        2. Iterative reduced shear correction
        3. DCT-domain inpainting with sparsity constraints
        4. Wavelet-based power spectrum preservation
        5. Optional Gaussian smoothing
        """
        # Get dimensions
        g1_grid, g2_grid = grid.g1, grid.g2
        npix_dec, npix_ra = g1_grid.shape

        # Initialize output
        kappa_e = np.zeros_like(g1_grid)
        kappa_b = np.zeros_like(g1_grid)

        # Set up mask (1 where data exists, 0 in gaps)
        mask = self._create_mask(grid)

        # Extend field to handle border effects
        if self.extension_size == 'double':
            # Double the field size (add half the field width on each side)
            extension_size_dec = npix_dec // 2
            extension_size_ra = npix_ra // 2
        else:
            extension_size_dec = extension_size_ra = self.extension_size

        g1_extended, g2_extended, mask_extended = self._extend_field(
            g1_grid, g2_grid, mask, extension_size_dec, extension_size_ra)
        nscales = self._starlet_nscales(*mask_extended.shape)

        # The DCT threshold starts at the largest coefficient of the first
        # reconstruction
        kappa_e_dct, kappa_b_dct = (fft.dctn(kappa, norm='ortho')
                                    for kappa in convergence_from_shear(g1_extended, g2_extended))
        lambda_max = np.abs(kappa_e_dct + 1j * kappa_b_dct).max()

        # Reduced shear correction loop
        max_iterations = self.reduced_shear_iterations
        kappa_e_extended = np.zeros_like(g1_extended)
        for _ in range(max_iterations):
            # Correct for reduced shear: γ = g(1-κ)
            g1_corrected = g1_extended.copy()
            g2_corrected = g2_extended.copy()
            g1_corrected[mask_extended > 0] *= (1 - kappa_e_extended[mask_extended > 0])
            g2_corrected[mask_extended > 0] *= (1 - kappa_e_extended[mask_extended > 0])

            # Perform inpainting-based reconstruction
            kappa_e_extended, kappa_b_extended = self._inpainting_reconstruction(
                g1_corrected, g2_corrected, mask_extended, nscales, lambda_max)

            # Extract the central part for next iteration
            start_dec = extension_size_dec
            start_ra = extension_size_ra
            end_ra = start_ra + npix_ra
            end_dec = start_dec + npix_dec
            kappa_e = kappa_e_extended[start_dec:end_dec, start_ra:end_ra]
            kappa_b = kappa_b_extended[start_dec:end_dec, start_ra:end_ra]

            kappa_e = gaussian_filter(kappa_e, sigma=self.smoothing)
            kappa_b = gaussian_filter(kappa_b, sigma=self.smoothing)

        return kappa_e, kappa_b

    def _create_mask(self, grid):
        """Create binary mask from data presence/validity.

        Construct the KS+ data-availability mask with the convention
        M=1 for pixels that have valid measurements and M=0 for gaps.
        Zero shear values are considered valid measurements and must
        not be treated as gaps.

        Data presence is defined as:
        - Both shear components are finite (not NaN/Inf); and
        - The pixel weight is positive.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear and weights.

        Returns
        -------
        mask : `numpy.ndarray`
            Binary mask where 1 indicates data present and 0 indicates a gap.
        """
        finite_data = np.isfinite(grid.g1) & np.isfinite(grid.g2)
        positive_weight = np.isfinite(grid.weights) & (grid.weights > 0)
        return (finite_data & positive_weight).astype(grid.g1.dtype)

    def _extend_field(self, g1_grid, g2_grid, mask, extension_size_dec, extension_size_ra):
        """Extend field to reduce border effects.

        Pad the input grids with zeros to create extended fields that
        minimize boundary artifacts during the reconstruction process.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.
        mask : `numpy.ndarray`
            Binary mask.
        extension_size_dec : `int`
            Size of extension in declination (vertical) pixels.
        extension_size_ra : `int`
            Size of extension in right ascension (horizontal) pixels.

        Returns
        -------
        g1_extended : `numpy.ndarray`
            Extended first shear component grid with zero padding.
        g2_extended : `numpy.ndarray`
            Extended second shear component grid with zero padding.
        mask_extended : `numpy.ndarray`
            Extended mask with zero padding.
        """
        # Get dimensions
        npix_dec, npix_ra = g1_grid.shape

        # Create extended grids
        new_dec = npix_dec + 2 * extension_size_dec
        new_ra = npix_ra + 2 * extension_size_ra

        g1_extended = np.zeros((new_dec, new_ra))
        g2_extended = np.zeros((new_dec, new_ra))
        mask_extended = np.zeros((new_dec, new_ra))

        # Insert original field in center
        g1_extended[extension_size_dec:extension_size_dec+npix_dec,
                   extension_size_ra:extension_size_ra+npix_ra] = g1_grid
        g2_extended[extension_size_dec:extension_size_dec+npix_dec,
                   extension_size_ra:extension_size_ra+npix_ra] = g2_grid
        mask_extended[extension_size_dec:extension_size_dec+npix_dec,
                     extension_size_ra:extension_size_ra+npix_ra] = mask

        return g1_extended, g2_extended, mask_extended

    def _inpainting_reconstruction(self, g1_grid, g2_grid, mask, nscales, lambda_max):
        """Fill the gaps in a shear field by sparse inpainting.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.
        mask : `numpy.ndarray`
            Binary mask indicating data locations.
        nscales : `int`
            Number of starlet scales for the wavelet constraint.
        lambda_max : `float`
            DCT threshold at the first iteration.

        Returns
        -------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Notes
        -----
        Steps 7 to 18 of Algorithm 1 of Pires et al. (2020): threshold the
        DCT of kappa_E + i kappa_B, match the wavelet power in the gaps, and
        restore the observed shear where there is data.
        """
        kappa_e, kappa_b = convergence_from_shear(g1_grid, g2_grid)
        for i in range(self.inpainting_iterations):
            # Keep the DCT coefficients of kappa_E + i kappa_B whose magnitude
            # reaches the threshold
            coefficients = fft.dctn(kappa_e, norm='ortho') + 1j * fft.dctn(kappa_b, norm='ortho')
            coefficients[np.abs(coefficients) < self._update_threshold(i, lambda_max)] = 0
            kappa_e = fft.idctn(coefficients.real, norm='ortho')
            kappa_b = fft.idctn(coefficients.imag, norm='ortho')

            if self.use_wavelet_constraints:
                kappa_e = self._apply_wavelet_constraints(kappa_e, mask, nscales)
                kappa_b = self._apply_wavelet_constraints(kappa_b, mask, nscales)

            # Keep the observed shear where there is data
            gamma1, gamma2 = shear_from_convergence(kappa_e, kappa_b)
            gamma1[mask > 0] = g1_grid[mask > 0]
            gamma2[mask > 0] = g2_grid[mask > 0]
            kappa_e, kappa_b = convergence_from_shear(gamma1, gamma2)

        return kappa_e, kappa_b

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

    def _apply_wavelet_constraints(self, kappa, mask, nscales):
        """Match the wavelet power of a map inside the gaps to that outside.

        Parameters
        ----------
        kappa : `numpy.ndarray`
            Convergence map.
        mask : `numpy.ndarray`
            Binary mask (1 where data exists, 0 in gaps).
        nscales : `int`
            Number of starlet scales.

        Returns
        -------
        kappa_corrected : `numpy.ndarray`
            Convergence map whose detail bands have the same standard
            deviation inside the gaps as outside.

        Notes
        -----
        Each starlet detail band is multiplied inside the gaps by its standard
        deviation outside the gaps over that inside, which constrains the power
        spectrum in the gaps (Pires et al. 2020). Bands with no power in the
        gaps are left as they are.
        """
        gaps = mask == 0
        if not gaps.any():
            return kappa
        bands = starlet_transform_2d(kappa, nscales)
        for band in bands[:-1]:
            sigma_in = band[gaps].std()
            if sigma_in > 0:
                band[gaps] *= band[~gaps].std() / sigma_in
        return inverse_starlet_transform_2d(bands)

    def _update_threshold(self, iteration, lambda_max):
        """Return the DCT threshold for an inpainting iteration.

        Parameters
        ----------
        iteration : `int`
            Current iteration number (0-indexed).
        lambda_max : `float`
            Threshold at the first iteration.

        Returns
        -------
        lambda_i : `float`
            Threshold for this iteration.

        Notes
        -----
        The threshold falls towards zero as
        ``lambda_max * (1 - erf(2.8 * iteration / inpainting_iterations))``
        (Pires et al. 2009, MNRAS 395, 1265).
        """
        return lambda_max * (1 - erf(2.8 * iteration / self.inpainting_iterations))
