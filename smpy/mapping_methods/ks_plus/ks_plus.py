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

from smpy.filters.starlet import (
    inverse_starlet_transform_2d,
    starlet_nscales_support_aware,
    starlet_transform_2d,
)

from ..base import MassMapper

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
        Number of reduced-shear correction steps.
    nscales : `int` or `None`
        Starlet scales for the power-spectrum constraint, or `None` for the
        largest safe number. Larger values are clipped to it.
    extension_size : `str` or `int`
        'double' to pad each side by half the field, or pixels per side.
    use_wavelet_constraints : `bool`
        Whether to rescale wavelet power inside the gaps.
    constrain_B : `bool`
        Whether to apply the wavelet constraint to B modes as well.
    threshold_tau : `float` or `None`
        Decay constant of the DCT threshold in iterations, or `None` for a
        quarter of ``inpainting_iterations``.

    Raises
    ------
    ValueError
        Raised if ``smoothing``, ``extension_size`` or ``threshold_tau`` is
        invalid.

    Notes
    -----
    The KS+ algorithm implements the iterative inpainting scheme described
    in the literature, combining sparsity priors in the DCT domain with
    wavelet-based power spectrum preservation for robust mass map
    reconstruction in the presence of missing data and systematic effects.

    The binary mask encodes data availability only. Pixels are marked as
    data (1) if measurements exist and are valid; gaps (0) are used solely
    where data are missing or invalid. Zero shear values are valid data and
    must not be masked.
    """

    def __init__(self, smoothing, inpainting_iterations, reduced_shear_iterations, nscales,
                 extension_size, use_wavelet_constraints, constrain_B, threshold_tau):
        if smoothing < 0:
            raise ValueError(f"smoothing must be at least 0, got {smoothing}")
        if extension_size != 'double' and not (isinstance(extension_size, int) and extension_size >= 0):
            raise ValueError(f"extension_size must be 'double' or a pixel count, got {extension_size!r}")
        if threshold_tau is not None and threshold_tau <= 0:
            raise ValueError(f"threshold_tau must be positive or null, got {threshold_tau}")
        self.smoothing = smoothing
        self.inpainting_iterations = inpainting_iterations
        self.reduced_shear_iterations = reduced_shear_iterations
        self.nscales = nscales
        self.extension_size = extension_size
        self.use_wavelet_constraints = use_wavelet_constraints
        self.constrain_B = constrain_B
        self.threshold_tau = threshold_tau

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
                g1_corrected, g2_corrected, mask_extended, nscales)

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

    def _inpainting_reconstruction(self, g1_grid, g2_grid, mask, nscales):
        """Perform inpainting-based reconstruction.

        Execute the core KS+ inpainting algorithm using DCT-domain
        sparsity constraints and wavelet-based power spectrum preservation.

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

        Returns
        -------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Notes
        -----
        The inpainting algorithm alternates between DCT thresholding
        and data consistency enforcement, with wavelet-based
        power spectrum constraints to preserve statistical properties.
        """
        # Initial KS inversion to estimate convergence
        kappa_e, kappa_b = self._standard_ks_inversion(g1_grid, g2_grid)

        # Initialize for DCT inpainting
        kappa_complex = kappa_e + 1j * kappa_b

        max_iterations = self.inpainting_iterations

        # Calculate initial threshold max (min fraction deprecated)
        dct_coeffs = fft.dctn(kappa_e, norm='ortho')
        lambda_max = np.max(np.abs(dct_coeffs))

        for i in range(max_iterations):
            # DCT thresholding
            kappa_e_dct = fft.dctn(np.real(kappa_complex), norm='ortho')
            kappa_b_dct = fft.dctn(np.imag(kappa_complex), norm='ortho')

            # Calculate threshold for current iteration
            lambda_i = self._update_threshold(i, lambda_max)

            # Apply threshold
            kappa_e_dct[np.abs(kappa_e_dct) < lambda_i] = 0
            kappa_b_dct[np.abs(kappa_b_dct) < lambda_i] = 0

            # Inverse DCT
            kappa_e = fft.idctn(kappa_e_dct, norm='ortho')
            kappa_b = fft.idctn(kappa_b_dct, norm='ortho')

            # Wavelet-based power spectrum constraints
            if self.use_wavelet_constraints:
                # Apply power matching to E-mode only by default. KS+ leaves B
                # free.
                kappa_e = self._apply_wavelet_constraints(kappa_e, mask, nscales)
                if self.constrain_B:
                    kappa_b = self._apply_wavelet_constraints(kappa_b, mask, nscales)

            # Enforce consistency with observed data
            gamma1, gamma2 = self._kappa_to_gamma(kappa_e, kappa_b)

            # Replace with observed data on data pixels (mask==1)
            gamma1[mask > 0] = g1_grid[mask > 0]
            gamma2[mask > 0] = g2_grid[mask > 0]

            # Convert back to convergence
            kappa_e, kappa_b = self._gamma_to_kappa(gamma1, gamma2)
            kappa_complex = kappa_e + 1j * kappa_b

        return kappa_e, kappa_b

    def _standard_ks_inversion(self, g1_grid, g2_grid):
        """Perform standard Kaiser-Squires inversion.

        Apply the classical Kaiser-Squires Fourier-domain inversion
        to convert shear components to convergence maps.

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
        Uses the standard KS relations in Fourier space:
        kappa_E = ((k1^2 - k2^2) * g1 + 2 * k1 * k2 * g2) / k^2
        kappa_B = ((k1^2 - k2^2) * g2 - 2 * k1 * k2 * g1) / k^2
        """
        # Get dimensions
        npix_dec, npix_ra = g1_grid.shape

        # Fourier transform the shear components
        g1_hat = np.fft.fft2(g1_grid)
        g2_hat = np.fft.fft2(g2_grid)

        # Create wavenumber grids
        k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
        k_squared = k1**2 + k2**2

        # Avoid division by zero
        k_squared[k_squared == 0] = np.finfo(float).eps

        # Kaiser-Squires inversion in Fourier space
        kappa_e_hat = (1 / k_squared) * ((k1**2 - k2**2) * g1_hat + 2 * k1 * k2 * g2_hat)
        kappa_b_hat = (1 / k_squared) * ((k1**2 - k2**2) * g2_hat - 2 * k1 * k2 * g1_hat)

        # Inverse Fourier transform
        kappa_e = np.real(np.fft.ifft2(kappa_e_hat))
        kappa_b = np.real(np.fft.ifft2(kappa_b_hat))

        return kappa_e, kappa_b

    def _kappa_to_gamma(self, kappa_e, kappa_b):
        """Convert convergence to shear using Fourier space relation.

        Apply the forward Kaiser-Squires transform to convert convergence
        maps to shear components for data consistency enforcement.

        Parameters
        ----------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Returns
        -------
        gamma1 : `numpy.ndarray`
            First shear component grid.
        gamma2 : `numpy.ndarray`
            Second shear component grid.

        Notes
        -----
        Implements the forward KS transform in Fourier space to ensure
        consistency between convergence and shear during the iterative
        inpainting process.
        """
        # Get dimensions
        npix_dec, npix_ra = kappa_e.shape

        # Combine E and B modes
        kappa_complex = kappa_e + 1j * kappa_b
        kappa_hat = np.fft.fft2(kappa_complex)

        # Create wavenumber grids
        k1, k2 = np.meshgrid(np.fft.fftfreq(npix_ra), np.fft.fftfreq(npix_dec))
        k_squared = k1**2 + k2**2

        # Avoid division by zero
        mask = k_squared > 0

        # Initialize shear components in Fourier space
        gamma1_hat = np.zeros_like(kappa_hat, dtype=complex)
        gamma2_hat = np.zeros_like(kappa_hat, dtype=complex)

        # Apply KS forward transform
        gamma1_hat[mask] = ((k1**2 - k2**2) / k_squared)[mask] * kappa_hat[mask]
        gamma2_hat[mask] = (2 * k1 * k2 / k_squared)[mask] * kappa_hat[mask]

        # Inverse Fourier transform
        gamma1 = np.real(np.fft.ifft2(gamma1_hat))
        gamma2 = np.real(np.fft.ifft2(gamma2_hat))

        return gamma1, gamma2

    def _gamma_to_kappa(self, gamma1, gamma2):
        """Convert shear to convergence using Fourier space relation.

        Apply the inverse Kaiser-Squires transform to convert shear
        components back to convergence maps.

        Parameters
        ----------
        gamma1 : `numpy.ndarray`
            First shear component grid.
        gamma2 : `numpy.ndarray`
            Second shear component grid.

        Returns
        -------
        kappa_e : `numpy.ndarray`
            E-mode convergence map.
        kappa_b : `numpy.ndarray`
            B-mode convergence map.

        Notes
        -----
        This method uses the standard KS inversion to maintain consistency
        in the forward-backward transform cycle during inpainting.
        """
        # This is the standard KS inversion
        return self._standard_ks_inversion(gamma1, gamma2)

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
        """Apply wavelet-based power spectrum constraints.

        Preserve statistical properties of the convergence field by
        normalizing wavelet coefficients in missing data regions to
        match the statistics of observed regions.

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
            Convergence map with corrected power spectrum.

        Notes
        -----
        Uses starlet wavelet decomposition to analyze power at different
        scales and ensures that the reconstructed field maintains proper
        statistical properties in both observed and inpainted regions.
        """
        # Decompose into wavelet coefficients
        wavelet_bands = starlet_transform_2d(kappa, nscales)

        # Use true binary mask across scales (no wavelet of mask)
        data_mask = mask.astype(bool)

        # Stability parameters (hard-coded defaults; not exposed to YAML)
        clip_min = 0.1
        clip_max = 10.0
        sigma_floor_abs = 1e-12
        sigma_floor_rel = 1e-6
        min_samples = 16

        # Process each scale except the coarsest
        for j in range(nscales - 1):
            band = wavelet_bands[j]

            # Collect observed/gap samples using the same mask at all scales
            obs_vals = band[data_mask]
            gap_vals = band[~data_mask]

            # Require minimum sample counts for stable statistics
            if obs_vals.size < min_samples or gap_vals.size < min_samples:
                continue

            # Compute standard deviations
            std_obs = np.std(obs_vals)
            std_gap = np.std(gap_vals)
            std_band = np.std(band)

            # Skip if non-finite
            if not (np.isfinite(std_obs) and np.isfinite(std_gap) and np.isfinite(std_band)):
                continue

            # Robust floor on denominator to avoid huge ratios
            sigma_floor = max(sigma_floor_abs, sigma_floor_rel * std_band)
            denom = max(std_gap, sigma_floor)

            # Compute and clip scale factor
            scale = float(std_obs / denom) if denom > 0 else 1.0
            scale = float(np.clip(scale, clip_min, clip_max))

            # Apply scaling inside gaps only
            band_gap_scaled = gap_vals * scale
            band[~data_mask] = band_gap_scaled
            wavelet_bands[j] = band

        # Reconstruct
        kappa_corrected = inverse_starlet_transform_2d(wavelet_bands)

        return kappa_corrected

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
        The threshold decays as ``lambda_max * exp(-iteration / tau)``, where
        ``tau`` is ``threshold_tau`` or, if that is `None`, a quarter of
        ``inpainting_iterations`` (at least 1).
        """
        tau = self.threshold_tau or max(1.0, float(self.inpainting_iterations) / 4.0)
        return float(lambda_max) * float(np.exp(-float(iteration) / tau))
