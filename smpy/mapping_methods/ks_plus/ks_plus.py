"""Implementation of Kaiser-Squires Plus (KS+) mass mapping method.

This enhanced Kaiser-Squires method corrects for systematic effects including
missing data, field borders, and reduced shear approximation using sparsity
priors in the DCT domain and wavelet-based power spectrum constraints.
"""

import numpy as np
from scipy import fft
from scipy.ndimage import gaussian_filter
from ..base import MassMapper
from smpy.filters.starlet import starlet_transform_2d, inverse_starlet_transform_2d

class KSPlusMapper(MassMapper):
    """Implementation of Kaiser-Squires Plus mass mapping.

    The KS+ method extends the standard Kaiser-Squires approach with
    several enhancements for improved mass reconstruction accuracy:

    1. Correcting for missing data using DCT-domain sparsity
    2. Reducing field border effects through field extension
    3. Iteratively correcting for reduced shear approximation
    4. Preserving proper statistical properties using wavelet constraints

    Notes
    -----
    The KS+ algorithm implements the iterative inpainting scheme described
    in the literature, combining sparsity priors in the DCT domain with
    wavelet-based power spectrum preservation for robust mass map
    reconstruction in the presence of missing data and systematic effects.
    """
    
    @property
    def name(self):
        """Name identifier for the KS+ method.

        Returns
        -------
        method_name : `str`
            String identifier 'ks_plus'.
        """
        return "ks_plus"
    
    def create_maps(self, g1_grid, g2_grid):
        """Create convergence maps using Kaiser-Squires Plus inversion.

        Perform enhanced mass mapping reconstruction with iterative
        inpainting, reduced shear corrections, and wavelet-based
        power spectrum preservation.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First reduced shear component grid.
        g2_grid : `numpy.ndarray`
            Second reduced shear component grid.

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
        # Get dimensions and configuration
        npix_dec, npix_ra = g1_grid.shape
        config = self.method_config
        
        # Initialize output
        kappa_e = np.zeros_like(g1_grid)
        kappa_b = np.zeros_like(g1_grid)
        
        # Set up mask (1 where data exists, 0 in gaps)
        mask = self._create_mask(g1_grid, g2_grid)
        
        # Extend field to handle border effects
        extension_config = config.get('extension_size', 'double')
        if extension_config == 'double':
            # Double the field size (add half the field width on each side)
            extension_size_dec = npix_dec // 2
            extension_size_ra = npix_ra // 2
        else:
            # Use the specified number of pixels
            try:
                extension_size_dec = extension_size_ra = int(extension_config)
            except (ValueError, TypeError):
                print(f"Warning: Invalid extension_size '{extension_config}', using default 'double'")
                extension_size_dec = npix_dec // 2
                extension_size_ra = npix_ra // 2

        g1_extended, g2_extended, mask_extended = self._extend_field(
            g1_grid, g2_grid, mask, extension_size_dec, extension_size_ra)
        
        # Reduced shear correction loop
        max_iterations = config.get('reduced_shear_iterations', 3)
        for k in range(max_iterations):
            # Correct for reduced shear: γ = g(1-κ)
            g1_corrected = g1_extended.copy()
            g2_corrected = g2_extended.copy()
            
            if k > 0:  # Skip on first iteration (κ=0)
                g1_corrected[mask_extended > 0] *= (1 - kappa_e_extended[mask_extended > 0])
                g2_corrected[mask_extended > 0] *= (1 - kappa_e_extended[mask_extended > 0])
            
            # Perform inpainting-based reconstruction
            kappa_e_extended, kappa_b_extended = self._inpainting_reconstruction(
                g1_corrected, g2_corrected, mask_extended, config)
            
            # Extract the central part for next iteration
            start_dec = extension_size_dec
            start_ra = extension_size_ra
            end_ra = start_ra + npix_ra
            end_dec = start_dec + npix_dec
            kappa_e = kappa_e_extended[start_dec:end_dec, start_ra:end_ra]
            kappa_b = kappa_b_extended[start_dec:end_dec, start_ra:end_ra]
            
            # Apply smoothing if configured
            smoothing_config = self.method_config.get('smoothing')
            if smoothing_config and smoothing_config.get('type'):
                sigma = smoothing_config.get('sigma')
                if sigma is None:
                    raise ValueError("Smoothing enabled but 'sigma' parameter missing. Please specify smoothing.sigma in your configuration.")
                kappa_e = gaussian_filter(kappa_e, sigma=sigma)
                kappa_b = gaussian_filter(kappa_b, sigma=sigma)
        
        return kappa_e, kappa_b
    
    def _create_mask(self, g1_grid, g2_grid):
        """Create binary mask from shear data.

        Identify missing data gaps by detecting NaN values and zero
        shear measurements in both components.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.

        Returns
        -------
        mask : `numpy.ndarray`
            Binary mask (1 where data exists, 0 in gaps).
        """
        # Identify missing data (gaps)
        mask = np.ones_like(g1_grid)
        mask[(np.isnan(g1_grid)) | (np.isnan(g2_grid))] = 0
        mask[(g1_grid == 0) & (g2_grid == 0)] = 0
        return mask
    
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
    
    def _inpainting_reconstruction(self, g1_grid, g2_grid, mask, config):
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
        config : `dict`
            Configuration dictionary with algorithm parameters.

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
        
        # Get algorithm parameters
        max_iterations = config.get('inpainting_iterations', 100)
        
        # Calculate initial threshold
        dct_coeffs = fft.dctn(kappa_e)
        lambda_max = np.max(np.abs(dct_coeffs))
        min_threshold_fraction = config.get('min_threshold_fraction', 0.0)
        if min_threshold_fraction > 0:
            lambda_min = lambda_max * min_threshold_fraction
        else:
            lambda_min = 0.0
        
        for i in range(max_iterations):
            # DCT thresholding
            kappa_e_dct = fft.dctn(np.real(kappa_complex))
            kappa_b_dct = fft.dctn(np.imag(kappa_complex))
            
            # Calculate threshold for current iteration
            lambda_i = self._update_threshold(i, max_iterations, lambda_min, lambda_max)
            
            # Apply threshold
            kappa_e_dct[np.abs(kappa_e_dct) < lambda_i] = 0
            kappa_b_dct[np.abs(kappa_b_dct) < lambda_i] = 0
            
            # Inverse DCT
            kappa_e = fft.idctn(kappa_e_dct)
            kappa_b = fft.idctn(kappa_b_dct)
            
            # Wavelet-based power spectrum constraints
            if config.get('use_wavelet_constraints', True):
                kappa_e = self._apply_wavelet_constraints(kappa_e, mask)
                kappa_b = self._apply_wavelet_constraints(kappa_b, mask)
            
            # Enforce consistency with observed data
            gamma1, gamma2 = self._kappa_to_gamma(kappa_e, kappa_b)
            
            # Replace with observed data outside the mask
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
    
    def _apply_wavelet_constraints(self, kappa, mask):
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
        # Determine number of scales
        min_dim = min(kappa.shape)
        nscales = int(np.log2(min_dim))
        
        # Decompose into wavelet coefficients
        wavelet_bands = starlet_transform_2d(kappa, nscales)
        
        # Also decompose the mask to get proper correspondence at each scale
        mask_bands = starlet_transform_2d(mask.astype(float), nscales)
        
        # Process each scale except the coarsest
        for j in range(nscales-1):
            # Create binary mask for this scale
            scale_mask = mask_bands[j] > 0.5
            
            if np.sum(~scale_mask) > 0 and np.sum(scale_mask) > 0:
                # Calculate standard deviations
                std_out = np.std(wavelet_bands[j][scale_mask])
                std_in = np.std(wavelet_bands[j][~scale_mask])
                
                if std_in > 0:
                    # Apply normalization factor inside the gaps
                    scale_factor = std_out / std_in
                    wavelet_bands[j][~scale_mask] *= scale_factor
        
        # Reconstruct
        kappa_corrected = inverse_starlet_transform_2d(wavelet_bands)
        
        return kappa_corrected
    
    def _update_threshold(self, iteration, max_iterations, lambda_min, lambda_max):
        """Update threshold following exponential decay.

        Compute the DCT coefficient threshold for the current iteration
        using exponential decay from maximum to minimum values.

        Parameters
        ----------
        iteration : `int`
            Current iteration number (0-indexed).
        max_iterations : `int`
            Maximum number of iterations.
        lambda_min : `float`
            Minimum threshold value (final iteration).
        lambda_max : `float`
            Maximum threshold value (first iteration).

        Returns
        -------
        lambda_i : `float`
            Threshold for current iteration.

        Notes
        -----
        The threshold decreases exponentially to gradually reduce the
        sparsity constraint and allow finer details to emerge in the
        reconstructed convergence field.
        """
        # Exponential threshold decrease
        alpha = float(iteration) / max_iterations
        return lambda_max * (lambda_min / lambda_max) ** alpha