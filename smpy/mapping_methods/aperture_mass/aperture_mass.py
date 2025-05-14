"""Implementation of aperture mass mapping using filter functions."""

import numpy as np
from smpy.mapping_methods.base import MassMapper
from smpy.filters.processing import schirmer_filter, schneider_filter
from scipy.ndimage import convolve

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping using configurable filters.
    
    This class implements the aperture mass mapping technique using 
    various filter functions, which can be specified in config.
    """
    
    @property
    def name(self):
        """Name of the mapping method (`str`)."""
        return "aperture_mass"
    
    def _create_aperture_mass_kernels(self, rs, filter_type, l_param, truncation):
        """Create the aperture mass convolution kernels.
        
        Parameters
        ----------
        rs : float
            Aperture filter radius in pixels
        filter_type : str
            Type of filter to use ('schirmer' or 'schneider')
        l_param : int or None
            Parameter for Schneider filter (ignored for Schirmer)
        truncation : float
            Truncation radius in units of scale radius
            
        Returns
        -------
        kernel_1, kernel_2 : numpy.ndarray
            Convolution kernels for E-mode and B-mode calculations:
            kernel_1 = Q(r/rs) * cos(2*phi)
            kernel_2 = Q(r/rs) * sin(2*phi)
        """
        # Create kernel grid
        kernel_radius_in_pixels = truncation * rs
        # Ensure kernel size is odd for a well-defined center pixel
        size = int(np.ceil(2 * kernel_radius_in_pixels))
        size = size + 1 if size % 2 == 0 else size
        
        # Create a grid of coordinates for the kernel
        half_extent = (size - 1) / 2.0
        kernel_coords = np.linspace(-half_extent, half_extent, size)
        X_k, Y_k = np.meshgrid(kernel_coords, kernel_coords)
        
        # Calculate radial distance and polar angle
        R_k = np.sqrt(X_k**2 + Y_k**2)
        Phi_k = np.arctan2(Y_k, X_k)
        
        # Get scaled radii for filter calculation
        scaled_radii_k = R_k / rs
        
        # Get Q filter values based on filter type
        if filter_type == 'schneider':
            q_vals_k = schneider_filter(scaled_radii_k, rs, l_param)
        else:  # Default to Schirmer
            q_vals_k = schirmer_filter(scaled_radii_k, rs)
            
        # Create the kernels
        kernel_1 = q_vals_k * np.cos(2 * Phi_k)
        kernel_2 = q_vals_k * np.sin(2 * Phi_k)
        
        return kernel_1, kernel_2

    def _compute_aperture_mass(self, g1_grid, g2_grid, rs):
        """Compute aperture mass maps using the selected filter via convolution.
        
        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid
        g2_grid : `numpy.ndarray`
            Second shear component grid
        rs : `float`
            Aperture filter radius in pixels
            
        Returns
        -------
        map_e, map_b : `numpy.ndarray`
            E-mode and B-mode aperture mass maps
        """
        # Get filter configuration
        filter_config = self.config.get('filter', {})
        filter_type = filter_config.get('type', 'schirmer').lower()
        l_param = filter_config.get('l', 3) if filter_type == 'schneider' else None
        # Truncation factor for the kernel size
        truncation = filter_config.get('truncation', 1.0) 

        # Create the aperture mass specific convolution kernels
        K1, K2 = self._create_aperture_mass_kernels(rs, filter_type, l_param, truncation)
        
        # E-mode: Map_E = conv(-g1, K1) + conv(-g2, K2)
        map_e = convolve(-g1_grid, K1, mode='constant', cval=0.0) + \
                convolve(-g2_grid, K2, mode='constant', cval=0.0)
        
        # B-mode: Map_B = conv(g1, K2) + conv(-g2, K1)
        map_b = convolve(g1_grid, K2, mode='constant', cval=0.0) + \
                convolve(-g2_grid, K1, mode='constant', cval=0.0)
                
        return map_e, map_b
    
    def create_maps(self, g1_grid, g2_grid):
        """Create aperture mass maps.
        
        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid
        g2_grid : `numpy.ndarray`
            Second shear component grid
            
        Returns
        -------
        map_e, map_b : `numpy.ndarray`
            Raw E-mode and B-mode aperture mass maps
        """
        # Get filter scale from config
        filter_config = self.config.get('filter', {})
        rs = filter_config.get('scale', 1.0)
        
        # Compute and return raw aperture mass maps
        return self._compute_aperture_mass(g1_grid, g2_grid, rs)