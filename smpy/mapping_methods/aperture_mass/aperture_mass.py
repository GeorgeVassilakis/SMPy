"""Implementation of aperture mass mapping using filter functions."""

import numpy as np
from smpy.mapping_methods.base import MassMapper
from smpy.filters.processing import schirmer_filter, schneider_filter

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping using configurable filters.
    
    This class implements the aperture mass mapping technique using 
    various filter functions, which can be specified in config.
    """
    
    @property
    def name(self):
        """Name of the mapping method (`str`)."""
        return "aperture_mass"
    
    def _compute_aperture_mass(self, g1_grid, g2_grid, rs):
        """Compute aperture mass maps using the selected filter.
        
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
        # Get dimensions
        ny, nx = g1_grid.shape
        map_e = np.zeros((ny, nx))
        map_b = np.zeros((ny, nx))
        
        # Select filter function based on config
        filter_config = self.config.get('filter', {})
        filter_type = filter_config.get('type', 'schirmer').lower()
        
        # Determine if we're using Schneider filter with specific l parameter
        l = filter_config.get('l', 3) if filter_type == 'schneider' else None
        
        # Compute aperture mass at each position
        for i in range(ny):
            for j in range(nx):
                # Calculate radius to all pixels (scaled by rs)
                x = abs(np.arange(nx) - j)
                y = abs(np.arange(ny) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                scaled_radii = radii / rs
                
                # Get filter values using the selected filter
                if filter_type == 'schneider':
                    filter_vals = schneider_filter(scaled_radii, rs, l)
                else:
                    filter_vals = schirmer_filter(scaled_radii, rs)
                
                # Calculate tangential/cross shear components
                dx = np.arange(nx) - j
                dy = np.arange(ny)[:, np.newaxis] - i
                theta = np.arctan2(dy, dx)
                
                # Compute tangential and cross components of ellipticity/shear
                et = -g1_grid * np.cos(2*theta) - g2_grid * np.sin(2*theta)
                ex = +g1_grid * np.sin(2*theta) - g2_grid * np.cos(2*theta)
                
                # Compute aperture mass values
                map_e[i,j] = np.sum(et * filter_vals)
                map_b[i,j] = np.sum(ex * filter_vals)
                
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