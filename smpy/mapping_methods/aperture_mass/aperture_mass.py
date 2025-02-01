"""Implementation of aperture mass mapping using Schneider+98 filter."""

import numpy as np
from smpy.mapping_methods.base import MassMapper
from smpy.filters.processing import s98_aperture_filter

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping using S98 filter.
    
    This class implements the aperture mass mapping technique using the 
    Schneider+98 filter function. The current implementation uses direct
    pixel-by-pixel computation.
    """
    
    @property
    def name(self):
        """Name of the mapping method (`str`)."""
        return "aperture_mass"
        
    def _compute_aperture_mass(self, g1_grid, g2_grid, scale):
        """Compute aperture mass maps using brute force pixel method.
        
        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid
        g2_grid : `numpy.ndarray`
            Second shear component grid
        scale : `float`
            Aperture scale radius in pixels
            
        Returns
        -------
        map_e, map_b : `numpy.ndarray`
            E-mode and B-mode aperture mass maps
        """
        # Get dimensions
        ny, nx = g1_grid.shape
        map_e = np.zeros((ny, nx))
        map_b = np.zeros((ny, nx))
        
        # Compute aperture mass at each position
        for i in range(ny):
            for j in range(nx):
                # Calculate radius to all pixels
                x = abs(np.arange(nx) - j)
                y = abs(np.arange(ny) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                
                # Get filter values
                filter_vals = s98_aperture_filter(radii, scale)
                
                # Calculate tangential/cross shear
                dx = np.arange(nx) - j
                dy = np.arange(ny)[:, np.newaxis] - i
                theta = np.arctan2(dy, dx)
                
                et = -g1_grid * np.cos(2*theta) - g2_grid * np.sin(2*theta)
                ex = +g1_grid * np.sin(2*theta) - g2_grid * np.cos(2*theta)
                
                # Sum filter * shear
                map_e[i,j] = np.sum(filter_vals * et)
                map_b[i,j] = np.sum(filter_vals * ex)
                
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
            E-mode and B-mode aperture mass maps
        """
        # Get filter scale from config
        filter_config = self.config.get('filter', {})
        scale = filter_config.get('scale', 1.0)
        
        # Compute aperture mass maps
        map_e, map_b = self._compute_aperture_mass(g1_grid, g2_grid, scale)
        
        # Compute noise from shape dispersion
        e_sq = g1_grid**2 + g2_grid**2
        noise = np.zeros_like(map_e)
        
        for i in range(map_e.shape[0]):
            for j in range(map_e.shape[1]):
                x = abs(np.arange(map_e.shape[1]) - j)
                y = abs(np.arange(map_e.shape[0]) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                filter_vals = s98_aperture_filter(radii, scale)
                noise[i,j] = np.sqrt(np.sum(filter_vals**2 * e_sq))/np.sqrt(2)
        
        # Return signal-to-noise maps
        return map_e/noise, map_b/noise