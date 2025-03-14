"""Implementation of aperture mass mapping using Schirmer filter."""

import numpy as np
from smpy.mapping_methods.base import MassMapper
from smpy.filters.processing import schirmer_filter

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping using Schirmer filter.
    
    This class implements the aperture mass mapping technique using the 
    Schirmer filter function, which is optimized for the signal-to-noise ratio
    of shear measurements. The implementation uses the direct pixel-by-pixel computation
    method based on Equation 7 and 8 from the documentation.
    """
    
    @property
    def name(self):
        """Name of the mapping method (`str`)."""
        return "aperture_mass"
    
    def _compute_aperture_mass(self, g1_grid, g2_grid, rs):
        """Compute aperture mass maps using the Schirmer filter.
        
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
        
        # Compute aperture mass at each position
        for i in range(ny):
            for j in range(nx):
                # Calculate radius to all pixels (scaled by rs)
                x = abs(np.arange(nx) - j)
                y = abs(np.arange(ny) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                scaled_radii = radii / rs
                
                # Get filter values using the Schirmer filter
                filter_vals = schirmer_filter(scaled_radii, rs)
                
                # Calculate tangential/cross shear components
                dx = np.arange(nx) - j
                dy = np.arange(ny)[:, np.newaxis] - i
                theta = np.arctan2(dy, dx)
                
                # Compute tangential and cross components of ellipticity/shear
                et = -g1_grid * np.cos(2*theta) - g2_grid * np.sin(2*theta)
                ex = +g1_grid * np.sin(2*theta) - g2_grid * np.cos(2*theta)
                
                # Compute aperture mass values (Equation 7)
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
            E-mode and B-mode aperture mass maps (signal-to-noise ratio)
        """
        # Get filter scale from config
        filter_config = self.config.get('filter', {})
        rs = filter_config.get('scale', 1.0)
        
        # Compute aperture mass maps
        map_e, map_b = self._compute_aperture_mass(g1_grid, g2_grid, rs)
        
        # Compute noise from shape dispersion
        e_sq = g1_grid**2 + g2_grid**2
        noise = np.zeros_like(map_e)
        
        for i in range(map_e.shape[0]):
            for j in range(map_e.shape[1]):
                x = abs(np.arange(map_e.shape[1]) - j)
                y = abs(np.arange(map_e.shape[0]) - i)
                X, Y = np.meshgrid(x**2, y**2)
                radii = np.sqrt(X + Y)
                scaled_radii = radii / rs
                
                filter_vals = schirmer_filter(scaled_radii, rs)
                noise[i,j] = np.sqrt(np.sum(filter_vals**2 * e_sq))/np.sqrt(2)
        
        # Return signal-to-noise maps
        return map_e/noise, map_b/noise