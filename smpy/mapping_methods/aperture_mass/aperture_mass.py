"""Aperture mass mapping implementation."""

import numpy as np
from ..base import MassMapper
from smpy.filters import apply_aperture_filter

class ApertureMassMapper(MassMapper):
    """Implementation of aperture mass mapping."""
    
    @property
    def name(self):
        return "aperture_mass"
    
    def _compute_tangential_shear(self, g1_grid, g2_grid):
        """Compute tangential shear relative to grid center.
        
        Parameters
        ----------
        g1_grid, g2_grid : numpy.ndarray
            Shear component grids
            
        Returns
        -------
        gt : numpy.ndarray
            Tangential shear grid
        """
        ny, nx = g1_grid.shape
        y, x = np.indices((ny, nx))
        
        # Center coordinates
        x0, y0 = nx//2, ny//2
        
        # Compute angle relative to center
        dx = x - x0
        dy = y - y0
        theta = np.arctan2(dy, dx)
        
        # Compute tangential shear
        gt = -(g1_grid * np.cos(2*theta) + g2_grid * np.sin(2*theta))
        
        return gt
    
    def create_maps(self, g1_grid, g2_grid):
        """Create aperture mass maps.
        
        Parameters
        ----------
        g1_grid, g2_grid : numpy.ndarray
            Shear component grids
            
        Returns
        -------
        map_e, map_b : numpy.ndarray
            E-mode and B-mode aperture mass maps
        """
        # Compute tangential shear
        gt = self._compute_tangential_shear(g1_grid, g2_grid)
        
        # Get filter configuration
        filter_config = self.config.get('filter', {})
        
        # Apply aperture filter to tangential shear
        map_e = apply_aperture_filter(gt, filter_config)
        
        # For B-mode, rotate shears by 45 degrees
        g1_rot = -g2_grid
        g2_rot = g1_grid
        gt_b = self._compute_tangential_shear(g1_rot, g2_rot)
        map_b = apply_aperture_filter(gt_b, filter_config)
        
        return map_e, map_b