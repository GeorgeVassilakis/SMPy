"""Base class for mass mapping methods.

This module defines the abstract base class that all mass mapping
implementations must inherit from, providing a consistent interface
for mass mapping operations.
"""

from abc import ABC, abstractmethod
from smpy.plotting import plot
import os

class MassMapper(ABC):
    """Abstract base class for mass mapping methods.

    This class defines the interface that all mass mapping implementations
    must follow. Subclasses must implement the abstract methods to provide
    specific mass mapping algorithms.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary with nested structure containing general
        settings, method-specific parameters, and plotting options.
    """
    
    def __init__(self, config):
        """Initialize mass mapper with configuration.

        Parameters
        ----------
        config : `dict`
            Configuration dictionary with nested structure containing
            'general', 'methods', and 'plotting' sections.
        """
        self.config = config
        # Direct access helper properties for nested config
        self.general_config = config['general']
        self.method_config = config['methods'][self.name]
        self.plotting_config = config['plotting']
        
    @abstractmethod
    def create_maps(self, g1_grid, g2_grid):
        """Create mass maps from shear grids.

        This method must be implemented by subclasses to perform the
        actual mass mapping computation from input shear grids.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.

        Returns
        -------
        map_e : `numpy.ndarray`
            E-mode mass map.
        map_b : `numpy.ndarray`
            B-mode mass map.
        """
        
    def run(self, g1_grid, g2_grid, scaled_boundaries, true_boundaries):
        """Run complete mass mapping pipeline.

        Execute the mass mapping algorithm and handle output generation
        including plotting and file saving based on configuration.

        Parameters
        ----------
        g1_grid : `numpy.ndarray`
            First shear component grid.
        g2_grid : `numpy.ndarray`
            Second shear component grid.
        scaled_boundaries : `dict`
            Scaled coordinate boundaries for plotting.
        true_boundaries : `dict`
            True coordinate boundaries for WCS information.

        Returns
        -------
        maps : `dict`
            Dictionary containing 'E' and 'B' mode mass maps.
        """
        # Create maps
        map_e, map_b = self.create_maps(g1_grid, g2_grid)
        
        # Store maps
        maps = {
            'E': map_e,
            'B': map_b
        }
        
        # Create method-specific output directory
        method_output_dir = f"{self.general_config['output_directory']}/{self.name}"
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Plot maps
        for mode in self.general_config['mode']:
            plot_map = maps[mode]
            plot_config = self.plotting_config.copy()
            # Ensure plotting knows which coordinate system to use
            plot_config['coordinate_system'] = self.general_config.get('coordinate_system', 'radec')
            if plot_config['coordinate_system'] == 'pixel':
                # Surface the axis reference choice from config when in pixel mode
                plot_config['axis_reference'] = self.config['general']['pixel'].get('pixel_axis_reference', 'catalog')
            plot_config['plot_title'] = f"{self.plotting_config['plot_title']} ({mode}-mode)"
            output_name = (f"{self.general_config['output_directory']}/{self.name}/"
                         f"{self.general_config['output_base_name']}_{self.name}_{mode.lower()}_mode.png")
            plot.plot_mass_map(plot_map, scaled_boundaries, true_boundaries, plot_config, output_name)
            
        return maps
        
    @property
    @abstractmethod
    def name(self):
        """Name of the mapping method.

        Returns
        -------
        method_name : `str`
            String identifier for the mass mapping method.
        """