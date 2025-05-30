from abc import ABC, abstractmethod
from smpy.plotting import plot
import os

class MassMapper(ABC):
    """Abstract base class for mass mapping methods.
    
    All mass mapping implementations should inherit from this class
    and implement the required abstract methods.
    """
    
    def __init__(self, config):
        """Initialize mass mapper.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        
    @abstractmethod
    def create_maps(self, g1_grid, g2_grid):
        """Create mass maps from shear grids.
        
        Parameters
        ----------
        g1_grid : numpy.ndarray
            First shear component grid
        g2_grid : numpy.ndarray
            Second shear component grid
            
        Returns
        -------
        map_e, map_b : numpy.ndarray
            E-mode and B-mode mass maps
        """
        
    def run(self, g1_grid, g2_grid, scaled_boundaries, true_boundaries):
        """Run mass mapping pipeline.
        
        Parameters
        ----------
        g1_grid, g2_grid : numpy.ndarray
            Shear component grids
        scaled_boundaries : dict
            Scaled coordinate boundaries
        true_boundaries : dict
            True coordinate boundaries
            
        Returns
        -------
        maps : dict
            Dictionary containing E/B mode maps
        """
        # Create maps
        map_e, map_b = self.create_maps(g1_grid, g2_grid)
        
        # Store maps
        maps = {
            'E': map_e,
            'B': map_b
        }
        
        # Create method-specific output directory
        method_output_dir = f"{self.config['output_directory']}/{self.name}"
        os.makedirs(method_output_dir, exist_ok=True)
        
        # Plot maps
        for mode in self.config['mode']:
            plot_map = maps[mode]
            plot_config = self.config.copy()
            plot_config['plot_title'] = f"{self.config['plot_title']} ({mode}-mode)"
            output_name = (f"{self.config['output_directory']}/{self.name}/"
                         f"{self.config['output_base_name']}_{self.name}_{mode.lower()}_mode.png")
            plot.plot_convergence(plot_map, scaled_boundaries, true_boundaries, plot_config, output_name)
            
        return maps
        
    @property
    @abstractmethod
    def name(self):
        """Name of the mapping method."""