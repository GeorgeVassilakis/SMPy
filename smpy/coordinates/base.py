from abc import ABC, abstractmethod
import numpy as np

class CoordinateSystem(ABC):
    """
    Abstract base class for coordinate systems in SMPy.
    Defines the interface for handling different coordinate systems (e.g., RA/Dec, pixel).
    """
    
    @abstractmethod
    def get_grid_parameters(self, config):
        """
        Get coordinate-system-specific grid parameters from config.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing coordinate system parameters
            
        Returns
        -------
        dict
            Grid parameters specific to this coordinate system:
            - For RA/Dec: resolution_arcmin
            - For Pixel: downsample_factor, max_grid_size
        """
        pass
    
    @abstractmethod
    def create_grid(self, data_df, boundaries, config):
        """
        Create a grid in the coordinate system.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame containing coordinates and shear data with columns:
            coord1, coord2 (original coordinates)
            coord1_scaled, coord2_scaled (transformed coordinates)
            g1, g2 (shear components)
            weight (optional weights)
        boundaries : dict
            Dictionary containing coordinate boundaries
        config : dict
            Configuration dictionary containing system-specific parameters
            
        Returns
        -------
        tuple
            (g1_grid, g2_grid) numpy arrays
        """
        pass
    
    @abstractmethod
    def calculate_boundaries(self, coord1, coord2):
        """
        Calculate field boundaries in the coordinate system.
        
        Parameters
        ----------
        coord1 : array-like
            First coordinate values (e.g., RA or X)
        coord2 : array-like
            Second coordinate values (e.g., Dec or Y)
        
        Returns
        -------
        tuple
            (scaled_boundaries, true_boundaries) dictionaries containing:
            - coord1_min, coord1_max: boundaries in first coordinate
            - coord2_min, coord2_max: boundaries in second coordinate
            - coord1_name, coord2_name: names of coordinates
            - units: coordinate system units
        """
        pass
    
    @abstractmethod
    def transform_coordinates(self, data_df):
        """
        Transform coordinates if needed (e.g., centering, scaling).
        
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame containing coordinates to transform with columns:
            coord1, coord2 (original coordinates)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with additional columns:
            coord1_scaled, coord2_scaled (transformed coordinates)
        """
        pass

    def _create_shear_grid(self, data_df, idx1, idx2, npix1, npix2):
        """
        Helper method to create shear grid from indices.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame containing shear data
        idx1, idx2 : array-like
            Indices for binning in each dimension
        npix1, npix2 : int
            Number of pixels in each dimension
            
        Returns
        -------
        tuple
            (g1_grid, g2_grid) numpy arrays with dimensions (npix1, npix2)
        """
        # Filter out indices outside the grid
        valid_mask = (idx1 >= 0) & (idx1 < npix2) & (idx2 >= 0) & (idx2 < npix1)
        idx1 = idx1[valid_mask]
        idx2 = idx2[valid_mask]
        g1 = data_df['g1'].values[valid_mask]
        g2 = data_df['g2'].values[valid_mask]
        weight = data_df['weight'].values[valid_mask]
        
        # Initialize grids
        g1_grid = np.zeros((npix1, npix2))
        g2_grid = np.zeros((npix1, npix2))
        weight_grid = np.zeros((npix1, npix2))
        
        # Accumulate weighted values
        np.add.at(g1_grid, (idx2, idx1), g1 * weight)
        np.add.at(g2_grid, (idx2, idx1), g2 * weight)
        np.add.at(weight_grid, (idx2, idx1), weight)
        
        # Normalize by weights
        nonzero_mask = weight_grid != 0
        g1_grid[nonzero_mask] /= weight_grid[nonzero_mask]
        g2_grid[nonzero_mask] /= weight_grid[nonzero_mask]
        
        return g1_grid, g2_grid

    def prepare_data(self, data_df):
        """
        Prepare data for gridding by ensuring correct columns exist.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            Input DataFrame with coordinates and shear data
            
        Returns
        -------
        pd.DataFrame
            Processed DataFrame with all required columns
        """
        required_cols = ['coord1', 'coord2', 'g1', 'g2', 'weight']
        if not all(col in data_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Transform coordinates if not already done
        if 'coord1_scaled' not in data_df.columns:
            data_df = self.transform_coordinates(data_df)
            
        return data_df