from abc import ABC, abstractmethod
import numpy as np

class CoordinateSystem(ABC):
    """Abstract base class for coordinate systems.
    
    Provides interface for RA/Dec and pixel coordinate systems used in mass mapping.
    Each coordinate system must implement methods for creating grids, handling boundaries,
    and transforming coordinates appropriately.
    """
    
    @abstractmethod
    def get_grid_parameters(self, config):
        """Get grid parameters from config for the specific coordinate system.

        Parameters
        ----------
        config : `dict`
            Configuration dictionary containing coordinate system parameters

        Returns
        -------
        dict
            Grid parameters dictionary:
            - For RA/Dec: resolution_arcmin for setting grid spacing
            - For Pixel: downsample_factor and max_grid_size for binning
        """
    
    @abstractmethod
    def create_grid(self, data_df, boundaries, config):
        """Create a shear grid by binning data in the coordinate system.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame containing coordinates, shear components and weights.
            Must include coord1_scaled and coord2_scaled from transform_coordinates.
        boundaries : `dict`
            Coordinate boundaries from calculate_boundaries()
        config : `dict`
            Configuration dictionary containing system parameters
            
        Returns
        -------
        g1_grid, g2_grid : `numpy.ndarray`
            2D arrays containing binned shear values on regular grid
        """
    
    @abstractmethod
    def calculate_boundaries(self, coord1, coord2):
        """Calculate field boundaries and setup coordinate labels.

        Parameters
        ----------
        coord1 : `numpy.ndarray`
            First coordinate values (RA or X pixel coordinates)
        coord2 : `numpy.ndarray`
            Second coordinate values (Dec or Y pixel coordinates)
        
        Returns
        -------
        scaled_boundaries, true_boundaries : `dict`
            Dictionaries containing coordinate ranges and labels:
            coord1_min/max, coord2_min/max, coord1_name, coord2_name, units
        """
    
    @abstractmethod
    def transform_coordinates(self, data_df):
        """Transform coordinates if needed (e.g., centering, scaling).

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with original coord1, coord2 coordinate columns
        
        Returns
        -------
        transformed_df : `pandas.DataFrame`
            DataFrame with additional coord1_scaled, coord2_scaled columns
            containing transformed coordinates ready for gridding
        """

    def _create_shear_grid(self, data_df, idx1, idx2, npix1, npix2):
        """Create weighted shear grid from binning indices.

        Helper method that bins shear values into a regular grid using provided indices.
        Handles weighting and normalization of shear values.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame containing g1, g2 shear components and weights
        idx1, idx2 : `numpy.ndarray`
            Bin indices for each dimension of the grid
        npix1, npix2 : `int`
            Number of pixels in each dimension of output grid
            
        Returns
        -------
        g1_grid, g2_grid : `numpy.ndarray`
            2D arrays of binned, weighted shear values
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
        """Prepare data for gridding by validating and transforming coordinates.

        Checks that required columns exist and ensures coordinates are transformed
        before gridding.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Input DataFrame containing coordinates and shear measurements
            
        Returns
        -------
        pd.DataFrame
            Processed DataFrame with all required columns and transformed coordinates
            
        Raises
        ------
        ValueError
            If required columns are missing from the input DataFrame
        """
        required_cols = ['coord1', 'coord2', 'g1', 'g2', 'weight']
        if not all(col in data_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Transform coordinates if not already done
        if 'coord1_scaled' not in data_df.columns:
            data_df = self.transform_coordinates(data_df)
            
        return data_df