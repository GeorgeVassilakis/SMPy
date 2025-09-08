"""Base class for coordinate systems.

This module defines the abstract base class for coordinate systems used
in mass mapping operations, providing a consistent interface for RA/Dec
and pixel coordinate transformations and gridding.
"""

from abc import ABC, abstractmethod
import numpy as np

class CoordinateSystem(ABC):
    """Abstract base class for coordinate systems.

    Provides interface for RA/Dec and pixel coordinate systems used in mass
    mapping. Each coordinate system must implement methods for creating grids,
    handling boundaries, and transforming coordinates appropriately.

    Notes
    -----
    Subclasses must implement all abstract methods to provide coordinate
    system specific functionality for gridding shear data and handling
    coordinate transformations.
    """
    
    @abstractmethod
    def get_grid_parameters(self, config):
        """Get grid parameters from config for the specific coordinate system.

        Extract coordinate system specific parameters needed for grid creation
        from the configuration dictionary.

        Parameters
        ----------
        config : `dict`
            Configuration dictionary containing coordinate system parameters.

        Returns
        -------
        grid_params : `dict`
            Grid parameters dictionary:
            - For RA/Dec: resolution_arcmin for setting grid spacing
            - For Pixel: downsample_factor and max_grid_size for binning
        """
    
    @abstractmethod
    def create_grid(self, data_df, boundaries, config):
        """Create a shear grid by binning data in the coordinate system.

        Bin shear measurements onto a regular 2D grid using coordinate system
        specific binning strategies.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame containing coordinates, shear components and weights.
            Must include coord1_scaled and coord2_scaled from
            transform_coordinates.
        boundaries : `dict`
            Coordinate boundaries from calculate_boundaries().
        config : `dict`
            Configuration dictionary containing system parameters.

        Returns
        -------
        g1_grid : `numpy.ndarray`
            2D array containing binned first shear component values.
        g2_grid : `numpy.ndarray`
            2D array containing binned second shear component values.
        """
    
    @abstractmethod
    def calculate_boundaries(self, coord1, coord2):
        """Calculate field boundaries and setup coordinate labels.

        Determine coordinate ranges and set up appropriate labels and units
        for the coordinate system.

        Parameters
        ----------
        coord1 : `numpy.ndarray`
            First coordinate values (RA or X pixel coordinates).
        coord2 : `numpy.ndarray`
            Second coordinate values (Dec or Y pixel coordinates).

        Returns
        -------
        scaled_boundaries : `dict`
            Dictionary containing scaled coordinate ranges and labels:
            coord1_min/max, coord2_min/max, coord1_name, coord2_name, units.
        true_boundaries : `dict`
            Dictionary containing true coordinate ranges and labels.
        """
    
    @abstractmethod
    def transform_coordinates(self, data_df):
        """Transform coordinates if needed (e.g., centering, scaling).

        Apply coordinate system specific transformations such as centering
        or scaling to prepare coordinates for gridding operations.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with original coord1, coord2 coordinate columns.

        Returns
        -------
        transformed_df : `pandas.DataFrame`
            DataFrame with additional coord1_scaled, coord2_scaled columns
            containing transformed coordinates ready for gridding.
        """

    def _create_shear_grid(self, data_df, idx1, idx2, npix1, npix2, accumulate_counts=False):
        """Create weighted shear grid from binning indices.

        Helper method that bins shear values into a regular grid using
        provided indices. Handles weighting and normalization of shear values.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame containing g1, g2 shear components and weights.
        idx1 : `numpy.ndarray`
            Bin indices for first dimension of the grid.
        idx2 : `numpy.ndarray`
            Bin indices for second dimension of the grid.
        npix1 : `int`
            Number of pixels in first dimension of output grid.
        npix2 : `int`
            Number of pixels in second dimension of output grid.

        Returns
        -------
        g1_grid : `numpy.ndarray`
            2D array of binned, weighted first shear component values.
        g2_grid : `numpy.ndarray`
            2D array of binned, weighted second shear component values.

        Notes
        -----
        The accumulated per-pixel weights used for normalization are stored
        on the coordinate system instance as ``_last_weight_grid`` for
        downstream consumers (e.g., KS+ mask construction). Pixels with
        non-positive or zero weight indicate gaps (no contributing data).
        If ``accumulate_counts`` is ``True``, the raw sample counts per pixel
        are accumulated and exposed as ``_last_count_grid``; otherwise, any
        stale ``_last_count_grid`` attribute is removed to prevent reuse.
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
        count_grid = None
        
        # Accumulate weighted values
        np.add.at(g1_grid, (idx2, idx1), g1 * weight)
        np.add.at(g2_grid, (idx2, idx1), g2 * weight)
        np.add.at(weight_grid, (idx2, idx1), weight)
        # Accumulate raw sample counts per pixel if requested
        if accumulate_counts:
            count_grid = np.zeros((npix1, npix2))
            np.add.at(count_grid, (idx2, idx1), 1)
        
        # Normalize by weights
        nonzero_mask = weight_grid != 0
        g1_grid[nonzero_mask] /= weight_grid[nonzero_mask]
        g2_grid[nonzero_mask] /= weight_grid[nonzero_mask]
        
        # Expose weight and optional count grids for downstream consumers
        self._last_weight_grid = weight_grid
        if accumulate_counts and count_grid is not None:
            self._last_count_grid = count_grid
        else:
            # Ensure stale counts are not reused across calls
            if hasattr(self, '_last_count_grid'):
                delattr(self, '_last_count_grid')
        
        return g1_grid, g2_grid

    def prepare_data(self, data_df):
        """Prepare data for gridding by validating and transforming coordinates.

        Check that required columns exist and ensure coordinates are transformed
        appropriately before gridding operations.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Input DataFrame containing coordinates and shear measurements.

        Returns
        -------
        processed_df : `pandas.DataFrame`
            Processed DataFrame with all required columns and transformed
            coordinates.

        Raises
        ------
        ValueError
            If required columns are missing from the input DataFrame.
        """
        required_cols = ['coord1', 'coord2', 'g1', 'g2', 'weight']
        if not all(col in data_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Transform coordinates if not already done
        if 'coord1_scaled' not in data_df.columns:
            data_df = self.transform_coordinates(data_df)
            
        return data_df