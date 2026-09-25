"""Base class for coordinate systems.

This module defines the abstract base class for coordinate systems used
in mass mapping operations, providing a consistent interface for RA/Dec
and pixel coordinate transformations and gridding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

@dataclass
class ShearGrid:
    """Shear catalogue binned onto a regular map grid.

    All arrays have shape ``(ny, nx)``: rows follow the second coordinate
    (Dec or y) and columns the first (RA or x).

    Parameters
    ----------
    g1 : `numpy.ndarray`
        Weighted mean first shear component per pixel; 0 in empty pixels.
    g2 : `numpy.ndarray`
        Weighted mean second shear component per pixel; 0 in empty pixels.
    weights : `numpy.ndarray`
        Summed galaxy weights per pixel. A weight of 0 marks an empty pixel.
    counts : `numpy.ndarray`
        Number of galaxies per pixel.
    """

    g1: np.ndarray
    g2: np.ndarray
    weights: np.ndarray
    counts: np.ndarray

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
    def create_grid(self, data_df, boundaries, settings):
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
        settings : `dict`
            The coordinate system's section of ``general``, such as
            ``general.radec``.

        Returns
        -------
        grid : `ShearGrid`
            Binned shear, weights and counts.
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

    def _create_shear_grid(self, data_df, idx1, idx2, npix1, npix2):
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
        grid : `ShearGrid`
            Weighted mean shear, summed weights and galaxy counts per pixel.
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
        count_grid = np.zeros((npix1, npix2))

        # Accumulate weighted values
        np.add.at(g1_grid, (idx2, idx1), g1 * weight)
        np.add.at(g2_grid, (idx2, idx1), g2 * weight)
        np.add.at(weight_grid, (idx2, idx1), weight)
        np.add.at(count_grid, (idx2, idx1), 1)

        # Normalize by weights
        nonzero_mask = weight_grid != 0
        g1_grid[nonzero_mask] /= weight_grid[nonzero_mask]
        g2_grid[nonzero_mask] /= weight_grid[nonzero_mask]

        return ShearGrid(g1_grid, g2_grid, weight_grid, count_grid)

    def prepare_data(self, data_df):
        """Prepare for gridding by validating and transforming coordinates.

        Check that required columns exist and ensure coordinates are
        transformed appropriately before gridding operations.

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
