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
    """Shear catalog binned onto a regular map grid.

    All arrays have shape ``(ny, nx)``: rows follow the map's y axis (north
    for RA/Dec) and columns its x axis (west for RA/Dec).

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
    mapping. Each coordinate system transforms a catalog into its gridding
    coordinates and places the map grid around it; binning onto the grid is
    shared.
    """

    @abstractmethod
    def calculate_boundaries(self, data_df, settings):
        """Calculate the map grid around a catalog.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog from `transform_coordinates`.
        settings : `dict`
            The coordinate system's section of ``general``, such as
            ``general.radec``.

        Returns
        -------
        scaled_boundaries : `dict`
            Map edges in the gridding coordinates (coord1_min/max,
            coord2_min/max), the map ``pixel_size`` in the same units, and
            labels: coord1_name, coord2_name, units.
        true_boundaries : `dict`
            Ranges of the catalog's own coordinates, with the same labels.
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

    def create_grid(self, data_df, boundaries):
        """Bin a catalog onto the map grid.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog with coord1_scaled and coord2_scaled from
            `transform_coordinates`, g1, g2 and weight.
        boundaries : `dict`
            Map edges and ``pixel_size`` from `calculate_boundaries`.

        Returns
        -------
        grid : `ShearGrid`
            Binned shear, weights and counts.
        """
        data_df = self.prepare_data(data_df)
        size = boundaries['pixel_size']
        indices = []
        for axis in ('coord1', 'coord2'):
            npix = round((boundaries[f'{axis}_max'] - boundaries[f'{axis}_min']) / size)
            index = np.floor((data_df[f'{axis}_scaled'].to_numpy() - boundaries[f'{axis}_min']) / size)
            # Rounding can put a galaxy on the outer edge just outside the map
            indices.append((np.clip(index, 0, npix - 1).astype(int), npix))
        (idx1, npix1), (idx2, npix2) = indices
        return self._create_shear_grid(data_df, idx1, idx2, npix2, npix1)

    @staticmethod
    def _grid_edges(data_df, pixel_size):
        """Return the edges of a map of whole pixels centered on a catalog.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog with coord1_scaled and coord2_scaled.
        pixel_size : `float`
            Pixel width in the same units.

        Returns
        -------
        boundaries : `dict`
            coord1_min/max, coord2_min/max and ``pixel_size``. Every galaxy
            lies inside the map.
        """
        boundaries = {'pixel_size': pixel_size}
        for axis in ('coord1', 'coord2'):
            values = data_df[f'{axis}_scaled']
            npix = int(np.floor((values.max() - values.min()) / pixel_size)) + 1
            middle = (values.max() + values.min()) / 2
            boundaries[f'{axis}_min'] = middle - npix * pixel_size / 2
            boundaries[f'{axis}_max'] = middle + npix * pixel_size / 2
        return boundaries

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
        g1 = data_df['g1'].values
        g2 = data_df['g2'].values
        weight = data_df['weight'].values

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
