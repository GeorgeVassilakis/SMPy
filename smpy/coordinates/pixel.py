"""Pixel coordinate system implementation.

This module provides the PixelSystem class for handling data in pixel
coordinates, including downsampling for grid creation and safety checks
for large pixel dimensions.
"""

import warnings

import numpy as np

from .base import CoordinateSystem

MAX_GRID_SIZE = 10000
"""Largest grid dimension before the downsampling is increased (`int`)."""

class PixelSystem(CoordinateSystem):
    """Implementation for pixel coordinates.

    Handles data in pixel coordinates, including downsampling for grid
    creation and safety checks for large pixel dimensions. Provides identity
    coordinate transformations since no scaling is needed for pixel data.
    """

    def calculate_boundaries(self, data_df, settings):
        """Calculate the map grid around a catalog in pixel space.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog from `transform_coordinates`.
        settings : `dict`
            The ``general.pixel`` settings, including ``downsample_factor``.

        Returns
        -------
        scaled_boundaries : `dict`
            Map edges in catalog pixels and ``pixel_size``, the number of
            catalog pixels per map pixel.
        true_boundaries : `dict`
            The same as ``scaled_boundaries``, since pixel coordinates are not
            transformed.

        Notes
        -----
        Warns if the coordinate range is unusually large (more than 1e5
        pixels), and raises the downsampling, with a warning, if the map would
        exceed `MAX_GRID_SIZE` pixels on a side.
        """
        x, y = data_df['coord1_scaled'], data_df['coord2_scaled']
        if np.ptp(x) > 1e5 or np.ptp(y) > 1e5:
            warnings.warn(f"Large pixel coordinate range: x {x.min()} to {x.max()}, y {y.min()} to {y.max()}")

        downsample = settings['downsample_factor']
        span = max(np.ptp(x), np.ptp(y))
        if span / downsample + 1 > MAX_GRID_SIZE:
            downsample = span / (MAX_GRID_SIZE - 1)
            warnings.warn(f"A map wider than {MAX_GRID_SIZE} pixels; downsample_factor raised to "
                          f"{downsample:.1f}")

        boundaries = self._grid_edges(data_df, downsample)
        boundaries.update(coord1_name='X', coord2_name='Y', units='pixels')
        return boundaries, boundaries

    def transform_coordinates(self, data_df):
        """Transform pixel coordinates (identity transform).

        Apply identity transformation to pixel coordinates since no scaling
        or centering is needed for pixel coordinate systems.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with coord1(X), coord2(Y) columns.

        Returns
        -------
        transformed_df : `pandas.DataFrame`
            DataFrame with coord1_scaled, coord2_scaled identical to input
            coordinates.
        """
        transformed_df = data_df.copy()

        # For pixel coordinates, scaled coordinates are the same as original
        transformed_df['coord1_scaled'] = data_df['coord1'].astype(float)
        transformed_df['coord2_scaled'] = data_df['coord2'].astype(float)

        return transformed_df
