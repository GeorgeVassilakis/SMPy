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

    def create_grid(self, data_df, boundaries, settings):
        """Create shear grid in pixel space.

        Create a regular grid by binning shear data. Includes automatic
        adjustment of grid size if dimensions exceed max_grid_size safety
        limits.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with scaled coordinates and shear data.
        boundaries : `dict`
            Dictionary with coordinate boundaries.
        settings : `dict`
            The ``general.pixel`` settings, including ``downsample_factor``.

        Returns
        -------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear, weights and counts.
        """
        # Ensure data is properly prepared
        data_df = self.prepare_data(data_df)

        downsample = settings['downsample_factor']
        max_size = MAX_GRID_SIZE

        # Get boundaries
        x_min, x_max = boundaries['coord1_min'], boundaries['coord1_max']
        y_min, y_max = boundaries['coord2_min'], boundaries['coord2_max']

        # Calculate raw dimensions
        raw_npix_x = int(np.ceil(x_max - x_min))
        raw_npix_y = int(np.ceil(y_max - y_min))

        # Calculate downsampled dimensions
        npix_x = int(np.ceil(raw_npix_x / downsample))
        npix_y = int(np.ceil(raw_npix_y / downsample))

        # Safety check for grid size
        if npix_x > max_size or npix_y > max_size:
            min_downsample = max(raw_npix_x, raw_npix_y) / max_size
            downsample = max(downsample, min_downsample)
            warnings.warn(f"A {npix_x}x{npix_y} grid exceeds {max_size} pixels; downsample_factor "
                          f"raised to {downsample:.1f}")
            npix_x = int(np.ceil(raw_npix_x / downsample))
            npix_y = int(np.ceil(raw_npix_y / downsample))

        # Create bins for scaled pixels
        x_bins = np.linspace(x_min, x_max, npix_x + 1)
        y_bins = np.linspace(y_min, y_max, npix_y + 1)

        # Digitize x and y coordinates
        x_idx = np.digitize(data_df['coord1_scaled'], x_bins) - 1
        y_idx = np.digitize(data_df['coord2_scaled'], y_bins) - 1

        return self._create_shear_grid(data_df, x_idx, y_idx, npix_y, npix_x)

    def calculate_boundaries(self, coord1, coord2):
        """Calculate field boundaries in pixel space.

        Determine coordinate ranges for pixel coordinates and set up
        appropriate labels. Provides warnings for unusually large coordinate
        ranges.

        Parameters
        ----------
        coord1 : `numpy.ndarray`
            X pixel coordinates.
        coord2 : `numpy.ndarray`
            Y pixel coordinates.

        Returns
        -------
        scaled_boundaries : `dict`
            Dictionary with coordinate boundaries and labels.
        true_boundaries : `dict`
            Identical to scaled_boundaries since no scaling needed in pixel
            space.

        Notes
        -----
        Warns if coordinate ranges are unusually large (>1e5 pixels).
        """
        # For pixel coordinates, we want integers as boundaries
        boundaries = {
            'coord1_min': float(np.floor(np.min(coord1))),
            'coord1_max': float(np.ceil(np.max(coord1))),
            'coord2_min': float(np.floor(np.min(coord2))),
            'coord2_max': float(np.ceil(np.max(coord2))),
            'coord1_name': 'X',
            'coord2_name': 'Y',
            'units': 'pixels'
        }

        # Check for potentially problematic coordinate ranges
        if np.ptp(coord1) > 1e5 or np.ptp(coord2) > 1e5:
            warnings.warn(f"Large pixel coordinate range: x {boundaries['coord1_min']} to "
                          f"{boundaries['coord1_max']}, y {boundaries['coord2_min']} to "
                          f"{boundaries['coord2_max']}")

        # For pixel coordinates, scaled and true boundaries are the same
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
