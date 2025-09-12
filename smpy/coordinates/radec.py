"""RA/Dec celestial coordinate system implementation.

This module provides the RADecSystem class for handling celestial coordinates
including spherical corrections and appropriate binning for sky projections.
"""

import numpy as np
from .base import CoordinateSystem

class RADecSystem(CoordinateSystem):
    """Implementation for RA/Dec celestial coordinates.

    Handles celestial coordinates including spherical corrections and
    appropriate binning for sky projections. Applies coordinate transformations
    to account for spherical geometry effects.
    """

    def get_grid_parameters(self, config):
        """Get RA/Dec specific grid parameters.

        Extract celestial coordinate system parameters including angular
        resolution for grid spacing.

        Parameters
        ----------
        config : `dict`
            Configuration dictionary containing RA/Dec settings.

        Returns
        -------
        grid_params : `dict`
            Grid parameters dictionary containing:
            - resolution_arcmin: Grid spacing in arcminutes
        """
        radec_config = config['general']['radec']
        if 'resolution' not in radec_config:
            print("Warning: No resolution specified in radec config, using default 0.4 arcmin")
            
        return {
            'resolution_arcmin': radec_config.get('resolution', 0.4)
        }

    def create_grid(self, data_df, boundaries, config):
        """Create shear grid in RA/Dec space.

        Create a regular grid in celestial coordinates accounting for
        spherical projection effects and appropriate angular binning.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with scaled coordinates and shear data.
        boundaries : `dict`
            Dictionary with coordinate boundaries.
        config : `dict`
            Configuration dictionary with RA/Dec parameters.

        Returns
        -------
        g1_grid : `numpy.ndarray`
            2D array containing binned first shear component values.
        g2_grid : `numpy.ndarray`
            2D array containing binned second shear component values.
        """
        # Ensure data is properly prepared
        data_df = self.prepare_data(data_df)
        
        # Get grid parameters
        grid_params = self.get_grid_parameters(config)
        resolution_arcmin = grid_params['resolution_arcmin']
        
        # Get boundaries
        ra_min, ra_max = boundaries['coord1_min'], boundaries['coord1_max']
        dec_min, dec_max = boundaries['coord2_min'], boundaries['coord2_max']
        
        # Calculate number of pixels based on field size and resolution (arcmin)
        npix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution_arcmin))
        npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution_arcmin))
        
        # Create bins
        ra_bins = np.linspace(ra_min, ra_max, npix_ra + 1)
        dec_bins = np.linspace(dec_min, dec_max, npix_dec + 1)
        
        # Digitize the RA and Dec to find bin indices
        ra_idx = np.digitize(data_df['coord1_scaled'], ra_bins) - 1
        dec_idx = np.digitize(data_df['coord2_scaled'], dec_bins) - 1
        
        # Determine whether to accumulate counts based on config
        accumulate_counts = bool(
            config.get('general', {}).get('create_counts_map', False)
            or config.get('general', {}).get('overlay_counts_map', False)
        )
        return self._create_shear_grid(
            data_df,
            ra_idx,
            dec_idx,
            npix_dec,
            npix_ra,
            accumulate_counts=accumulate_counts,
        )

    def calculate_boundaries(self, coord1, coord2):
        """Calculate field boundaries in RA/Dec space.

        Determine coordinate ranges for celestial coordinates and set up
        both scaled boundaries (accounting for spherical projection) and
        true boundaries (preserving original coordinates).

        Parameters
        ----------
        coord1 : `numpy.ndarray`
            RA values in degrees.
        coord2 : `numpy.ndarray`
            Dec values in degrees.

        Returns
        -------
        scaled_boundaries : `dict`
            Dictionary with scaled coordinate boundaries accounting for
            spherical projection effects.
        true_boundaries : `dict`
            Dictionary with original celestial coordinate boundaries.
        """
        # True boundaries (original coordinates)
        true_boundaries = {
            'coord1_min': np.min(coord1),
            'coord1_max': np.max(coord1),
            'coord2_min': np.min(coord2),
            'coord2_max': np.max(coord2),
            'coord1_name': 'RA',
            'coord2_name': 'Dec',
            'units': 'deg'
        }
        
        # Calculate central coordinates for scaling
        coord1_0 = (true_boundaries['coord1_max'] + true_boundaries['coord1_min']) / 2
        coord2_0 = (true_boundaries['coord2_max'] + true_boundaries['coord2_min']) / 2
        
        # Calculate scaled coordinates (flatten RA by cos(Dec))
        scaled_coords1 = (coord1 - coord1_0) * np.cos(np.deg2rad(coord2))
        scaled_coords2 = coord2 - coord2_0
        
        scaled_boundaries = {
            'coord1_min': np.min(scaled_coords1),
            'coord1_max': np.max(scaled_coords1),
            'coord2_min': np.min(scaled_coords2),
            'coord2_max': np.max(scaled_coords2),
            'coord1_name': 'Scaled RA',
            'coord2_name': 'Scaled Dec',
            'units': 'deg'
        }
        
        return scaled_boundaries, true_boundaries

    def transform_coordinates(self, data_df):
        """Transform RA/Dec coordinates by centering and flattening RA.

        Apply coordinate transformations to account for spherical geometry
        by centering coordinates and flattening RA using cos(Dec) correction.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with coord1(RA), coord2(Dec) columns.

        Returns
        -------
        transformed_df : `pandas.DataFrame`
            DataFrame with additional scaled coordinates corrected for
            spherical projection effects.
        """
        transformed_df = data_df.copy()
        
        # Compute central coordinates
        ra_0 = (data_df['coord1'].max() + data_df['coord1'].min()) / 2
        dec_0 = (data_df['coord2'].max() + data_df['coord2'].min()) / 2
        
        # Apply the transformation
        # Flatten RA by cos(Dec) and center both coordinates
        transformed_df['coord1_scaled'] = (data_df['coord1'] - ra_0) * np.cos(np.deg2rad(data_df['coord2']))
        transformed_df['coord2_scaled'] = data_df['coord2'] - dec_0
        
        return transformed_df
