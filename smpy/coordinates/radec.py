import numpy as np
from .base import CoordinateSystem

class RADecSystem(CoordinateSystem):
    """
    Implementation for RA/Dec celestial coordinates.
    Handles coordinate transformations and gridding for celestial coordinates.
    """
    
    def get_grid_parameters(self, config):
        """
        Get RA/Dec specific grid parameters.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
            
        Returns
        -------
        dict
            Grid parameters including resolution in arcminutes
        """
        radec_config = config.get('radec', {})
        if 'resolution' not in radec_config:
            print("Warning: No resolution specified in radec config, using default 0.4 arcmin")
            
        return {
            'resolution_arcmin': radec_config.get('resolution', 0.4)
        }
    
    def create_grid(self, data_df, boundaries, config):
        """
        Create grid in RA/Dec space.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame with columns: coord1(RA), coord2(Dec), g1, g2, weight
        boundaries : dict
            Dictionary with scaled coordinate boundaries
        config : dict
            Configuration dictionary
            
        Returns
        -------
        tuple
            (g1_grid, g2_grid) numpy arrays
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
        
        return self._create_shear_grid(data_df, ra_idx, dec_idx, npix_dec, npix_ra)
    
    def calculate_boundaries(self, coord1, coord2):
        """
        Calculate field boundaries in RA/Dec space.
        Returns both true and scaled boundaries.
        
        Parameters
        ----------
        coord1 : array-like
            RA values in degrees
        coord2 : array-like
            Dec values in degrees
            
        Returns
        -------
        tuple
            (scaled_boundaries, true_boundaries) dictionaries
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
        """
        Transform RA/Dec coordinates by centering and flattening RA.
        
        Parameters
        ----------
        data_df : pd.DataFrame
            DataFrame with columns: coord1(RA), coord2(Dec)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with additional columns: coord1_scaled, coord2_scaled
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