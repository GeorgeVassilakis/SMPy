import numpy as np
import pandas as pd
import random

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

def load_shear_data(shear_cat_path, ra_col, dec_col, g1_col, g2_col, weight_col):
    """ 
    Load shear data from a FITS file and return a pandas DataFrame.

    :param path: Path to the FITS file.
    :param ra_col: Column name for right ascension.
    :param dec_col: Column name for declination.
    :param g1_col: Column name for the first shear component.
    :param g2_col: Column name for the second shear component.
    :param weight_col: Column name for the weight.
    :return: pandas DataFrame with the specified columns.
    """
    # Read data from the FITS file
    shear_catalog = Table.read(shear_cat_path)

    # Convert to pandas DataFrame
    shear_df = pd.DataFrame({
        'ra': shear_catalog[ra_col],
        'dec': shear_catalog[dec_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
    })

    # If weight_col is none, set all weights to 1, if it's passed, add a column to shear df of 'weight': shear_catalog[weight_col]
    # Make the weight column a column of ones of length = ra_col
    if weight_col is None:
        shear_df['weight'] = np.ones(len(shear_df))
    else:
        shear_df['weight'] = shear_catalog[weight_col]
        
    return shear_df

def scale_ra_dec(shear_df):
    """
    Correct the RA and Dec coordinates by centering and flattening RA.

    Parameters
    ----------
    shear_df : pd.DataFrame
        DataFrame containing 'ra' and 'dec' columns representing
        right ascension and declination in degrees.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with corrected 'ra' and 'dec' columns.
    """
    # Create a copy to avoid modifying the original DataFrame
    corrected_df = shear_df.copy()
    
    # Extract RA and Dec columns
    ra = corrected_df['ra']
    dec = corrected_df['dec']
    
    # Compute the central RA and Dec
    ra_0 = (ra.max() + ra.min()) / 2  # Center of RA
    dec_0 = (dec.max() + dec.min()) / 2  # Center of Dec
    
    # Apply the transformation using vectorized operations
    corrected_df['ra_scaled'] = (ra - ra_0) * np.cos(np.deg2rad(dec))
    corrected_df['dec_scaled'] = dec - dec_0
    
    return corrected_df

def calculate_field_boundaries(ra, dec):
    """
    Calculate the boundaries of the field in right ascension (RA) and declination (Dec).
    
    :param ra: Dataframe column containing the right ascension values.
    :param dec: Dataframe column containing the declination values.
    :return: A dictionary containing the corners of the map {'ra_min', 'ra_max', 'dec_min', 'dec_max'}.
    """
    boundaries = {
        'ra_min': np.min(ra),
        'ra_max': np.max(ra),
        'dec_min': np.min(dec),
        'dec_max': np.max(dec)
    }
    
    return boundaries

def create_shear_grid(ra, dec, g1, g2, weight, boundaries, resolution):
    '''
    Bin values of shear data according to position on the sky.
    '''
    ra_min, ra_max = boundaries['ra_min'], boundaries['ra_max']
    dec_min, dec_max = boundaries['dec_min'], boundaries['dec_max']
    
    # Calculate number of pixels based on field size and resolution
    npix_ra = int(np.ceil((ra_max - ra_min) * 60 / resolution))
    npix_dec = int(np.ceil((dec_max - dec_min) * 60 / resolution))
    
    ra_bins = np.linspace(ra_min, ra_max, npix_ra + 1)
    dec_bins = np.linspace(dec_min, dec_max, npix_dec + 1)
    
    # Digitize the RA and Dec to find bin indices
    ra_idx = np.digitize(ra, ra_bins) - 1
    dec_idx = np.digitize(dec, dec_bins) - 1
    
    # Filter out indices that are outside the grid boundaries
    valid_mask = (ra_idx >= 0) & (ra_idx < npix_ra) & (dec_idx >= 0) & (dec_idx < npix_dec)
    ra_idx = ra_idx[valid_mask]
    dec_idx = dec_idx[valid_mask]
    g1 = g1[valid_mask]
    g2 = g2[valid_mask]
    weight = weight[valid_mask]
    
    # Initialize the grids
    g1_grid = np.zeros((npix_dec, npix_ra))
    g2_grid = np.zeros((npix_dec, npix_ra))
    weight_grid = np.zeros((npix_dec, npix_ra))
    
    # Accumulate weighted values using np.add.at
    np.add.at(g1_grid, (dec_idx, ra_idx), g1 * weight)
    np.add.at(g2_grid, (dec_idx, ra_idx), g2 * weight)
    np.add.at(weight_grid, (dec_idx, ra_idx), weight)
    
    # Normalize the grid by the total weight in each bin (weighted average)
    #try with commented out 
    nonzero_weight_mask = weight_grid != 0
    g1_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    g2_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    
    return g1_grid, g2_grid

def save_convergence_fits(convergence, boundaries, config, output_name):
    """
    Save the convergence map as a FITS file with WCS information if configured to do so.

    Parameters:
    -----------
    convergence : numpy.ndarray
        The 2D convergence map.
    boundaries : dict
        Dictionary containing 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    config : dict
        Configuration dictionary containing output path and other settings.

    Returns:
    --------
    None
    """
    if not config.get('save_fits', False):
        return

    # Create a WCS object
    wcs = WCS(naxis=2)
    
    # Set up the WCS parameters
    npix_dec, npix_ra = convergence.shape
    wcs.wcs.crpix = [npix_ra / 2, npix_dec / 2]
    wcs.wcs.cdelt = [(boundaries['ra_max'] - boundaries['ra_min']) / npix_ra, 
                     (boundaries['dec_max'] - boundaries['dec_min']) / npix_dec]
    wcs.wcs.crval = [(boundaries['ra_max'] + boundaries['ra_min']) / 2, 
                     (boundaries['dec_max'] + boundaries['dec_min']) / 2]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Create a FITS header from the WCS information
    header = wcs.to_header()

    # Add some additional information to the header
    header['AUTHOR'] = 'SMPy'
    header['CONTENT'] = 'Convergence Map'

    # Create a primary HDU containing the convergence map
    hdu = fits.PrimaryHDU(convergence, header=header)

    # Create a FITS file
    hdul = fits.HDUList([hdu])

    # Save the FITS file
    hdul.writeto(output_name, overwrite=True)

    print(f"Convergence map saved as FITS file: {output_name}")


def _shuffle_ra_dec(shear_df):
    """
    Shuffle the 'ra' and 'dec' columns of the input DataFrame together.
    
    :param shear_df: Input pandas DataFrame.
    :return: A new pandas DataFrame with shuffled 'ra' and 'dec' columns.
    """
    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()

    # Combine RA and DEC into pairs
    ra_dec_pairs = list(zip(shuffled_df['ra'], shuffled_df['dec']))
    
    # Shuffle the pairs
    random.shuffle(ra_dec_pairs)
    
    # Unzip the shuffled pairs back into RA and DEC
    shuffled_ra, shuffled_dec = zip(*ra_dec_pairs)
    
    shuffled_df['ra'] = shuffled_ra
    shuffled_df['dec'] = shuffled_dec

    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, seed=42):
    """
    Generate a list of multiple data frames with shuffled RA and DEC columns by calling the load and shuffle functions.
    :return: A list of shuffled pandas DataFrames.
    """

    # List to store the shuffled data frames (not sure if a list of these data frames is the best format rn)
    shuffled_dfs = []
    
    #set a seed for reproducibility
    random.seed(seed)
        
    # Loop to generate multiple shuffled data frames
    for i in range(num_shuffles):
        shuffled_df = _shuffle_ra_dec(og_shear_df)
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs

def shear_grids_for_shuffled_dfs(list_of_dfs, boundaries, config): 
    grid_list = []
    for shear_df in list_of_dfs: 
        g1map, g2map = create_shear_grid(shear_df['ra'], 
                                           shear_df['dec'], 
                                           shear_df['g1'],
                                           shear_df['g2'], 
                                           shear_df['weight'], 
                                           boundaries=boundaries,
                                           resolution=config['resolution'])

        grid_list.append((g1map, g2map))

    return grid_list
