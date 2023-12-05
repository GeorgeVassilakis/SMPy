import numpy as np
import pandas as pd
from astropy.table import Table

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
        'weight': shear_catalog[weight_col]
    })

    return shear_df

def calculate_field_boundaries(ra, dec, resolution, width):
    """
    Calculate the boundaries of the field in right ascension (RA) and declination (Dec).

    :param ra: Dataframe column containing the right ascension values.
    :param dec: Dataframe column containing the declination values.
    :param resolution: Resolution of the map in arcminutes.
    :param width: Width of the map in pixels.
    :return: A list containing the corners of the map [ra_min, ra_max, dec_min, dec_max].
    """
    # Calculate median RA and Dec
    med_ra = np.median(ra)
    med_dec = np.median(dec)

    # Calculate size of the field in degrees
    size = width * resolution / 60.  # Convert from arcminutes to degrees

    # Calculate RA and Dec extents and store in a dictionary
    boundaries = {
        'ra_min': med_ra - size/2,
        'ra_max': med_ra + size/2,
        'dec_min': med_dec - size/2,
        'dec_max': med_dec + size/2
    }
    return boundaries

def create_shear_grid(ra, dec, g1, g2, weight, boundaries, npix):
    '''
    Bin values of shear data according to position on the sky.

    Parameters
    ----------
    ra : array_like
        Array of right ascension values.
    dec : array_like
        Array of declination values.
    g1 : array_like
        Array of g1 values (shear component 1).
    g2 : array_like
        Array of g2 values (shear component 2).
    weight : array_like
        Array of weight values for each shear measurement.
    boundaries : dict
        Boundaries of the field of view, given as a dictionary
        with keys 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    npix : int
        Number of pixels (bins) along each axis.

    Returns
    -------
    g1_grid, g2_grid : ndarray
        2D numpy arrays of binned g1 and g2 values.
    '''
    
    # Extract the boundariess from the dictionary
    ra_min, ra_max = boundaries['ra_min'], boundaries['ra_max']
    dec_min, dec_max = boundaries['dec_min'], boundaries['dec_max']

    # Create 2D histogram bins for RA and Dec
    ra_bins = np.linspace(ra_min, ra_max, npix)
    dec_bins = np.linspace(dec_min, dec_max, npix)

    # Initialize the grid
    g1_grid = np.zeros((npix, npix))
    g2_grid = np.zeros((npix, npix))
    weight_grid = np.zeros((npix, npix))

    # Digitize the RA and Dec to find bin indices
    ra_idx = np.digitize(ra, ra_bins) - 1
    dec_idx = np.digitize(dec, dec_bins) - 1

    # Iterate over each point and accumulate weighted g1, g2 values
    for i in range(len(ra)):
        if 0 <= ra_idx[i] < npix and 0 <= dec_idx[i] < npix:
            g1_grid[dec_idx[i], ra_idx[i]] += g1[i] * weight[i]
            g2_grid[dec_idx[i], ra_idx[i]] += g2[i] * weight[i]
            weight_grid[dec_idx[i], ra_idx[i]] += weight[i]

    # Normalize the grid by the total weight in each bin
    nonzero_weight_mask = weight_grid != 0
    g1_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    g2_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]

    return g1_grid, g2_grid

