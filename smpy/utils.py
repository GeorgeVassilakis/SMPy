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

def _shuffle_ra_dec(shear_df, seed=None):
    """
    Shuffle the scaled 'ra' and 'dec' columns of the input DataFrame together.
    
    :param shear_df: Input pandas DataFrame.
    :param seed: Random seed for reproducibility.
    :return: A new pandas DataFrame with shuffled 'ra_scaled' and 'dec_scaled' columns.
    """
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        
    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()

    # Combine scaled RA and DEC into pairs
    ra_dec_pairs = list(zip(shuffled_df['ra_scaled'], shuffled_df['dec_scaled']))
    
    # Shuffle the pairs
    random.shuffle(ra_dec_pairs)
    
    # Unzip the shuffled pairs back into RA and DEC
    shuffled_ra, shuffled_dec = zip(*ra_dec_pairs)
    
    # Update the scaled coordinates
    shuffled_df['ra_scaled'] = shuffled_ra
    shuffled_df['dec_scaled'] = shuffled_dec

    return shuffled_df

def _shuffle_galaxy_rotation(shear_df):
    """The function will shuffle the galaxy rotation in the input shear_df DataFrame.

    Args:
        shear_df (_type_): _description_
    """
    
    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()
    
    # Shuffle the galaxy rotation
    g1, g2 = shuffled_df['g1'], shuffled_df['g2']
    
    # Add a random angle to the galaxy rotation
    angle = np.random.uniform(0, 2 * np.pi, len(g1))
    g1g2_len = np.sqrt(np.array(g1)**2 + np.array(g2)**2)
    g1g2_angle  = np.arctan2(g2, g1) + angle
    g1_new = g1g2_len * np.cos(g1g2_angle)
    g2_new = g1g2_len * np.sin(g1g2_angle)
    
    shuffled_df['g1'] = g1_new
    shuffled_df['g2'] = g2_new
    
    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, shuffle_type='position', seed=0):
    """
    Generate a list of multiple data frames with shuffled RA and DEC columns.
    :param og_shear_df: Original shear DataFrame to shuffle
    :param num_shuffles: Number of shuffled copies to generate
    :param seed: Random starting seed for reproducibility
    :return: A list of shuffled pandas DataFrames
    """
    # List to store the shuffled data frames
    shuffled_dfs = []
    
    # Loop to generate multiple shuffled data frames
    for i in range(num_shuffles):
        # Shuffle based on the specified type
        if shuffle_type == 'spatial':
            shuffled_df = _shuffle_ra_dec(og_shear_df, seed=seed+i)
        elif shuffle_type == 'orientation':
            shuffled_df = _shuffle_galaxy_rotation(og_shear_df)
        else:
            raise ValueError(f"Invalid shuffle type: {shuffle_type}")
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs

def g1g2_to_gt_gc(g1, g2, ra, dec, ra_c, dec_c, pix_ra = 100):
    """
    Convert reduced shear to tangential and cross shear (Eq. 10, 11 in McCleary et al. 2023).
    args:
    - g1, g2: Reduced shear components.
    - ra, dec: Right ascension and declination of the catalogue,i.e. shear_df['ra'], shear_df['dec'].
    - ra_c, dec_c: Right ascension and declination of the cluster-centre.
    
    returns:
    - gt, gc: Tangential and cross shear components.
    - phi: Polar angle in the plane of the sky.
    """ 
    ra_max, ra_min, dec_max, dec_min = np.max(ra), np.min(ra), np.max(dec), np.min(dec)
    aspect_ratio = (ra_max - ra_min) / (dec_max - dec_min)
    pix_dec = int(pix_ra / aspect_ratio)
    ra_grid, dec_grid = np.meshgrid(np.linspace(ra_min, ra_max, pix_ra), np.linspace(dec_min, dec_max, pix_dec))

    phi = np.arctan2(dec_grid - dec_c, ra_grid - ra_c)
    
    # Calculate the tangential and cross components
    gt = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)
    gc = -g1 * np.sin(2 * phi) + g2 * np.cos(2 * phi)

    return gt, gc, phi

def find_peaks2d(image, threshold=None):
    """
    Identify peaks in a 2D array (image) above a specified threshold.
    A peak is a pixel with a value greater than its 8 neighbors.

    Minimized version of function from Lenspack.

    Parameters:
    - image (np.ndarray): 2D array representing the image.
    - threshold (float, optional): Minimum pixel value to consider as a peak. Defaults to the minimum of `image`.

    Returns:
    - X, Y, heights (tuple): Indices of peaks (X, Y) and their corresponding heights.
    """
    image = np.atleast_2d(image)

    # Set threshold to the minimum value in the image if none provided
    threshold = threshold if threshold is not None else image.min()

    # Pad the image to simplify border peak checks
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=image.min())

    # Check for peaks by comparing to neighbors
    is_peak = (
        (padded_image[1:-1, 1:-1] > padded_image[:-2, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[:-2, 1:-1]) &
        (padded_image[1:-1, 1:-1] > padded_image[:-2, 2:]) &
        (padded_image[1:-1, 1:-1] > padded_image[1:-1, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[1:-1, 2:]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, 1:-1]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, 2:])
    )

    # Apply threshold
    peaks_mask = is_peak & (image >= threshold)

    # Get peak coordinates and their heights
    Y, X = np.nonzero(peaks_mask)
    heights = image[Y, X]

    return X, Y, heights
