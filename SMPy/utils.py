import numpy as np
import pandas as pd
import random

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

def load_shear_data(shear_cat_path, x_col, y_col, g1_col, g2_col, weight_col):
    """ 
    Load shear data from a FITS file and return a pandas DataFrame.

    :param path: Path to the FITS file.
    :param x_col: Column name for the x pixel coordinate.
    :param y_col: Column name for y pixel coordinate.
    :param g1_col: Column name for the first shear component.
    :param g2_col: Column name for the second shear component.
    :param weight_col: Column name for the weight.
    :return: pandas DataFrame with the specified columns.
    """
    # Read data from the FITS file
    shear_catalog = Table.read(shear_cat_path)

    # Convert to pandas DataFrame
    shear_df = pd.DataFrame({
        'x_col': shear_catalog[x_col],
        'y_col': shear_catalog[y_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
        'weight': shear_catalog[weight_col]
    })

    return shear_df

def calculate_field_boundaries(x_col, y_col):
    """
    Calculate the boundaries of the field in X/Y Pixel Space.
    
    :param x_col: Dataframe column containing the x pixel values.
    :param y_col: Dataframe column containing the y pixel values.
    :param resolution: Resolution of the map in arcminutes.
    :return: A dictionary containing the corners of the map {'x_min', 'x_max', 'y_min', 'y_max'}.
    """
    boundaries = {
        'x_min': np.min(x_col),
        'x_max': np.max(x_col),
        'y_min': np.min(y_col),
        'y_max': np.max(y_col)
    }
    
    return boundaries

def create_shear_grid(x_col, y_col, g1, g2, weight, boundaries, scaling_factor):
    '''
    Bin values of shear data according to position on the pixel grid, using a scaling factor to reduce the output size.
    '''
    x_min, x_max = boundaries['x_min'], boundaries['x_max']
    y_min, y_max = boundaries['y_min'], boundaries['y_max']
    
    # Calculate number of pixels in the output grid based on the scaling factor
    npix_x = int(np.ceil((x_max - x_min) / scaling_factor))
    npix_y = int(np.ceil((y_max - y_min) / scaling_factor))
    
    # Create bins for x and y
    x_bins = np.linspace(x_min, x_max, npix_x)
    y_bins = np.linspace(y_min, y_max, npix_y)
    
    # Digitize the x and y to find bin indices
    x_idx = np.digitize(x_col, x_bins) - 1
    y_idx = np.digitize(y_col, y_bins) - 1
    
    # Filter out indices that are outside the grid boundaries
    valid_mask = (x_idx >= 0) & (x_idx < npix_x) & (y_idx >= 0) & (y_idx < npix_y)
    x_idx = x_idx[valid_mask]
    y_idx = y_idx[valid_mask]
    g1 = g1[valid_mask]
    g2 = g2[valid_mask]
    weight = weight[valid_mask]
    
    # Initialize the grids
    g1_grid = np.zeros((npix_y, npix_x))
    g2_grid = np.zeros((npix_y, npix_x))
    weight_grid = np.zeros((npix_y, npix_x))
    
    # Accumulate weighted values using np.add.at
    np.add.at(g1_grid, (y_idx, x_idx), g1 * weight)
    np.add.at(g2_grid, (y_idx, x_idx), g2 * weight)
    np.add.at(weight_grid, (y_idx, x_idx), weight)
    
    # Normalize the grid by the total weight in each bin (weighted average)
    nonzero_weight_mask = weight_grid != 0
    g1_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    g2_grid[nonzero_weight_mask] /= weight_grid[nonzero_weight_mask]
    
    print(f"Size of g1_grid: {g1_grid.shape}")
    print(f"Size of g2_grid: {g2_grid.shape}")
    
    return g1_grid, g2_grid



def save_convergence_fits(convergence, boundaries, config, output_name):
    """
    Save the convergence map as a FITS file with WCS information if configured to do so.

    Parameters:
    -----------
    convergence : numpy.ndarray
        The 2D convergence map.
    boundaries : dict
        Dictionary containing 'x_min', 'x_max', 'y_min', 'y_max'.
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
    npix_y, npix_x = convergence.shape
    wcs.wcs.crpix = [npix_x / 2, npix_y / 2]
    wcs.wcs.cdelt = [(boundaries['x_max'] - boundaries['x_min']) / npix_x, 
                     (boundaries['y_max'] - boundaries['y_min']) / npix_y]
    wcs.wcs.crval = [(boundaries['x_max'] + boundaries['x_min']) / 2, 
                     (boundaries['y_max'] + boundaries['y_min']) / 2]
    wcs.wcs.ctype = ["X", "Y"]

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

### Shuffling Functions ###

def _shuffle_xy(shear_df, seed=None):
    """
    Shuffle the 'x_col' and 'y_col' columns of the input DataFrame together.
    
    :param shear_df: Input pandas DataFrame.
    :param seed: Optional seed for reproducibility.
    :return: A new pandas DataFrame with shuffled 'x_col' and 'y_col' columns.
    """
    # Set the random seed for reproducibility if provided
    if seed is None:
        random.seed(42)
    else:
        random.seed(seed)

    # Make a copy to avoid modifying the original
    shuffled_df = shear_df.copy()

    # Combine x and y into pairs
    xy_pairs = list(zip(shuffled_df['x_col'], shuffled_df['y_col']))
    
    # Shuffle the pairs
    random.shuffle(xy_pairs)
    
    # Unzip the shuffled pairs back into x and y
    shuffled_x, shuffled_y = zip(*xy_pairs)
    
    shuffled_df['x_col'] = shuffled_x
    shuffled_df['y_col'] = shuffled_y

    return shuffled_df


def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100):
    """
    Generate a list of multiple data frames with shuffled x_col and y_col columns by calling the load and shuffle functions.
    :return: A list of shuffled pandas DataFrames.
    """

    # List to store the shuffled data frames
    shuffled_dfs = []
    
    # Loop to generate multiple shuffled data frames
    for i in range(num_shuffles):
        shuffled_df = _shuffle_xy(og_shear_df)
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs

def shear_grids_for_shuffled_dfs(list_of_dfs, boundaries, config): 
    '''
    Create shear grids for a list of dataframes.
    '''

    grid_list = []
    for shear_df in list_of_dfs: 
        g1map, g2map = create_shear_grid(shear_df['x_col'], 
                                           shear_df['y_col'], 
                                           shear_df['g1'],
                                           shear_df['g2'], 
                                           shear_df['weight'], 
                                           boundaries=boundaries,
                                           resolution=config['resolution'])

        grid_list.append((g1map, g2map))

    return grid_list
