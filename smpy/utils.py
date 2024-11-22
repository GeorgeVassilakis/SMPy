import numpy as np
import pandas as pd
import random

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS

def load_shear_data(shear_cat_path, coord1_col, coord2_col, g1_col, g2_col, weight_col=None):
    """
    Load shear data from a FITS file and return a pandas DataFrame.
    
    Parameters
    ----------
    shear_cat_path : str
        Path to the FITS file
    coord1_col : str
        Column name for first coordinate (RA or X)
    coord2_col : str
        Column name for second coordinate (Dec or Y)
    g1_col : str
        Column name for first shear component
    g2_col : str
        Column name for second shear component
    weight_col : str, optional
        Column name for weights, if None uses unit weights
        
    Returns
    -------
    pd.DataFrame
        DataFrame with coordinates, shear, and weight columns
    """
    # Read data from FITS file
    shear_catalog = Table.read(shear_cat_path)
    
    # Convert to pandas DataFrame with generic column names
    shear_df = pd.DataFrame({
        'coord1': shear_catalog[coord1_col],
        'coord2': shear_catalog[coord2_col],
        'g1': shear_catalog[g1_col],
        'g2': shear_catalog[g2_col],
    })
    
    # Add weights (unit weights if not specified)
    if weight_col is None:
        shear_df['weight'] = np.ones(len(shear_df))
    else:
        shear_df['weight'] = shear_catalog[weight_col]
    
    return shear_df

def save_convergence_fits(convergence, boundaries, true_boundaries, config, output_name):
    """
    Save convergence map as a FITS file with WCS information.
    
    Parameters
    ----------
    convergence : np.ndarray
        2D convergence map
    boundaries : dict
        Dictionary containing scaled coordinate boundaries
    true_boundaries : dict
        Dictionary containing true coordinate boundaries
    config : dict
        Configuration dictionary
    output_name : str
        Output file path
    """
    if not config.get('save_fits', False):
        return
    
    # Create a WCS object
    wcs = WCS(naxis=2)
    
    # Set up the WCS parameters based on coordinate system
    npix_dec, npix_ra = convergence.shape
    
    # Use true boundaries for WCS information
    wcs.wcs.crpix = [npix_ra / 2, npix_dec / 2]
    wcs.wcs.cdelt = [
        (true_boundaries['coord1_max'] - true_boundaries['coord1_min']) / npix_ra,
        (true_boundaries['coord2_max'] - true_boundaries['coord2_min']) / npix_dec
    ]
    wcs.wcs.crval = [
        (true_boundaries['coord1_max'] + true_boundaries['coord1_min']) / 2,
        (true_boundaries['coord2_max'] + true_boundaries['coord2_min']) / 2
    ]
    
    # Set coordinate type based on coordinate system
    if config.get('coordinate_system', 'radec').lower() == 'radec':
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    else:
        wcs.wcs.ctype = ["X", "Y"]
    
    # Create a FITS header from the WCS information
    header = wcs.to_header()
    
    # Add metadata to header
    header['AUTHOR'] = 'SMPy'
    header['CONTENT'] = 'Convergence Map'
    header['COORDSYS'] = config.get('coordinate_system', 'radec').upper()
    header['UNIT1'] = true_boundaries.get('units', '')
    header['UNIT2'] = true_boundaries.get('units', '')
    
    # Create HDU and save
    hdu = fits.PrimaryHDU(convergence, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(output_name, overwrite=True)
    
    print(f"Convergence map saved as FITS file: {output_name}")

def find_peaks2d(image, threshold=None, verbose=False, true_boundaries=None, scaled_boundaries=None):
    """
    Identify peaks in a 2D array above a specified threshold.
    A peak is a pixel with a value greater than its 8 neighbors.

    Parameters
    ----------
    image : np.ndarray
        2D array representing the image
    threshold : float, optional
        Minimum pixel value to consider as a peak
    verbose : bool
        Whether to print peak information
    true_boundaries : dict
        Dictionary containing true coordinate boundaries for coordinate conversion
    scaled_boundaries : dict
        Dictionary containing scaled coordinate boundaries for coordinate conversion

    Returns
    -------
    tuple
        (X, Y, heights, coords) where:
        - X, Y are peak indices
        - heights are peak values
        - coords are true coordinates (if boundaries provided)
    """
    image = np.atleast_2d(image)
    threshold = threshold if threshold is not None else image.min()

    # Pad the image for border peak checks
    padded_image = np.pad(image, pad_width=1, mode='constant', 
                         constant_values=image.min())

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

    # Get peak coordinates and heights
    Y, X = np.nonzero(peaks_mask)
    heights = image[Y, X]

    # Sort peaks by height in descending order
    sort_indices = np.argsort(-heights)
    X = X[sort_indices]
    Y = Y[sort_indices]
    heights = heights[sort_indices]

    # Convert pixel coordinates to true coordinates if boundaries are provided
    coords = None
    if verbose and true_boundaries and scaled_boundaries:
        coords = []
        coord_system = 'pixel' if 'X' in true_boundaries['coord1_name'] else 'radec'
        
        for x, y in zip(X, Y):
            # Convert pixel indices to scaled coordinates
            scaled_coord1 = scaled_boundaries['coord1_min'] + (x + 0.5) * (
                scaled_boundaries['coord1_max'] - scaled_boundaries['coord1_min']
            ) / image.shape[1]
            
            scaled_coord2 = scaled_boundaries['coord2_min'] + (y + 0.5) * (
                scaled_boundaries['coord2_max'] - scaled_boundaries['coord2_min']
            ) / image.shape[0]
            
            # Convert scaled coordinates to true coordinates
            true_coord1 = np.interp(
                scaled_coord1,
                [scaled_boundaries['coord1_min'], scaled_boundaries['coord1_max']],
                [true_boundaries['coord1_min'], true_boundaries['coord1_max']]
            )
            true_coord2 = np.interp(
                scaled_coord2,
                [scaled_boundaries['coord2_min'], scaled_boundaries['coord2_max']],
                [true_boundaries['coord2_min'], true_boundaries['coord2_max']]
            )
            
            coords.append((true_coord1, true_coord2))

        # Print peak information with appropriate coordinate labels
        if coord_system == 'radec':
            coord1_label, coord2_label = 'RA', 'Dec'
        else:
            coord1_label, coord2_label = 'X', 'Y'
            
        print("\nDetected Peaks:")
        print("-" * 60)
        print(f"{'Peak #':<8}{'Value':<12}{coord1_label:<12}{coord2_label:<12}")
        print("-" * 60)
        for i, ((c1, c2), height) in enumerate(zip(coords, heights), 1):
            print(f"{i:<8}{height:.<12.5f}{c1:.<12.5f}{c2:.<12.5f}")
        print("-" * 60)

    return X, Y, heights, coords

def g1g2_to_gt_gc(g1, g2, coord1, coord2, center_coord1, center_coord2, pix_coord1=100):
    """
    Convert reduced shear to tangential and cross components.
    Works with either RA/Dec or pixel coordinates.
    
    Parameters
    ----------
    g1, g2 : np.ndarray
        Reduced shear components
    coord1, coord2 : np.ndarray
        Coordinates (either RA/Dec or X/Y)
    center_coord1, center_coord2 : float
        Center coordinates
    pix_coord1 : int
        Number of pixels in first coordinate dimension
        
    Returns
    -------
    tuple
        (gt, gc, phi) tangential shear, cross shear, and polar angle
    """
    coord1_max = np.max(coord1)
    coord1_min = np.min(coord1)
    coord2_max = np.max(coord2)
    coord2_min = np.min(coord2)
    
    aspect_ratio = (coord1_max - coord1_min) / (coord2_max - coord2_min)
    pix_coord2 = int(pix_coord1 / aspect_ratio)
    
    # Create coordinate grid
    coord1_grid, coord2_grid = np.meshgrid(
        np.linspace(coord1_min, coord1_max, pix_coord1),
        np.linspace(coord2_min, coord2_max, pix_coord2)
    )
    
    # Calculate polar angle
    phi = np.arctan2(coord2_grid - center_coord2, 
                     coord1_grid - center_coord1)
    
    # Calculate tangential and cross components
    gt = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)
    gc = -g1 * np.sin(2 * phi) + g2 * np.cos(2 * phi)
    
    return gt, gc, phi

def _shuffle_coordinates(shear_df, seed=None):
    """
    Shuffle the scaled coordinates of the input DataFrame together.
    
    Parameters
    ----------
    shear_df : pd.DataFrame
        Input DataFrame with scaled coordinates
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shuffled coordinates
    """
    if seed is not None:
        random.seed(seed)
        
    shuffled_df = shear_df.copy()

    # Combine scaled coordinates into pairs
    coord_pairs = list(zip(shuffled_df['coord1_scaled'], 
                          shuffled_df['coord2_scaled']))
    
    # Shuffle the pairs
    random.shuffle(coord_pairs)
    
    # Unzip the shuffled pairs
    shuffled_coord1, shuffled_coord2 = zip(*coord_pairs)
    
    # Update the scaled coordinates
    shuffled_df['coord1_scaled'] = shuffled_coord1
    shuffled_df['coord2_scaled'] = shuffled_coord2

    return shuffled_df

def _shuffle_galaxy_rotation(shear_df):
    """
    Shuffle the galaxy rotation in the input shear DataFrame.
    
    Parameters
    ----------
    shear_df : pd.DataFrame
        Input DataFrame with shear components
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shuffled shear components
    """
    shuffled_df = shear_df.copy()
    
    # Add a random angle to the galaxy rotation
    g1, g2 = shuffled_df['g1'], shuffled_df['g2']
    angle = np.random.uniform(0, 2 * np.pi, len(g1))
    g1g2_len = np.sqrt(np.array(g1)**2 + np.array(g2)**2)
    g1g2_angle = np.arctan2(g2, g1) + angle
    
    shuffled_df['g1'] = g1g2_len * np.cos(g1g2_angle)
    shuffled_df['g2'] = g1g2_len * np.sin(g1g2_angle)
    
    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, shuffle_type='spatial', seed=0):
    """
    Generate multiple shuffled versions of the input DataFrame.
    
    Parameters
    ----------
    og_shear_df : pd.DataFrame
        Original shear DataFrame
    num_shuffles : int
        Number of shuffled copies to generate
    shuffle_type : str
        Type of shuffling ('spatial' or 'orientation')
    seed : int
        Starting random seed
        
    Returns
    -------
    list
        List of shuffled DataFrames
    """
    shuffled_dfs = []
    
    for i in range(num_shuffles):
        if shuffle_type == 'spatial':
            shuffled_df = _shuffle_coordinates(og_shear_df, seed=seed+i)
        elif shuffle_type == 'orientation':
            shuffled_df = _shuffle_galaxy_rotation(og_shear_df)
        else:
            raise ValueError(f"Invalid shuffle type: {shuffle_type}")
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs
