import numpy as np
import pandas as pd
import random
import secrets

from astropy.table import Table
from astropy.io import fits

def load_shear_data(shear_cat_path, coord1_col, coord2_col, g1_col, g2_col, weight_col=None, hdu=0):
    """Load shear catalog from FITS file.

    Parameters
    ----------
    shear_cat_path : `str`
        Path to FITS catalog
    coord1_col : `str`
        Name of first coordinate column
    coord2_col : `str`
        Name of second coordinate column
    g1_col : `str`
        Name of g1 shear column
    g2_col : `str`
        Name of g2 shear column
    weight_col : `str`, optional
        Name of weight column
    hdu : `int`
        HDU number to read

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardized column names
        
    Raises
    ------
    IndexError
        If HDU not found
    KeyError
        If required columns not found
    """
    # Read data from FITS file
    try:
        shear_catalog = Table.read(shear_cat_path, hdu=hdu)
    except IndexError:
        raise IndexError(f"HDU {hdu} not found in {shear_cat_path}")
    
    # Convert to pandas DataFrame with generic column names
    try:
        shear_df = pd.DataFrame({
            'coord1': shear_catalog[coord1_col],
            'coord2': shear_catalog[coord2_col],
            'g1': shear_catalog[g1_col],
            'g2': shear_catalog[g2_col],
        })
    except KeyError as e:
        raise KeyError(f"Column {e} not found in HDU {hdu}. Available columns: {shear_catalog.colnames}")
    
    # Add weights (unit weights if not specified)
    if weight_col is None:
        shear_df['weight'] = np.ones(len(shear_df))
    else:
        shear_df['weight'] = shear_catalog[weight_col]
    
    return shear_df

def find_peaks2d(image, threshold=None, verbose=False, true_boundaries=None, scaled_boundaries=None):
    """
    Identify peaks in a 2D array above a specified threshold.
    A peak is a pixel with a value greater than its 8 neighbors.
    Refactored from cosmostat/lenspack.

    Parameters
    ----------
    image : `numpy.ndarray`
        2D input map
    threshold : `float`, optional
        Detection threshold
    verbose : `bool`
        Print peak information
    true_boundaries : `dict`, optional
        True coordinate boundaries for position conversion
    scaled_boundaries : `dict`, optional
        Scaled coordinate boundaries for position conversion

    Returns
    -------
    X, Y : `numpy.ndarray`
        Peak pixel indices
    heights : `numpy.ndarray`
        Peak values
    coords : `list`
        Peak coordinates in true system if boundaries provided
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
    """Convert shear components to tangential/cross components.

    Parameters
    ----------
    g1, g2 : `numpy.ndarray`
        Shear components
    coord1, coord2 : `numpy.ndarray`
        Coordinates (RA/Dec or X/Y)
    center_coord1, center_coord2 : `float`
        Center coordinates
    pix_coord1 : `int`
        Grid size in first dimension

    Returns
    -------
    gt, gc, phi : `numpy.ndarray`
        Tangential shear, cross shear, polar angle
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
    """Shuffle the scaled coordinates of the input DataFrame together.
    
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

def _shuffle_galaxy_rotation(shear_df, rng=None):
    """Shuffle the galaxy rotation in the input shear DataFrame.
    
    Parameters
    ----------
    shear_df : pd.DataFrame
        Input DataFrame with shear components
    rng : np.random.Generator, optional
        Random number generator for orientation shuffling
        
    Returns
    -------
    pd.DataFrame
        DataFrame with shuffled shear components
    """
    shuffled_df = shear_df.copy()
    
    # Add a random angle to the galaxy rotation
    g1, g2 = shuffled_df['g1'], shuffled_df['g2']
    if rng is None:
        angle = np.random.uniform(0, 2 * np.pi, len(g1))
    else:
        angle = rng.uniform(0, 2 * np.pi, len(g1))
    g1g2_len = np.sqrt(np.array(g1)**2 + np.array(g2)**2)
    g1g2_angle = np.arctan2(g2, g1) + angle
    
    shuffled_df['g1'] = g1g2_len * np.cos(g1g2_angle)
    shuffled_df['g2'] = g1g2_len * np.sin(g1g2_angle)
    
    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, shuffle_type='spatial', seed=0):
    """Generate shuffled versions of shear catalog.

    Parameters
    ----------
    og_shear_df : `pandas.DataFrame`
        Original shear catalog
    num_shuffles : `int`
        Number of shuffled versions
    shuffle_type : `str`
        'spatial' or 'orientation'
    seed : `int` or `str`
        Random seed for reproducibility. If 'random', uses cryptographically 
        secure random number from secrets module.
        
    Returns
    -------
    list
        List of shuffled DataFrames
        
    Raises
    ------
    ValueError
        If invalid shuffle_type specified
    """
    shuffled_dfs = []
    
    # Handle 'random' seed option
    if seed == 'random':
        seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)
        # For orientation shuffling, we'll use NumPy's RNG directly
        # For spatial shuffling, we still use Python's random module with the seed
        if shuffle_type == 'spatial':
            random.seed(seed)
    
    for i in range(num_shuffles):
        if shuffle_type == 'spatial':
            if seed != 'random':
                # Only set seed for each iteration if not using the 'random' option
                shuffled_df = _shuffle_coordinates(og_shear_df, seed=seed+i)
            else:
                # For 'random', we already set the seed once outside the loop
                shuffled_df = _shuffle_coordinates(og_shear_df, seed=None)
        elif shuffle_type == 'orientation':
            if seed == 'random':
                # Use NumPy's RNG directly for orientation shuffling with 'random' option
                shuffled_df = _shuffle_galaxy_rotation(og_shear_df, rng=rng)
            else:
                # Standard orientation shuffling
                shuffled_df = _shuffle_galaxy_rotation(og_shear_df)
        else:
            raise ValueError(f"Invalid shuffle type: {shuffle_type}")
        shuffled_dfs.append(shuffled_df)
    
    return shuffled_dfs

def save_fits(data, true_boundaries, filename):
    """
    Save a 2D array as a FITS file with proper WCS information.
    
    Parameters
    ----------
    - data: 2D numpy array containing the map.
    - true_boundaries: Dictionary with 'ra_min', 'ra_max', 'dec_min', 'dec_max'.
    - filename: Output filename.
    """
    hdu = fits.PrimaryHDU(data)
    header = hdu.header

    ny, nx = data.shape
    ra_min, ra_max = true_boundaries['ra_min'], true_boundaries['ra_max']
    dec_min, dec_max = true_boundaries['dec_min'], true_boundaries['dec_max']

    pixel_scale_ra = (ra_max - ra_min) / nx
    pixel_scale_dec = (dec_max - dec_min) / ny

    header["CTYPE1"] = "RA---TAN"
    header["CUNIT1"] = "deg"
    header["CRVAL1"] = (ra_max + ra_min) / 2
    header["CRPIX1"] = nx / 2
    header["CD1_1"]  = -pixel_scale_ra
    header["CD1_2"]  = 0.0

    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT2"] = "deg"
    header["CRVAL2"] = (dec_max + dec_min) / 2
    header["CRPIX2"] = ny / 2
    header["CD2_1"]  = 0.0
    header["CD2_2"]  = pixel_scale_dec

    hdu.writeto(filename, overwrite=True)
    print(f"Saved FITS file: {filename}")