"""Signal-to-noise maps from shuffled catalogs.

Shuffling galaxy positions or orientations removes the lensing signal but keeps
the shape noise, so maps of shuffled catalogs sample the noise of a mass map.
"""

import logging
import random
import secrets
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from smpy.mapping_methods import METHODS

logger = logging.getLogger(__name__)

def compute_snr(shear_df, coord_system, boundaries, maps, settings):
    """Compute E- and B-mode signal-to-noise maps.

    Parameters
    ----------
    shear_df : `pandas.DataFrame`
        Catalog with transformed coordinates, as used to make ``maps``.
    coord_system : `smpy.coordinates.CoordinateSystem`
        Coordinate system that gridded the catalog.
    boundaries : `dict`
        Scaled boundaries of the map grid.
    maps : `dict` [`str`, `numpy.ndarray`]
        E- and B-mode maps of the unshuffled catalog.
    settings : `dict`
        Full SMPy settings. The ``snr`` section controls the shuffles.

    Returns
    -------
    snr : `dict` [`str`, `numpy.ndarray`]
        E- and B-mode signal-to-noise maps.

    Notes
    -----
    The signal-to-noise ratio is the smoothed map divided by the standard
    deviation of the smoothed maps of the shuffled catalogs. For KS+, the
    shuffled catalogs are mapped with plain Kaiser-Squires.
    """
    general, snr_settings = settings["general"], settings["snr"]
    method = "kaiser_squires" if general["method"] == "ks_plus" else general["method"]
    mapper = METHODS[method](**settings["methods"][method])
    sigma = snr_settings["smoothing"]

    start = time.perf_counter()
    noise = {"E": [], "B": []}
    shuffled_dfs = generate_multiple_shear_dfs(shear_df, snr_settings["num_shuffles"],
                                               snr_settings["shuffle_type"], snr_settings["seed"])
    for shuffled_df in shuffled_dfs:
        grid = coord_system.create_grid(shuffled_df, boundaries)
        for mode, noise_map in zip("EB", mapper.create_maps(grid)):
            noise[mode].append(gaussian_filter(noise_map, sigma))
    snr = {mode: gaussian_filter(maps[mode], sigma) / np.sqrt(np.var(np.stack(noise[mode], axis=0), axis=0))
           for mode in "EB"}
    logger.info("Created %s SNR maps in %.2f s", general["method"], time.perf_counter() - start)
    return snr

def _shuffle_coordinates(shear_df, rng):
    """Shuffle the scaled coordinates of the input DataFrame together.

    Randomly permute coordinate pairs while maintaining the pairing between
    coord1_scaled and coord2_scaled columns.

    Parameters
    ----------
    shear_df : `pandas.DataFrame`
        Input DataFrame with scaled coordinates.
    rng : `random.Random`
        Random number generator for the permutation.

    Returns
    -------
    shuffled_df : `pandas.DataFrame`
        DataFrame with shuffled coordinate pairs.
    """
    shuffled_df = shear_df.copy()

    # Combine scaled coordinates into pairs
    coord_pairs = list(zip(shuffled_df['coord1_scaled'],
                          shuffled_df['coord2_scaled']))

    # Shuffle the pairs
    rng.shuffle(coord_pairs)

    # Unzip the shuffled pairs
    shuffled_coord1, shuffled_coord2 = zip(*coord_pairs)

    # Update the scaled coordinates
    shuffled_df['coord1_scaled'] = shuffled_coord1
    shuffled_df['coord2_scaled'] = shuffled_coord2

    return shuffled_df

def _shuffle_galaxy_rotation(shear_df, rng):
    """Shuffle galaxy orientations in the input shear DataFrame.

    Randomly rotate galaxy orientations by adding random angles to the
    shear components while preserving shear magnitudes.

    Parameters
    ----------
    shear_df : `pandas.DataFrame`
        Input DataFrame with shear components.
    rng : `numpy.random.Generator`
        Random number generator for the rotation angles.

    Returns
    -------
    shuffled_df : `pandas.DataFrame`
        DataFrame with randomly rotated shear components.
    """
    shuffled_df = shear_df.copy()

    # Add a random angle to the galaxy rotation
    g1, g2 = shuffled_df['g1'], shuffled_df['g2']
    angle = rng.uniform(0, 2 * np.pi, len(g1))
    g1g2_len = np.sqrt(np.array(g1)**2 + np.array(g2)**2)
    g1g2_angle = np.arctan2(g2, g1) + angle

    shuffled_df['g1'] = g1g2_len * np.cos(g1g2_angle)
    shuffled_df['g2'] = g1g2_len * np.sin(g1g2_angle)

    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, shuffle_type='spatial', seed=0):
    """Generate shuffled versions of shear catalog.

    Create multiple randomized versions of the input shear catalog for
    null hypothesis testing and error estimation.

    Parameters
    ----------
    og_shear_df : `pandas.DataFrame`
        Original shear catalog.
    num_shuffles : `int`, optional
        Number of shuffled versions to generate.
    shuffle_type : `str`, optional
        Type of shuffling: 'spatial' (randomize positions) or 'orientation'
        (randomize galaxy orientations).
    seed : `int` or `str`, optional
        Random seed for reproducibility. If 'random', the seed is drawn from
        the `secrets` module.

    Returns
    -------
    shuffled_catalogs : `list` of `pandas.DataFrame`
        List of shuffled DataFrame copies.

    Raises
    ------
    ValueError
        If invalid shuffle_type is specified.

    Notes
    -----
    Each shuffle type draws from its own generator seeded by ``seed``, so the
    global `random` and `numpy.random` states are neither used nor changed.
    """
    if seed == 'random':
        seed = secrets.randbits(128)

    if shuffle_type == 'spatial':
        return [_shuffle_coordinates(og_shear_df, random.Random(seed + i)) for i in range(num_shuffles)]
    if shuffle_type == 'orientation':
        rng = np.random.default_rng(seed)
        return [_shuffle_galaxy_rotation(og_shear_df, rng) for _ in range(num_shuffles)]
    raise ValueError(f"Invalid shuffle type: {shuffle_type}")
