"""Signal-to-noise maps from shuffled catalogs.

Turning each galaxy's shear by a random angle, or moving the shears between
galaxy positions, removes the lensing signal but keeps the shape noise, so
maps of shuffled catalogs sample the noise of a mass map.
"""

import logging
import random
import time

import jax
import jax.numpy as jnp
import numpy as np

from smpy._precision import enable_x64, to_numpy
from smpy.coordinates.base import _bin_shear, _grid_inputs

logger = logging.getLogger(__name__)

@jax.jit
def _rotate_shear(g1, g2, angle):
    """Turn the complex shear by a phase without changing its amplitude."""
    cosine, sine = jnp.cos(angle), jnp.sin(angle)
    return g1 * cosine - g2 * sine, g1 * sine + g2 * cosine

@jax.jit
def _noise_step(mean, m2, trial, count):
    """Accumulate one pair of null maps using Welford's recurrence."""
    delta = trial - mean
    mean = mean + delta / count
    return mean, m2 + delta * (trial - mean)

@jax.jit
def _signal_to_noise(signal, m2, count):
    """Divide signal maps by the sample standard deviation."""
    var = m2 / (count - 1)
    defined = var > 0
    sigma = jnp.sqrt(jnp.where(defined, var, 1))
    return jnp.where(defined, signal / sigma, jnp.nan)

def compute_snr(shear_df, coord_system, boundaries, mapper, maps, snr_settings):
    """Compute E- and B-mode signal-to-noise maps.

    Parameters
    ----------
    shear_df : `pandas.DataFrame`
        Catalog with transformed coordinates, as used to make ``maps``.
    coord_system : `smpy.coordinates.CoordinateSystem`
        Coordinate system that gridded the catalog.
    boundaries : `dict`
        Scaled boundaries of the map grid.
    mapper : `smpy.mapping_methods.MassMapper`
        Mapper that made ``maps``.
    maps : `dict` [`str`, `numpy.ndarray`]
        E- and B-mode maps of the unshuffled catalog.
    snr_settings : `dict`
        The ``snr`` settings, with an integer ``seed``.

    Returns
    -------
    snr : `dict` [`str`, `numpy.ndarray`]
        E- and B-mode signal-to-noise maps, NaN where the maps of the
        shuffled catalogs do not vary.

    Notes
    -----
    The signal-to-noise ratio is the map divided by the standard deviation
    of the maps of the shuffled catalogs, each gridded and mapped exactly as
    the catalog was. It is a detection statistic: it measures how unusual a
    pixel is next to maps without lensing signal. Shuffled catalogs keep each
    galaxy's shear amplitude, lensing included, and for KS+ the spread of maps
    without signal need not equal the uncertainty of a map with signal.
    """
    count = snr_settings["num_shuffles"]
    shuffled = generate_multiple_shear_dfs(
        shear_df, count, snr_settings["shuffle_type"], snr_settings["seed"],
    )
    grids = (coord_system.create_grid(catalog, boundaries) for catalog in shuffled)
    return to_numpy(_snr_from_grids(grids, mapper, maps, count))

def _compute_device_snr(shear_df, coord_system, boundaries, mapper, maps, snr_settings):
    """Bin null catalogs for the built-in coordinate systems on the device."""
    count = snr_settings["num_shuffles"]
    grids = _device_shuffle_grids(shear_df, coord_system, boundaries, count,
                                  snr_settings["shuffle_type"], snr_settings["seed"])
    return _snr_from_grids(grids, mapper, maps, count)

def _snr_from_grids(grids, mapper, maps, count):
    """Accumulate null-map variance without transferring intermediate maps."""
    start = time.perf_counter()
    with enable_x64(True):
        signal = jnp.stack((maps["E"], maps["B"]))
        mean = m2 = jnp.zeros_like(signal)
        for count_so_far, grid in enumerate(grids, 1):
            trial = jnp.stack(mapper.create_maps(grid))
            mean, m2 = _noise_step(mean, m2, trial, count_so_far)
        snr = dict(zip("EB", _signal_to_noise(signal, m2, count)))
        jax.block_until_ready(snr)
    logger.info("Created SNR maps in %.2f s", time.perf_counter() - start)
    return snr

def _device_shuffle_grids(shear_df, coord_system, boundaries, num_shuffles, shuffle_type, seed):
    """Grid null catalogs without copying their columns through the host."""
    (x, y, g1, g2, weight), shape = _grid_inputs(coord_system.prepare_data(shear_df), boundaries)
    if shuffle_type == "orientation":
        rng = np.random.default_rng(seed)
        for _ in range(num_shuffles):
            angle = rng.uniform(0, 2 * np.pi, len(g1))
            rotated = _rotate_shear(g1, g2, angle)
            yield _bin_shear(x, y, *rotated, weight, shape=shape)
    elif shuffle_type == "spatial":
        for index in range(num_shuffles):
            perm = _permutation(len(g1), random.Random(seed + index))
            yield _bin_shear(x[perm], y[perm], g1, g2, weight, shape=shape)
    else:
        raise ValueError(f"Invalid shuffle type: {shuffle_type}")

def _permutation(size, rng):
    """Permute row numbers in the legacy spatial shuffle order."""
    indices = list(range(size))
    rng.shuffle(indices)
    return np.asarray(indices, dtype=np.int32)

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

    columns = ['coord1_scaled', 'coord2_scaled']
    shuffled_df[columns] = shear_df[columns].to_numpy()[_permutation(len(shear_df), rng)]
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

    g1, g2 = (shear_df[key].to_numpy(dtype=np.float64) for key in ("g1", "g2"))
    angle = rng.uniform(0, 2 * np.pi, len(g1))
    with enable_x64(True):
        shuffled_df['g1'], shuffled_df['g2'] = jax.device_get(_rotate_shear(g1, g2, angle))
    return shuffled_df

def generate_multiple_shear_dfs(og_shear_df, num_shuffles=100, shuffle_type='orientation', seed=0):
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
        Type of shuffling: 'orientation' (turn each shear by a random angle)
        or 'spatial' (permute the galaxy positions).
    seed : `int`, optional
        Random seed for reproducibility.

    Returns
    -------
    shuffled_catalogs : `collections.abc.Iterator` [`pandas.DataFrame`]
        Shuffled copies of the catalog, each made only when it is reached,
        so one copy is held in memory at a time.

    Raises
    ------
    ValueError
        If invalid shuffle_type is specified.

    Notes
    -----
    Orientation shuffles keep every galaxy's position, weight and shear
    amplitude, so each pixel keeps its own noise. Spatial shuffles move shears
    and weights between positions, so they keep each pixel's noise only if
    the noise is the same across the field; where depth, weights or shape
    noise vary, they mix the noise of different regions.

    Each shuffle type draws from its own generator seeded by ``seed``, so the
    global `random` and `numpy.random` states are neither used nor changed.
    """
    if shuffle_type == 'spatial':
        return (_shuffle_coordinates(og_shear_df, random.Random(seed + i)) for i in range(num_shuffles))
    if shuffle_type == 'orientation':
        rng = np.random.default_rng(seed)
        return (_shuffle_galaxy_rotation(og_shear_df, rng) for _ in range(num_shuffles))
    raise ValueError(f"Invalid shuffle type: {shuffle_type}")
