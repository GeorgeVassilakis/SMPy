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

from .coordinates import _bin_shear

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

def compute_snr(columns, shape, mapper, maps, snr_settings):
    """Compute SNR without transferring intermediate grids or maps."""
    start = time.perf_counter()
    count = snr_settings["num_shuffles"]
    grids = _device_shuffle_grids(columns, shape, count, snr_settings["shuffle_type"], snr_settings["seed"])
    signal = jnp.stack((maps["E"], maps["B"]))
    mean = m2 = jnp.zeros_like(signal)
    for count_so_far, grid in enumerate(grids, 1):
        trial = jnp.stack(mapper.create_maps(grid))
        mean, m2 = _noise_step(mean, m2, trial, count_so_far)
    snr = dict(zip("EB", _signal_to_noise(signal, m2, count)))
    jax.block_until_ready(snr)
    logger.info("Created SNR maps in %.2f s", time.perf_counter() - start)
    return snr

def _device_shuffle_grids(columns, shape, num_shuffles, shuffle_type, seed):
    """Grid null catalogs without copying their columns through the host."""
    x, y, g1, g2, weight = columns
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
