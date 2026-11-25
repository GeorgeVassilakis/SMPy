"""Device mapping for the catalog pipeline."""

import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

from .arrays import METHODS
from .coordinates import _bin_shear
from .snr import compute_snr

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

logger = logging.getLogger(__name__)

def _grid_inputs(data_df, boundaries):
    """Return device catalog columns in pixel units and the static shape."""
    size = boundaries["pixel_size"]
    shape = tuple(round((boundaries[f"{axis}_max"] - boundaries[f"{axis}_min"]) / size)
                  for axis in ("coord2", "coord1"))
    left, bottom = boundaries["coord1_min"], boundaries["coord2_min"]
    x, y, g1, g2, weight = (jnp.asarray(data_df[key].to_numpy(dtype=np.float64))
                             for key in ("coord1_scaled", "coord2_scaled", "g1", "g2", "weight"))
    return ((x - left) / size, (y - bottom) / size, g1, g2, weight), shape

def _to_numpy(tree):
    """Transfer arrays to the host with writable NumPy storage."""
    def writable(array):
        if isinstance(array, np.ndarray) and not array.flags.writeable:
            return array.copy()
        return array

    return jax.tree_util.tree_map(writable, jax.device_get(tree))

def compute_maps(shear_df, boundaries, method, method_settings, snr_settings):
    """Bin a projected catalog and return writable float64 host results."""
    with enable_x64(True):
        columns, shape = _grid_inputs(shear_df, boundaries)
        grid = _bin_shear(*columns, shape=shape)
        start = time.perf_counter()
        mapper = METHODS[method](**method_settings)
        maps = dict(zip("EB", mapper.create_maps(grid)))
        jax.block_until_ready(maps)
        logger.info("Created %s maps in %.2f s", method, time.perf_counter() - start)
        snr = None if snr_settings is None else compute_snr(columns, shape, mapper, maps, snr_settings)
        return _to_numpy((maps, snr, grid))
