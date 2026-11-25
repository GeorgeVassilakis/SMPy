"""Base class for coordinate systems.

This module defines the abstract base class for coordinate systems used
in mass mapping operations, providing a consistent interface for RA/Dec
and pixel coordinate transformations and gridding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from smpy._precision import enable_x64, to_numpy

@jax.tree_util.register_dataclass
@dataclass
class ShearGrid:
    """Shear catalog binned onto a regular map grid.

    All arrays have shape ``(ny, nx)``: rows follow the map's y axis (north
    for RA/Dec) and columns its x axis (west for RA/Dec).

    Parameters
    ----------
    g1 : `numpy.ndarray` or `jax.Array`
        Weighted mean first shear component per pixel; 0 in empty pixels.
    g2 : `numpy.ndarray` or `jax.Array`
        Weighted mean second shear component per pixel; 0 in empty pixels.
    weights : `numpy.ndarray` or `jax.Array`
        Summed galaxy weights per pixel. A weight of 0 marks an empty pixel.
    counts : `numpy.ndarray` or `jax.Array`
        Number of galaxies per pixel, fractional for bilinear gridding.
    """

    g1: np.ndarray | jax.Array
    g2: np.ndarray | jax.Array
    weights: np.ndarray | jax.Array
    counts: np.ndarray | jax.Array

def _accumulate(indices, contributions, size):
    """Sum contributions in a reproducible order on each backend."""
    zeros = jnp.zeros((size, contributions.shape[-1]), dtype=contributions.dtype)
    if not indices.size:
        return zeros

    def scatter(indices, contributions):
        return zeros.at[indices].add(contributions)

    def segmented_sum(indices, contributions):
        # Floating-point atomics can sum repeated indices in different
        # orders. Sort stably and write each segment's total once.
        order = jnp.argsort(indices, stable=True)
        indices, contributions = indices[order], contributions[order]

        def combine(left, right):
            left_bin, left_sum = left
            right_bin, right_sum = right
            return right_bin, right_sum + jnp.where((left_bin == right_bin)[:, None], left_sum, 0)

        _, sums = lax.associative_scan(combine, (indices, contributions))
        last = jnp.concatenate((indices[:-1] != indices[1:], jnp.ones(1, dtype=bool)))
        destinations = jnp.where(last, indices, size + jnp.arange(indices.size, dtype=indices.dtype))
        return zeros.at[destinations].set(sums, mode="drop", unique_indices=True)

    return lax.platform_dependent(indices, contributions, cpu=scatter, default=segmented_sum)

@partial(jax.jit, static_argnames=("shape", "gridding"))
def _bin_shear(x, y, g1, g2, weights, *, shape, gridding="hard"):
    """Grid weighted shear in edge-based pixel coordinates."""
    ny, nx = shape
    contributions = jnp.stack((g1 * weights, g2 * weights, weights, jnp.ones_like(weights)), axis=-1)
    if gridding == "hard":
        ix = jnp.clip(jnp.floor(x), 0, nx - 1).astype(jnp.int32)
        iy = jnp.clip(jnp.floor(y), 0, ny - 1).astype(jnp.int32)
    else:
        px = jnp.clip(x - 0.5, 0, nx - 1)
        py = jnp.clip(y - 0.5, 0, ny - 1)
        ix = jnp.floor(px).astype(jnp.int32)
        iy = jnp.floor(py).astype(jnp.int32)
        fx, fy = px - ix, py - iy
        fractions = jnp.stack(((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy), axis=-1)
        ix = jnp.minimum(ix[:, None] + jnp.array([0, 1, 0, 1], dtype=ix.dtype), nx - 1)
        iy = jnp.minimum(iy[:, None] + jnp.array([0, 0, 1, 1], dtype=iy.dtype), ny - 1)
        contributions = contributions[:, None, :] * fractions[..., None]
    totals = _accumulate((iy * nx + ix).reshape(-1), contributions.reshape(-1, 4), ny * nx)
    sum_g1, sum_g2, sum_w, counts = (totals[:, i].reshape(shape) for i in range(4))
    denominator = jnp.where(sum_w != 0, sum_w, 1)
    return ShearGrid(sum_g1 / denominator, sum_g2 / denominator, sum_w, counts)

def _grid_inputs(data_df, boundaries):
    """Return device catalog columns in pixel units and the static shape."""
    size = boundaries["pixel_size"]
    shape = tuple(round((boundaries[f"{axis}_max"] - boundaries[f"{axis}_min"]) / size)
                  for axis in ("coord2", "coord1"))
    left, bottom = boundaries["coord1_min"], boundaries["coord2_min"]
    x, y, g1, g2, weight = (jnp.asarray(data_df[key].to_numpy(dtype=np.float64))
                             for key in ("coord1_scaled", "coord2_scaled", "g1", "g2", "weight"))
    return ((x - left) / size, (y - bottom) / size, g1, g2, weight), shape

class CoordinateSystem(ABC):
    """Abstract base class for coordinate systems.

    Provides interface for RA/Dec and pixel coordinate systems used in mass
    mapping. Each coordinate system transforms a catalog into its gridding
    coordinates and places the map grid around it; binning onto the grid is
    shared.
    """

    @abstractmethod
    def calculate_boundaries(self, data_df, settings):
        """Calculate the map grid around a catalog.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog from `transform_coordinates`.
        settings : `dict`
            The coordinate system's section of ``general``, such as
            ``general.radec``.

        Returns
        -------
        scaled_boundaries : `dict`
            Map edges in the gridding coordinates (coord1_min/max,
            coord2_min/max), the map ``pixel_size`` in the same units, and
            labels: coord1_name, coord2_name, units.
        true_boundaries : `dict`
            Ranges of the catalog's own coordinates, with the same labels.
        """

    @abstractmethod
    def transform_coordinates(self, data_df):
        """Transform coordinates if needed (e.g., centering, scaling).

        Apply coordinate system specific transformations such as centering
        or scaling to prepare coordinates for gridding operations.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with original coord1, coord2 coordinate columns.

        Returns
        -------
        transformed_df : `pandas.DataFrame`
            DataFrame with additional coord1_scaled, coord2_scaled columns
            containing transformed coordinates ready for gridding.
        """

    def create_grid(self, data_df, boundaries):
        """Bin a catalog onto the map grid.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog with coord1_scaled and coord2_scaled from
            `transform_coordinates`, g1, g2 and weight.
        boundaries : `dict`
            Map edges and ``pixel_size`` from `calculate_boundaries`.

        Returns
        -------
        grid : `ShearGrid`
            Binned shear, weights and counts.
        """
        return to_numpy(self._create_device_grid(data_df, boundaries))

    def _create_device_grid(self, data_df, boundaries):
        """Bin the catalog and leave the resulting grid on the device."""
        data_df = self.prepare_data(data_df)
        with enable_x64(True):
            columns, shape = _grid_inputs(data_df, boundaries)
            return _bin_shear(*columns, shape=shape)

    @staticmethod
    def _grid_edges(data_df, pixel_size):
        """Return the edges of a map of whole pixels centered on a catalog.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog with coord1_scaled and coord2_scaled.
        pixel_size : `float`
            Pixel width in the same units.

        Returns
        -------
        boundaries : `dict`
            coord1_min/max, coord2_min/max and ``pixel_size``. Every galaxy
            lies inside the map.
        """
        boundaries = {'pixel_size': pixel_size}
        for axis in ('coord1', 'coord2'):
            values = data_df[f'{axis}_scaled']
            npix = int(np.floor((values.max() - values.min()) / pixel_size)) + 1
            middle = (values.max() + values.min()) / 2
            boundaries[f'{axis}_min'] = middle - npix * pixel_size / 2
            boundaries[f'{axis}_max'] = middle + npix * pixel_size / 2
        return boundaries

    def prepare_data(self, data_df):
        """Prepare for gridding by validating and transforming coordinates.

        Check that required columns exist and ensure coordinates are
        transformed appropriately before gridding operations.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Input DataFrame containing coordinates and shear measurements.

        Returns
        -------
        processed_df : `pandas.DataFrame`
            Processed DataFrame with all required columns and transformed
            coordinates.

        Raises
        ------
        ValueError
            If required columns are missing from the input DataFrame.
        """
        required_cols = ['coord1', 'coord2', 'g1', 'g2', 'weight']
        if not all(col in data_df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Transform coordinates if not already done
        if 'coord1_scaled' not in data_df.columns:
            data_df = self.transform_coordinates(data_df)

        return data_df
