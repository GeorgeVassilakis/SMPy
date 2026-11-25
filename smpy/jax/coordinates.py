"""JAX projection and binning of shear catalogs."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from smpy.coordinates.base import ShearGrid

jax.tree_util.register_dataclass(
    ShearGrid, data_fields=("g1", "g2", "weights", "counts"), meta_fields=(),
)

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
def _bin_shear(x, y, g1, g2, weights, *, shape, gridding="hard", valid=None):
    """Grid weighted shear in edge-based pixel coordinates."""
    ny, nx = shape
    contributions = jnp.stack((g1 * weights, g2 * weights, weights, jnp.ones_like(weights)), axis=-1)
    if valid is not None:
        contributions = jnp.where(valid[:, None], contributions, 0)
        x, y = jnp.where(valid, x, 0), jnp.where(valid, y, 0)
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

@jax.jit
def _project(ra, dec, ra_center, dec_center):
    """Return TAN coordinates and the projection-domain mask."""
    ra, dec = jnp.deg2rad(ra), jnp.deg2rad(dec)
    ra0, dec0 = jnp.deg2rad(ra_center), jnp.deg2rad(dec_center)
    offset = ra - ra0
    cos_distance = jnp.sin(dec0) * jnp.sin(dec) + jnp.cos(dec0) * jnp.cos(dec) * jnp.cos(offset)
    # Rounding can leave a small positive cosine exactly on the horizon
    visible = cos_distance > jnp.finfo(cos_distance.dtype).eps
    denominator = jnp.where(visible, cos_distance, 1)
    x = -jnp.cos(dec) * jnp.sin(offset) / denominator
    y = (jnp.cos(dec0) * jnp.sin(dec) - jnp.sin(dec0) * jnp.cos(dec) * jnp.cos(offset)) / denominator
    return jnp.rad2deg(x), jnp.rad2deg(y), visible

@jax.jit
def _project_catalog(ra, dec, g1, g2, ra_center, dec_center):
    """Project positions and rotate local shear into the map frame."""
    x, y, visible = _project(ra, dec, ra_center, dec_center)
    offset = jnp.deg2rad(ra - ra_center)
    north = jnp.arctan2(jnp.sin(offset) * jnp.sin(jnp.deg2rad(dec_center)), jnp.cos(offset))
    cos_turn, sin_turn = jnp.cos(2 * north), jnp.sin(2 * north)
    return x, y, g1 * cos_turn + g2 * sin_turn, g2 * cos_turn - g1 * sin_turn, visible
