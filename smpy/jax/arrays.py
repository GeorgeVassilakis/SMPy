"""JAX interfaces for gridded shear and fixed-size catalogs."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from smpy.config import Config
from smpy.coordinates.base import ShearGrid

from .aperture_mass import ApertureMassMapper
from .coordinates import _bin_shear, _project_catalog
from .kaiser_squires import KaiserSquiresMapper
from .ks_plus import KSPlusMapper
from .snr import _noise_step, _rotate_shear, _signal_to_noise

METHODS = {
    "kaiser_squires": KaiserSquiresMapper,
    "aperture_mass": ApertureMassMapper,
    "ks_plus": KSPlusMapper,
}

def _asarray(array):
    """Make a JAX array, converting the byte order of host inputs."""
    if isinstance(array, np.ndarray) and not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    return jnp.asarray(array)

class ArrayMassMap(NamedTuple):
    """Mapping results for a fixed-size catalog, as a JAX pytree.

    Parameters
    ----------
    maps : `dict` of `jax.Array`
        E- and B-mode maps keyed by "E" and "B".
    grid : `smpy.coordinates.ShearGrid`
        Weighted mean gridded shear, weight sums and counts.
    snr : `dict` of `jax.Array` or `None`
        Signal-to-noise maps when requested.
    """

    maps: dict[str, jax.Array]
    grid: ShearGrid
    snr: dict[str, jax.Array] | None

class ArrayMapper:
    """Prepare a mapping method for JAX transforms.

    Construct once, then call `map_grid` or `map_catalog` inside a function
    passed to `jax.jit`, `jax.grad` or `jax.vmap`. Set JAX's x64 option before
    creating arrays to work in double precision.

    Parameters
    ----------
    method : `str`
        'kaiser_squires', 'ks_plus' or 'aperture_mass'.
    **method_settings
        Static method settings as in `smpy.map_mass`. Settings that control
        filter support also bound the support for traced overrides.
    """

    def __init__(self, method="kaiser_squires", **method_settings):
        config = Config.from_defaults(method, **method_settings).to_dict()
        method = config["general"]["method"]
        settings = config["methods"][method]
        unknown = method_settings.keys() - settings.keys()
        if unknown:
            raise ValueError(f"Unknown {method} setting {sorted(unknown)[0]!r}")
        self._mapper = METHODS[method](**settings)

    def map_grid(self, g1, g2, weights=None, **parameters):
        """Map binned shear on a regular grid.

        Parameters
        ----------
        g1, g2 : array-like
            Real shear component arrays with shape (ny, nx).
        weights : array-like, optional
            Pixel weights with the same shape, used by KS+ to distinguish
            observed shear from gaps. Defaults to an observed full grid.
        **parameters
            Traced scalar overrides. KS and KS+ accept ``smoothing`` between
            zero and the configured sigma in pixels. Aperture mass accepts
            positive ``filter_scale`` and ``truncation`` whose product is at
            most the product of the two configured values.

        Returns
        -------
        maps : `dict` of `jax.Array`
            E- and B-mode maps keyed by "E" and "B".
        """
        g1, g2 = _asarray(g1), _asarray(g2)
        dtype = jnp.result_type(g1, g2, float)
        if g1.ndim != 2 or min(g1.shape) < 1 or g2.shape != g1.shape:
            raise ValueError("g1 and g2 must have the same nonempty two-dimensional shape")
        weights = jnp.ones_like(g1, dtype=dtype) if weights is None else _asarray(weights)
        if weights.shape != g1.shape:
            raise ValueError("weights must have the same shape as g1 and g2")
        grid = ShearGrid(g1.astype(dtype), g2.astype(dtype), weights, None)
        return dict(zip("EB", self._mapper.create_maps(grid, **parameters)))

    def map_catalog(self, coord1, coord2, g1, g2, weights=None, *, shape,
                    origin=(0.0, 0.0), pixel_size=1.0, tangent_point=None,
                    gridding="hard", key=None, num_shuffles=0,
                    shuffle_type="orientation", **parameters):
        """Bin a fixed-size catalog and map it on a specified grid.

        Parameters
        ----------
        coord1, coord2 : array-like
            One-dimensional x and y coordinates. With ``tangent_point``,
            these are RA and Dec in degrees on its visible hemisphere.
        g1, g2 : array-like
            One-dimensional, finite shear components in the map frame or,
            with ``tangent_point``, local west/north frames.
        weights : array-like, optional
            Finite per-galaxy weights. Defaults to unit weights.
        shape : `tuple` of `int`
            Static shape (ny, nx) of the map.
        origin : `tuple` of scalar, optional
            Lower x and y edges in the gridding coordinates.
        pixel_size : scalar, optional
            Positive width of a pixel in the gridding coordinates (degrees
            on the tangent plane for sky coordinates).
        tangent_point : `tuple` of scalar, optional
            TAN projection center (RA, Dec), in degrees. The projected
            x axis points west and the y axis points north.
        gridding : `str`, optional
            'hard' assigns galaxies to the containing pixel, as in
            `smpy.map_mass`. 'bilinear' deposits among adjacent pixel
            centers. Deposits at the map edge are clamped.
        key : `jax.Array`, optional
            JAX random key for the null maps, required with shuffles.
        num_shuffles : `int`, optional
            Static number of null maps. Use zero or at least two.
        shuffle_type : `str`, optional
            'orientation' rotates each shear. 'spatial' permutes projected
            positions while keeping shear and weight paired.
        **parameters
            Traced scalar overrides as in `map_grid`.

        Returns
        -------
        result : `ArrayMassMap`
            Device-resident E and B maps, gridded inputs and optional SNR.

        Notes
        -----
        A galaxy outside the grid, at a non-finite position, or 90 degrees
        or more from the tangent point adds nothing to the shear, weights
        or counts. A galaxy exactly on an upper grid edge goes in the last
        pixel, as in `smpy.map_mass`.

        Bilinear gridding makes position derivatives available within each
        interval between pixel centers. With hard bins, the bin indices
        are piecewise constant. Sky-frame shear rotation can still depend
        on position. Map dimensions and random choices are discrete.
        Derivatives of SNR hold the random key fixed and are defined only
        where the sample variance is positive.
        """
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("shape must be a pair of positive grid dimensions")
        if gridding not in ("hard", "bilinear"):
            raise ValueError("gridding must be 'hard' or 'bilinear'")
        if shuffle_type not in ("orientation", "spatial"):
            raise ValueError("shuffle_type must be 'orientation' or 'spatial'")
        if num_shuffles != 0 and (num_shuffles < 2 or key is None):
            raise ValueError("SNR requires a key and at least two shuffles")
        x, y, g1, g2 = (_asarray(column) for column in (coord1, coord2, g1, g2))
        weights = jnp.ones_like(g1) if weights is None else _asarray(weights)
        if g1.ndim != 1 or any(column.shape != g1.shape for column in (x, y, g2, weights)):
            raise ValueError("Catalog coordinates, shears and weights must have matching 1D shapes")
        dtype = jnp.result_type(g1, g2, weights, float)
        g1, g2, weights = (column.astype(dtype) for column in (g1, g2, weights))
        valid = jnp.isfinite(x) & jnp.isfinite(y)
        # A finite stand-in keeps the gradients of the other galaxies finite
        stand_in = origin if tangent_point is None else tangent_point
        x, y = jnp.where(valid, x, stand_in[0]), jnp.where(valid, y, stand_in[1])
        if tangent_point is not None:
            x, y, g1, g2, visible = _project_catalog(x, y, g1, g2, *tangent_point)
            valid &= visible
        x, y = (x - origin[0]) / pixel_size, (y - origin[1]) / pixel_size
        ny, nx = shape
        valid &= (x >= 0) & (x <= nx) & (y >= 0) & (y <= ny)

        grid = _bin_shear(x, y, g1, g2, weights, shape=shape, gridding=gridding, valid=valid)
        mapped = self._mapper.create_maps(grid, **parameters)
        snr = None
        if num_shuffles:
            snr = self._catalog_snr(x, y, g1, g2, weights, valid, shape, gridding,
                                    mapped, num_shuffles, shuffle_type, key, parameters)
        return ArrayMassMap(dict(zip("EB", mapped)), grid, snr)

    def _catalog_snr(self, x, y, g1, g2, weights, valid, shape, gridding,
                     mapped, count, shuffle_type, key, parameters):
        """Compute null maps and accumulate their sample variance."""
        signal = jnp.stack(mapped)
        ranks = jnp.maximum(jnp.cumsum(valid) - 1, 0)

        @jax.checkpoint
        def trial(index):
            trial_key = jax.random.fold_in(key, index)
            if shuffle_type == "orientation":
                angle = jax.random.uniform(trial_key, g1.shape, dtype=g1.dtype, maxval=2 * jnp.pi)
                rotated = _rotate_shear(g1, g2, angle)
                grid = _bin_shear(x, y, *rotated, weights, shape=shape, gridding=gridding, valid=valid)
            else:
                perm = jax.random.permutation(trial_key, len(g1))
                # Pair valid shear rows with shuffled valid positions.
                perm = perm[jnp.argsort(~valid[perm], stable=True)][ranks]
                grid = _bin_shear(x[perm], y[perm], g1, g2, weights,
                                  shape=shape, gridding=gridding, valid=valid)
            return jnp.stack(self._mapper.create_maps(grid, **parameters))

        def accumulate(index, state):
            return _noise_step(*state, trial(index), index + 1)

        zero = jnp.zeros_like(signal)
        _, m2 = jax.lax.fori_loop(0, count, accumulate, (zero, zero))
        return dict(zip("EB", _signal_to_noise(signal, m2, count)))
