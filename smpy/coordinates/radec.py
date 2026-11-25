"""RA/Dec celestial coordinate system implementation.

Catalogs in RA/Dec are projected onto the plane tangent to the sky at the
field center (the gnomonic, or TAN, projection) and gridded there. The map x
axis points west and y north, so east is to the left as the sky is seen.
"""

import jax
import jax.numpy as jnp
import numpy as np

from smpy._precision import enable_x64, to_numpy

from .base import CoordinateSystem

@jax.jit
def _project(ra, dec, ra_center, dec_center):
    """Return TAN coordinates and the projection-domain mask."""
    ra, dec = jnp.deg2rad(ra), jnp.deg2rad(dec)
    ra0, dec0 = jnp.deg2rad(ra_center), jnp.deg2rad(dec_center)
    offset = ra - ra0
    cos_distance = jnp.sin(dec0) * jnp.sin(dec) + jnp.cos(dec0) * jnp.cos(dec) * jnp.cos(offset)
    visible = cos_distance > 0
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

def _check_projection(visible):
    """Reject catalog positions outside the tangent hemisphere."""
    if not np.all(visible):
        raise ValueError("Positions 90 degrees or more from the field center cannot be projected onto its "
                         "tangent plane")

def project(ra, dec, ra_center, dec_center):
    """Project sky positions onto the plane tangent at the field center.

    Parameters
    ----------
    ra, dec : `float` or `numpy.ndarray`
        Sky positions in degrees.
    ra_center, dec_center : `float`
        Tangent point in degrees.

    Returns
    -------
    x, y : `float` or `numpy.ndarray`
        Gnomonic plane coordinates in degrees, x towards the west and y
        towards the north.

    Raises
    ------
    ValueError
        Raised if a position is 90 degrees or more from the tangent point,
        where the projection is undefined.
    """
    with enable_x64(True):
        ra, dec = np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)
        x, y, visible = to_numpy(_project(ra, dec, ra_center, dec_center))
    _check_projection(visible)
    return x, y

def unproject(x, y, ra_center, dec_center):
    """Return the sky positions of points on the plane tangent at the field
    center; the inverse of `project`.

    Parameters
    ----------
    x, y : `float` or `numpy.ndarray`
        Gnomonic plane coordinates in degrees, x towards the west and y
        towards the north.
    ra_center, dec_center : `float`
        Tangent point in degrees.

    Returns
    -------
    ra, dec : `float` or `numpy.ndarray`
        Sky positions in degrees, with RA in [0, 360).
    """
    xi, eta = -np.deg2rad(x), np.deg2rad(y)
    ra0, dec0 = np.deg2rad(ra_center), np.deg2rad(dec_center)
    denominator = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, denominator)
    dec = np.arctan2(np.sin(dec0) + eta * np.cos(dec0), np.hypot(xi, denominator))
    return np.rad2deg(ra) % 360, np.rad2deg(dec)

def _sky_coordinates(data_df):
    """Return RA without the jump at 0/360, Dec, and the field center.

    The field center is the middle of the RA and Dec ranges.
    """
    ra, dec = data_df['coord1'].to_numpy(dtype=float), data_df['coord2'].to_numpy(dtype=float)
    ra = ra[0] + (ra - ra[0] + 180) % 360 - 180
    return ra, dec, (ra.min() + ra.max()) / 2, (dec.min() + dec.max()) / 2

class RADecSystem(CoordinateSystem):
    """RA/Dec coordinates, gridded on the plane tangent at the field center.

    The catalog's g1 and g2 are taken in each galaxy's local frame, with x
    towards the west and y towards the north. They are turned into the map
    frame, whose y axis points north only along the central meridian.
    """

    def calculate_boundaries(self, data_df, settings):
        """Calculate the map grid around a catalog.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            Catalog from `transform_coordinates`.
        settings : `dict`
            The ``general.radec`` settings, including ``pixel_scale`` in
            arcminutes.

        Returns
        -------
        scaled_boundaries : `dict`
            Map edges on the tangent plane, in degrees, and ``pixel_size``.
        true_boundaries : `dict`
            RA and Dec ranges of the catalog and the field center
            (``coord1_center``, ``coord2_center``), in degrees. RA may run
            past 360 or below 0 for fields that cross RA = 0.
        """
        scaled_boundaries = self._grid_edges(data_df, settings['pixel_scale'] / 60)
        scaled_boundaries.update(coord1_name='Projected RA', coord2_name='Projected Dec', units='deg')
        ra, dec, ra_center, dec_center = _sky_coordinates(data_df)
        true_boundaries = {
            'coord1_min': ra.min(),
            'coord1_max': ra.max(),
            'coord2_min': dec.min(),
            'coord2_max': dec.max(),
            'coord1_center': ra_center,
            'coord2_center': dec_center,
            'coord1_name': 'RA',
            'coord2_name': 'Dec',
            'units': 'deg'
        }
        return scaled_boundaries, true_boundaries

    def transform_coordinates(self, data_df):
        """Project the catalog onto the tangent plane and turn its shear into
        the map frame.

        Parameters
        ----------
        data_df : `pandas.DataFrame`
            DataFrame with coord1 (RA), coord2 (Dec), g1 and g2 columns.

        Returns
        -------
        transformed_df : `pandas.DataFrame`
            Copy with tangent-plane coordinates in coord1_scaled and
            coord2_scaled, and g1 and g2 in the map frame.
        """
        transformed_df = data_df.copy()
        ra, dec, ra_center, dec_center = _sky_coordinates(data_df)
        g1, g2 = (data_df[key].to_numpy(dtype=np.float64) for key in ("g1", "g2"))
        with enable_x64(True):
            x, y, g1, g2, visible = jax.device_get(_project_catalog(ra, dec, g1, g2, ra_center, dec_center))
        _check_projection(visible)
        transformed_df['coord1_scaled'], transformed_df['coord2_scaled'] = x, y
        transformed_df['g1'], transformed_df['g2'] = g1, g2
        return transformed_df
