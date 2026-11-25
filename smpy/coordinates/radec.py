"""RA/Dec celestial coordinate system implementation.

Catalogs in RA/Dec are projected onto the plane tangent to the sky at the
field center (the gnomonic, or TAN, projection) and gridded there. The map x
axis points west and y north, so east is to the left as the sky is seen.
"""

import numpy as np

from .base import CoordinateSystem

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
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    ra0, dec0 = np.deg2rad(ra_center), np.deg2rad(dec_center)
    cos_distance = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    if np.any(cos_distance <= 0):
        raise ValueError("Positions 90 degrees or more from the field center cannot be projected onto its "
                         "tangent plane")
    x = -np.cos(dec) * np.sin(ra - ra0) / cos_distance
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_distance
    return np.rad2deg(x), np.rad2deg(y)

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
        transformed_df['coord1_scaled'], transformed_df['coord2_scaled'] = project(ra, dec, ra_center,
                                                                                   dec_center)

        # Angle from the map's y axis to each galaxy's local north
        ra_offset = np.deg2rad(ra - ra_center)
        north = np.arctan2(np.sin(ra_offset) * np.sin(np.deg2rad(dec_center)), np.cos(ra_offset))
        cos_turn, sin_turn = np.cos(2 * north), np.sin(2 * north)
        g1, g2 = data_df['g1'].to_numpy(), data_df['g2'].to_numpy()
        transformed_df['g1'] = g1 * cos_turn + g2 * sin_turn
        transformed_df['g2'] = g2 * cos_turn - g1 * sin_turn
        return transformed_df
