"""High-level Python interface for SMPy."""

from smpy.config import Config
from smpy.pipeline import compute

def map_mass(catalog, method=None, *, coord_system=None, pixel_scale=None, downsample_factor=None,
             coord1_col=None, coord2_col=None, g1_col=None, g2_col=None, weight_col=None,
             input_hdu=None, create_snr=None, snr=None, **method_settings):
    """Make E- and B-mode mass maps from a shear catalog.

    Settings left as `None` take their values from ``default.yaml``; defaults
    are given in brackets below. Nothing is written to disk. Use the returned
    `MassMap` to plot or save the maps.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Shear catalog, or the path to a FITS file.
    method : `str`, optional
        'kaiser_squires' [default], 'aperture_mass', or 'ks_plus'.
    coord_system : `str`, optional
        'radec' [default] or 'pixel'.
    pixel_scale : `float`, optional
        Map pixel size in arcminutes. Required for 'radec'.
    downsample_factor : `int`, optional
        Catalog pixels per map pixel. Required for 'pixel'.
    coord1_col, coord2_col : `str`, optional
        Coordinate columns ['ra' and 'dec', in degrees, for 'radec';
        'X_IMAGE' and 'Y_IMAGE' for 'pixel'].
    g1_col, g2_col : `str`, optional
        Shear columns ['g1' and 'g2'].
    weight_col : `str`, optional
        Weight column [none, for unit weights].
    input_hdu : `int`, optional
        FITS extension holding the catalog [1].
    create_snr : `bool`, optional
        Also compute signal-to-noise maps [False].
    snr : `dict`, optional
        Overrides for the ``snr`` settings, such as ``{"num_shuffles": 200}``.
    **method_settings
        Settings of the chosen method from the ``methods`` section of
        ``default.yaml``, such as ``smoothing=1.5`` or
        ``inpainting_iterations=50``.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used.

    Raises
    ------
    ValueError
        Raised if a setting does not exist, is invalid, or is required but
        missing.

    Examples
    --------
    >>> result = map_mass("catalog.fits", pixel_scale=0.4)
    >>> fig, ax, im = result.plot("E")
    """
    config = Config.from_defaults(
        method, coord_system=coord_system, pixel_scale=pixel_scale, downsample_factor=downsample_factor,
        coord1_col=coord1_col, coord2_col=coord2_col, g1_col=g1_col, g2_col=g2_col, weight_col=weight_col,
        input_hdu=input_hdu, create_snr=create_snr, snr=snr, **method_settings)
    config.validate()
    return compute(catalog, config.resolved())

def map_kaiser_squires(catalog, **settings):
    """Make mass maps with the Kaiser-Squires inversion.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Shear catalog, or the path to a FITS file.
    **settings
        Arguments of `map_mass`. The method setting is ``smoothing``.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used.
    """
    return map_mass(catalog, "kaiser_squires", **settings)

def map_aperture_mass(catalog, **settings):
    """Make aperture mass maps.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Shear catalog, or the path to a FITS file.
    **settings
        Arguments of `map_mass`. The method settings are ``filter_type``,
        ``filter_scale``, ``truncation`` and ``ell``.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used.
    """
    return map_mass(catalog, "aperture_mass", **settings)

def map_ks_plus(catalog, **settings):
    """Make mass maps with the KS+ inversion.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Shear catalog, or the path to a FITS file.
    **settings
        Arguments of `map_mass`. The method settings are ``smoothing``,
        ``inpainting_iterations``, ``reduced_shear_iterations``, ``nscales``,
        ``extension_size`` and ``use_wavelet_constraints``.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used.
    """
    return map_mass(catalog, "ks_plus", **settings)
