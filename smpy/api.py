"""High-level API for SMPy mass mapping.

This module provides user-friendly functions for performing mass mapping
operations without requiring detailed YAML configuration knowledge.
"""

from .config import Config


def map_mass(
    data,
    method='kaiser_squires',
    coord_system='radec',
    pixel_scale=None,
    downsample_factor=None,
    output_dir='.',
    output_base_name='smpy_output',
    g1_col='g1',
    g2_col='g2',
    weight_col=None,
    mode='E',
    create_snr=False,
    create_counts_map=False,
    overlay_counts_map=False,
    save_fits=False,
    print_timing=False,
    smoothing=None,
    **kwargs
):
    """Perform mass mapping using the specified method.

    This is the main entry point for SMPy mass mapping operations.
    It provides a simple interface while maintaining access to all
    underlying functionality.

    Parameters
    ----------
    data : `str` or `pathlib.Path`
        Path to input FITS file containing shear catalog.
    method : `str`, optional
        Mass mapping method to use. Options: 'kaiser_squires',
        'aperture_mass', 'ks_plus'.
    coord_system : `str`, optional
        Coordinate system of input data. Options: 'radec', 'pixel'.
    pixel_scale : `float`, optional
        Pixel scale in arcminutes. Required for 'radec' coordinate system.
    downsample_factor : `int`, optional
        Downsampling factor for pixel coordinates. Required for 'pixel'
        coordinate system.
    output_dir : `str`, optional
        Directory to save output files.
    output_base_name : `str`, optional
        Base name for output files.
    g1_col : `str`, optional
        Column name for first shear component.
    g2_col : `str`, optional
        Column name for second shear component.
    weight_col : `str`, optional
        Column name for weights. If None, no weights are used.
    mode : `str` or `list`, optional
        Shear mode(s) to compute. Options: 'E', 'B', or ['E', 'B'].
    create_snr : `bool`, optional
        Whether to create signal-to-noise ratio map.
    create_counts_map : `bool`, optional
        Whether to create and save a per-pixel counts map PNG.
    overlay_counts_map : `bool`, optional
        Whether to overlay per-pixel counts as text on the convergence map plots.
    save_fits : `bool`, optional
        Whether to save maps as FITS files.
    print_timing : `bool`, optional
        Whether to print timing information.
    smoothing : `float`, optional
        Smoothing scale in pixels. If None, uses method default.
    **kwargs
        Additional method-specific parameters.

    Returns
    -------
    result : `dict`
        Dictionary containing the computed mass maps.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.
    FileNotFoundError
        If input file does not exist.

    Examples
    --------
    Basic usage with minimal parameters:

    >>> result = map_mass(
    ...     data='catalog.fits',
    ...     coord_system='radec',
    ...     pixel_scale=0.168
    ... )

    More advanced usage with custom parameters:

    >>> result = map_mass(
    ...     data='catalog.fits',
    ...     method='ks_plus',
    ...     coord_system='radec',
    ...     pixel_scale=0.168,
    ...     smoothing=1.5,
    ...     create_snr=True,
    ...     mode=['E', 'B']
    ... )
    """
    # Load default configuration for the method
    config = Config.from_defaults(method)
    
    # Update config with user parameters
    config.update_from_kwargs(
        data=data,
        coord_system=coord_system,
        pixel_scale=pixel_scale,
        downsample_factor=downsample_factor,
        method=method,
        output_dir=output_dir,
        output_base_name=output_base_name,
        g1_col=g1_col,
        g2_col=g2_col,
        weight_col=weight_col,
        mode=mode,
        create_snr=create_snr,
        create_counts_map=create_counts_map,
        overlay_counts_map=overlay_counts_map,
        save_fits=save_fits,
        print_timing=print_timing,
        smoothing=smoothing,
        **kwargs
    )
    
    # Validate configuration structure
    config.validate()
    
    # Execute the mapping
    from .run import run
    return run(config)


def map_kaiser_squires(
    data,
    coord_system='radec',
    pixel_scale=None,
    downsample_factor=None,
    smoothing=2.0,
    **kwargs
):
    """Perform Kaiser-Squires mass mapping.

    This function provides direct access to the Kaiser-Squires method
    with sensible defaults for quick analysis.

    Parameters
    ----------
    data : `str` or `pathlib.Path`
        Path to input FITS file containing shear catalog.
    coord_system : `str`, optional
        Coordinate system of input data. Options: 'radec', 'pixel'.
    pixel_scale : `float`, optional
        Pixel scale in arcminutes. Required for 'radec' coordinate system.
    downsample_factor : `int`, optional
        Downsampling factor for pixel coordinates. Required for 'pixel'
        coordinate system.
    smoothing : `float`, optional
        Gaussian smoothing scale in pixels.
    **kwargs
        Additional parameters passed to map_mass().

    Returns
    -------
    result : `dict`
        Dictionary containing the computed mass maps.

    Examples
    --------
    >>> result = map_kaiser_squires(
    ...     data='catalog.fits',
    ...     coord_system='radec',
    ...     pixel_scale=0.168
    ... )
    """
    return map_mass(
        data=data,
        method='kaiser_squires',
        coord_system=coord_system,
        pixel_scale=pixel_scale,
        downsample_factor=downsample_factor,
        smoothing=smoothing,
        **kwargs
    )


def map_aperture_mass(
    data,
    coord_system='radec',
    pixel_scale=None,
    downsample_factor=None,
    filter_type='schirmer',
    filter_scale=60,
    **kwargs
):
    """Perform aperture mass mapping.

    This function provides direct access to the aperture mass method
    with sensible defaults for quick analysis.

    Parameters
    ----------
    data : `str` or `pathlib.Path`
        Path to input FITS file containing shear catalog.
    coord_system : `str`, optional
        Coordinate system of input data. Options: 'radec', 'pixel'.
    pixel_scale : `float`, optional
        Pixel scale in arcminutes. Required for 'radec' coordinate system.
    downsample_factor : `int`, optional
        Downsampling factor for pixel coordinates. Required for 'pixel'
        coordinate system.
    filter_type : `str`, optional
        Type of aperture filter. Options: 'schirmer', 'schneider'.
    filter_scale : `int`, optional
        Filter scale in pixels.
    **kwargs
        Additional parameters passed to map_mass().

    Returns
    -------
    result : `dict`
        Dictionary containing the computed mass maps.

    Examples
    --------
    >>> result = map_aperture_mass(
    ...     data='catalog.fits',
    ...     coord_system='radec',
    ...     pixel_scale=0.168,
    ...     filter_scale=80
    ... )
    """
    # Handle aperture mass specific parameters
    if 'filter' not in kwargs:
        kwargs['filter'] = {}
    if isinstance(kwargs['filter'], dict):
        kwargs['filter']['type'] = filter_type
        kwargs['filter']['scale'] = filter_scale
    
    return map_mass(
        data=data,
        method='aperture_mass',
        coord_system=coord_system,
        pixel_scale=pixel_scale,
        downsample_factor=downsample_factor,
        **kwargs
    )


def map_ks_plus(
    data,
    coord_system='radec',
    pixel_scale=None,
    downsample_factor=None,
    smoothing=2.0,
    inpainting_iterations=100,
    reduced_shear_iterations=3,
    **kwargs
):
    """Perform KS+ mass mapping.

    This function provides direct access to the KS+ method
    with sensible defaults for quick analysis.

    Parameters
    ----------
    data : `str` or `pathlib.Path`
        Path to input FITS file containing shear catalog.
    coord_system : `str`, optional
        Coordinate system of input data. Options: 'radec', 'pixel'.
    pixel_scale : `float`, optional
        Pixel scale in arcminutes. Required for 'radec' coordinate system.
    downsample_factor : `int`, optional
        Downsampling factor for pixel coordinates. Required for 'pixel'
        coordinate system.
    smoothing : `float`, optional
        Gaussian smoothing scale in pixels.
    inpainting_iterations : `int`, optional
        Number of iterations for inpainting algorithm.
    reduced_shear_iterations : `int`, optional
        Number of iterations for reduced shear correction.
    **kwargs
        Additional parameters passed to map_mass().

    Returns
    -------
    result : `dict`
        Dictionary containing the computed mass maps.

    Examples
    --------
    >>> result = map_ks_plus(
    ...     data='catalog.fits',
    ...     coord_system='radec',
    ...     pixel_scale=0.168,
    ...     inpainting_iterations=200
    ... )
    """
    # Handle KS+ specific parameters
    kwargs['inpainting_iterations'] = inpainting_iterations
    kwargs['reduced_shear_iterations'] = reduced_shear_iterations
    
    return map_mass(
        data=data,
        method='ks_plus',
        coord_system=coord_system,
        pixel_scale=pixel_scale,
        downsample_factor=downsample_factor,
        smoothing=smoothing,
        **kwargs
    )
