"""Plotting utilities for mass mapping visualization.

This module provides comprehensive plotting functions for convergence and
SNR maps, supporting both RA/Dec and pixel coordinate systems. Includes
advanced features like peak detection visualization, center marking, and
configurable scaling options.
"""

from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from smpy.utils import find_peaks2d

def _get_center_coordinates(cluster_center, scaled_boundaries, true_boundaries, coord_system_type):
    """Get center coordinates for plotting.

    Process center specification and convert to scaled coordinates for
    plotting markers on mass maps.

    Parameters
    ----------
    cluster_center : `str` or `dict` or None
        Center specification:
        - 'auto': use field center
        - dict: specific coordinates with appropriate keys
        - None: no center
    scaled_boundaries : `dict`
        Scaled coordinate boundaries.
    true_boundaries : `dict`
        True coordinate boundaries.
    coord_system_type : `str`
        Coordinate system type ('radec' or 'pixel').

    Returns
    -------
    center_coord1 : `float` or None
        First coordinate of center in scaled system.
    center_coord2 : `float` or None
        Second coordinate of center in scaled system.

    Notes
    -----
    For RA/Dec systems, expects 'ra_center' and 'dec_center' keys.
    For pixel systems, expects 'x_center' and 'y_center' keys.
    """

    if cluster_center is None:
        return None, None
        
    if cluster_center == 'auto':
        # For both systems, use the center of the scaled boundaries
        center_coord1 = (scaled_boundaries['coord1_max'] + scaled_boundaries['coord1_min']) / 2
        center_coord2 = (scaled_boundaries['coord2_max'] + scaled_boundaries['coord2_min']) / 2
        return center_coord1, center_coord2
        
    if isinstance(cluster_center, dict):
        # Handle both coordinate systems
        if coord_system_type == 'radec':
            key1, key2 = 'ra_center', 'dec_center'
        else:  # pixel
            key1, key2 = 'x_center', 'y_center'
            
        if key1 not in cluster_center or key2 not in cluster_center:
            print(f"Warning: Expected {key1} and {key2} in cluster_center dictionary")
            return None, None
            
        # Convert from true to scaled coordinates
        center_coord1 = np.interp(
            cluster_center[key1],
            [true_boundaries['coord1_min'], true_boundaries['coord1_max']],
            [scaled_boundaries['coord1_min'], scaled_boundaries['coord1_max']]
        )
        center_coord2 = np.interp(
            cluster_center[key2],
            [true_boundaries['coord2_min'], true_boundaries['coord2_max']],
            [scaled_boundaries['coord2_min'], scaled_boundaries['coord2_max']]
        )
        return center_coord1, center_coord2
        
    print("Warning: Unrecognized cluster_center format")
    return None, None

def _create_normalization(scaling, data, vmin=None, vmax=None, map_type='convergence'):
    """Create normalization object based on scaling configuration.

    Generate appropriate matplotlib normalization for colormap scaling
    supporting linear, power-law, and symmetric logarithmic scaling.

    Parameters
    ----------
    scaling : `dict` or `str` or None
        Scaling configuration:
        - None: linear scaling with vmin/vmax
        - 'linear': linear scaling
        - 'power': power-law scaling with gamma parameter
        - 'symlog': symmetric logarithmic scaling
        - dict with type and parameters
    data : `numpy.ndarray`
        Data array for percentile-based scaling.
    vmin : `float` or None, optional
        Minimum value for scaling.
    vmax : `float` or None, optional
        Maximum value for scaling.
    map_type : `str`, optional
        Type of map ('convergence' or 'snr') for map-specific parameters.

    Returns
    -------
    norm : `matplotlib.colors.Normalize`
        Normalization object for colormap scaling.

    Notes
    -----
    Supports percentile-based vmin/vmax calculation and map-specific
    parameters for symlog scaling (e.g., different thresholds for
    convergence vs SNR maps).
    """
    if scaling is None:
        return colors.Normalize(vmin=vmin, vmax=vmax)
        
    # Handle string shortcuts
    if isinstance(scaling, str):
        scaling = {'type': scaling}
        
    scale_type = scaling.get('type', 'linear')
    
    # Process percentile-based min/max
    percentile = scaling.get('percentile')
    if percentile is not None:
        # If percentile is a list or tuple with two values, use them for vmin and vmax
        if isinstance(percentile, (list, tuple)) and len(percentile) == 2:
            vmin = np.percentile(data, percentile[0])
            vmax = np.percentile(data, percentile[1])
        else:
            print(f"Warning: 'percentile' should be a list or tuple with two values [min, max]")
    
    # Create normalizer based on type
    if scale_type == 'linear':
        return colors.Normalize(vmin=vmin, vmax=vmax)
    
    elif scale_type == 'power':
        # Get parameters
        gamma = scaling.get('gamma', 1.0)
        return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)
        
    elif scale_type == 'symlog':
        # Check for map-specific parameters
        map_specific_params = scaling.get(map_type, {})
        
        # Get parameters with map-specific overrides if they exist
        linthresh = map_specific_params.get('linthresh', scaling.get('linthresh', 0.1))
        linscale = map_specific_params.get('linscale', scaling.get('linscale', 1.0))
        
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
        
    else:
        print(f"Warning: Unknown scaling type '{scale_type}', falling back to linear")
        return colors.Normalize(vmin=vmin, vmax=vmax)

def plot_convergence(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name, map_type='convergence'):
    """Create plot of convergence map.

    Generate publication-quality plots of mass maps with configurable
    styling, coordinate systems, peak detection, and center marking.

    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence or SNR map data.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries for plotting extent.
    true_boundaries : `dict`
        True coordinate boundaries for tick labels.
    config : `dict`
        Plot configuration settings including figsize, cmap, scaling options.
    output_name : `str`
        Path for saving the plot file.
    map_type : `str`, optional
        Type of map ('convergence' or 'snr') for appropriate labeling.

    Notes
    -----
    Automatically detects coordinate system from config and delegates to
    appropriate plotting function. Supports both RA/Dec and pixel coordinate
    systems with different formatting and tick handling.
    """

    # Check coordinate system
    coord_system = config.get('coordinate_system', 'radec').lower()
    
    if coord_system == 'radec':
        _plot_convergence_radec(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name, map_type)
    else:
        _plot_convergence_pixel(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name, map_type)

def _set_plot_params(fontsize=15):
    """Set standard matplotlib parameters.

    Configure matplotlib parameters for consistent, publication-quality
    plot appearance with appropriate tick sizes and styling.

    Parameters
    ----------
    fontsize : `int`, optional
        Base font size for plot elements.

    Notes
    -----
    Sets tick parameters, axis line widths, and font sizes for consistent
    appearance across all mass mapping plots.
    """
    plt.rcParams.update({
        'axes.linewidth': 1.3,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'xtick.major.size': 8,
        'xtick.major.width': 1.3,
        'xtick.minor.visible': True,
        'xtick.minor.width': 1.,
        'xtick.minor.size': 6,
        'xtick.direction': 'in',
        'ytick.major.width': 1.3,
        'ytick.major.size': 8,
        'ytick.minor.visible': True,
        'ytick.minor.width': 1.,
        'ytick.minor.size': 6,
        'ytick.direction': 'in',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize
    })

def _plot_convergence_pixel(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name, map_type='convergence'):
    """Create convergence plot for pixel coordinates.

    Generate mass map plots using pixel coordinate system with appropriate
    axis labels, peak detection, and center marking.

    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence or SNR map data.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries for plotting extent.
    true_boundaries : `dict`
        True coordinate boundaries for peak detection.
    config : `dict`
        Plot configuration settings.
    output_name : `str`
        Path for saving the plot file.
    map_type : `str`, optional
        Type of map ('convergence' or 'snr') for appropriate labeling.

    Notes
    -----
    Handles pixel coordinate specific formatting including axis labels
    in pixel units and appropriate peak coordinate conversions.
    """

    _set_plot_params()
    
    # Create figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, 
        figsize=config['figsize']
    )
    
    # Create normalization for the colormap based on scaling config
    norm = _create_normalization(
        config.get('scaling'),
        filtered_convergence,
        vmin=config.get('vmin'),
        vmax=config.get('vmax'),
        map_type=map_type
    )
    
    # Plot convergence map
    im = ax.imshow(
        filtered_convergence,
        cmap=config['cmap'],
        norm=norm,
        extent=[scaled_boundaries['coord1_min'],
                scaled_boundaries['coord1_max'],
                scaled_boundaries['coord2_min'],
                scaled_boundaries['coord2_max']],
        origin='lower'
    )
    
    # Handle center marking
    center_x, center_y = _get_center_coordinates(
        config.get('cluster_center'),
        scaled_boundaries,
        true_boundaries,
        'pixel')
    
    if center_x is not None:
        ax.plot(center_x, center_y, 'rx', markersize=10)

    # Add peaks if threshold specified
    threshold = config.get('threshold')
    if threshold is not None:
        # Only show verbose peak information for SNR maps
        verbose_peaks = config.get('verbose', False) and map_type.lower() == 'snr'
        X, Y, _, _ = find_peaks2d(filtered_convergence, 
                                           threshold=threshold,
                                           verbose=verbose_peaks,
                                           true_boundaries=true_boundaries,
                                           scaled_boundaries=scaled_boundaries)
        # Convert peak indices to pixel coordinates
        peak_x = [scaled_boundaries['coord1_min'] + 
                 (x + 0.5) * (scaled_boundaries['coord1_max'] - scaled_boundaries['coord1_min']) / 
                 filtered_convergence.shape[1] for x in X]
        peak_y = [scaled_boundaries['coord2_min'] + 
                 (y + 0.5) * (scaled_boundaries['coord2_max'] - scaled_boundaries['coord2_min']) / 
                 filtered_convergence.shape[0] for y in Y]
        ax.scatter(peak_x, peak_y, s=100, facecolors='none', edgecolors='g', linewidth=1.5)
    
    # Set labels
    xlabel = config.get('xlabel')
    ylabel = config.get('ylabel')

    if xlabel == 'auto':
        ax.set_xlabel('X (pixels)')
    elif xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel == 'auto':
        ax.set_ylabel('Y (pixels)')
    elif ylabel is not None:
        ax.set_ylabel(ylabel)
    
    ax.set_title(config.get('plot_title', ''))
    
    # Add grid if requested
    if config.get('gridlines', False):
        ax.grid(color='black')
    
    # Add colorbar
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)
    
    # Save figure
    if output_name:
        fig.savefig(output_name)
        map_label = "Convergence" if map_type.lower() == "convergence" else "SNR"
        print(f"{map_label} map saved as PNG file: {output_name}")
    
    plt.close(fig)

def _plot_convergence_radec(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name, map_type='convergence'):
    """Create convergence plot for RA/Dec coordinates.

    Generate mass map plots using celestial coordinate system with proper
    axis formatting, tick spacing, and astronomical coordinate conventions.

    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence or SNR map data.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries accounting for cos(Dec) projection.
    true_boundaries : `dict`
        True celestial coordinate boundaries for tick labels.
    config : `dict`
        Plot configuration settings.
    output_name : `str`
        Path for saving the plot file.
    map_type : `str`, optional
        Type of map ('convergence' or 'snr') for appropriate labeling.

    Notes
    -----
    Handles celestial coordinate specific formatting including automatic
    tick spacing calculation, coordinate conversion for peaks, and proper
    axis orientation (RA increases leftward).
    """

    # Set plotting parameters
    _set_plot_params()

    # Create figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    # Create normalization for the colormap based on scaling config
    norm = _create_normalization(
        config.get('scaling'),
        filtered_convergence,
        vmin=config.get('vmin'),
        vmax=config.get('vmax'),
        map_type=map_type
    )
    
    # Plot convergence map
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        norm=norm,
        extent=[scaled_boundaries['coord1_max'], 
                scaled_boundaries['coord1_min'], 
                scaled_boundaries['coord2_min'], 
                scaled_boundaries['coord2_max']],
        origin='lower'
    )
    
    # Plot peaks if threshold specified
    threshold = config.get('threshold')
    if threshold is not None:
        # Only show verbose peak information for SNR maps
        verbose_peaks = config.get('verbose', False) and map_type.lower() == 'snr'
        X, Y, _, _ = find_peaks2d(filtered_convergence, 
                                           threshold=threshold,
                                           verbose=verbose_peaks,
                                           true_boundaries=true_boundaries,
                                           scaled_boundaries=scaled_boundaries)
        
        # Convert peak indices to RA/Dec coordinates
        ra_peaks = [scaled_boundaries['coord1_min'] + 
                   (x + 0.5) * (scaled_boundaries['coord1_max'] - scaled_boundaries['coord1_min']) / 
                   filtered_convergence.shape[1] for x in X]
        dec_peaks = [scaled_boundaries['coord2_min'] + 
                    (y + 0.5) * (scaled_boundaries['coord2_max'] - scaled_boundaries['coord2_min']) / 
                    filtered_convergence.shape[0] for y in Y]
        
        ax.scatter(ra_peaks, dec_peaks, s=100, facecolors='none', edgecolors='g', linewidth=1.5)

    # Handle center marking
    ra_center, dec_center = _get_center_coordinates(
        config.get('cluster_center'),
        scaled_boundaries,
        true_boundaries,
        'radec'
    )
    
    if ra_center is not None:
        ax.plot(ra_center, dec_center, 'rx', markersize=10)

    # Handle ticks
    ra_range = true_boundaries['coord1_max'] - true_boundaries['coord1_min']
    dec_range = true_boundaries['coord2_max'] - true_boundaries['coord2_min']
    
    possible_steps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    ra_step = possible_steps[np.abs(ra_range/5 - possible_steps).argmin()]
    dec_step = possible_steps[np.abs(dec_range/5 - possible_steps).argmin()]
    
    x_ticks = np.arange(
        np.ceil(true_boundaries['coord1_min']/ra_step)*ra_step,
        np.floor(true_boundaries['coord1_max']/ra_step)*ra_step + ra_step/2,
        ra_step
    )
    y_ticks = np.arange(
        np.ceil(true_boundaries['coord2_min']/dec_step)*dec_step,
        np.floor(true_boundaries['coord2_max']/dec_step)*dec_step + dec_step/2,
        dec_step
    )
    
    # Convert to scaled coordinates
    scaled_x_ticks = np.interp(
        x_ticks,
        [true_boundaries['coord1_min'], true_boundaries['coord1_max']],
        [scaled_boundaries['coord1_min'], scaled_boundaries['coord1_max']]
    )
    scaled_y_ticks = np.interp(
        y_ticks,
        [true_boundaries['coord2_min'], true_boundaries['coord2_max']],
        [scaled_boundaries['coord2_min'], scaled_boundaries['coord2_max']]
    )
    
    ax.set_xticks(scaled_x_ticks)
    ax.set_yticks(scaled_y_ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
    
    # Set labels
    xlabel = config.get('xlabel')
    ylabel = config.get('ylabel')
    
    if xlabel == 'auto':
        ax.set_xlabel('Right Ascension (degrees)')
    elif xlabel is not None:
        ax.set_xlabel(xlabel)
        
    if ylabel == 'auto':
        ax.set_ylabel('Declination (degrees)')
    elif ylabel is not None:
        ax.set_ylabel(ylabel)
    
    ax.set_title(config['plot_title'])
    
    if config['gridlines']:
        ax.grid(color='black')
    
    # Add colorbar
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)
    
    # Save figure
    fig.tight_layout()
    fig.savefig(output_name)
    map_label = "Convergence" if map_type.lower() == "convergence" else "SNR"
    print(f"{map_label} map saved as PNG file: {output_name}")
    plt.close(fig)