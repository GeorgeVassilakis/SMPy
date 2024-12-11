import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib.animation import FuncAnimation
from smpy.utils import find_peaks2d

def _get_center_coordinates(cluster_center, scaled_boundaries, true_boundaries, coord_system_type):
    """Get center coordinates for plotting.
    
    Parameters
    ----------
    cluster_center : `str` or `dict` or None
        Center specification:
        - 'auto': use field center
        - dict: specific coordinates
        - None: no center
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
    coord_system_type : `str`
        'radec' or 'pixel'
        
    Returns
    -------
    center_coord1, center_coord2 : `float`
        Center coordinates in scaled system
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

def plot_convergence(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name):
    """Create plot of convergence map.
    
    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence map data
    scaled_boundaries : `dict`
        Scaled coordinate boundaries 
    true_boundaries : `dict`
        True coordinate boundaries
    config : `dict`
        Plot configuration settings
    output_name : `str`
        Path for saving plot
    """

    # Check coordinate system
    coord_system = config.get('coordinate_system', 'radec').lower()
    
    if coord_system == 'radec':
        _plot_convergence_radec(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name)
    else:
        _plot_convergence_pixel(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name)

def _set_plot_params(fontsize=15):
    """Set standard matplotlib parameters.
    
    Parameters
    ----------
    fontsize : `int`
        Base font size for plot elements
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

def _plot_convergence_pixel(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name):
    """Create convergence plot for pixel coordinates.
    
    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence map data
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
    config : `dict`
        Plot configuration settings
    output_name : `str`
        Path for saving plot
    """

    _set_plot_params()
    
    # Create figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, 
        figsize=config['figsize']
    )
    
    # Plot convergence map
    im = ax.imshow(
        filtered_convergence,
        cmap=config['cmap'],
        vmax=config.get('vmax'),
        vmin=config.get('vmin'),
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
        X, Y, heights, coords = find_peaks2d(filtered_convergence, 
                                           threshold=threshold,
                                           verbose=config.get('verbose', False),
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
        print(f"Convergence map saved as PNG file: {output_name}")
    
    plt.close(fig)

def _plot_convergence_radec(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name):
    """Create convergence plot for RA/Dec coordinates.
    
    Parameters
    ----------
    filtered_convergence : `numpy.ndarray`
        2D convergence map data
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
    config : `dict`
        Plot configuration settings
    output_name : `str`
        Path for saving plot
    """

    # Set plotting parameters
    _set_plot_params()

    # Create figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    # Plot convergence map
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=config['vmax'], 
        vmin=config['vmin'],
        extent=[scaled_boundaries['coord1_max'], 
                scaled_boundaries['coord1_min'], 
                scaled_boundaries['coord2_min'], 
                scaled_boundaries['coord2_max']],
        origin='lower'
    )
    
    # Plot peaks if threshold specified
    threshold = config.get('threshold')
    if threshold is not None:
        X, Y, heights, coords = find_peaks2d(filtered_convergence, 
                                           threshold=threshold,
                                           verbose=config.get('verbose', False),
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
    print(f"Convergence map saved as PNG file: {output_name}")
    plt.close(fig)

def plot_animation(convergence, boundaries, config, output_name='animation.mp4', 
                  center_cl=None, smoothing=False, num_frames=50, fps=5):
    """Create animation of convergence maps.
    
    Parameters
    ----------
    convergence : `list`
        List of 2D convergence maps
    boundaries : `dict`
        Coordinate boundaries
    config : `dict`
        Plot configuration
    output_name : `str`
        Output file path
    center_cl : `dict` or None
        Center coordinates
    smoothing : `bool`
        Whether to apply smoothing
    num_frames : `int`
        Number of animation frames
    fps : `int`
        Frames per second
    """
    _set_plot_params()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True)
        
    def update(frame):
        if smoothing:
            filtered_convergence = gaussian_filter(convergence[frame], config['gaussian_kernel'])
        else:
            filtered_convergence = convergence[frame]
            
        im = ax.imshow(
            filtered_convergence[:, ::-1], 
            cmap=config['cmap'],
            vmax=1.5, 
            vmin=-1.5,
            extent=[boundaries['coord1_max'], 
                    boundaries['coord1_min'], 
                    boundaries['coord2_min'], 
                    boundaries['coord2_max']],
            origin='lower'
        )  
        if center_cl is not None:
            ra_c, dec_c = center_cl["ra_c"], center_cl["dec_c"]
            ax.plot(ra_c, dec_c, 'rx', markersize=10)

        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])

        if config['gridlines']:
            ax.grid(color='black')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(im, cax=cax)

        fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_name, writer='ffmpeg', fps=fps)
    print(f"Convergence map saved as MP4 file: {output_name}")
    plt.close(fig)