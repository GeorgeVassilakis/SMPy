import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib.animation import FuncAnimation
#from lenspack.peaks import find_peaks2d

## By the time plot_convergence is called in the respective run.py files, convergence has already
## been smoothed with a guassian
def plot_convergence(filtered_convergence, scaled_boundaries, true_boundaries, config, output_name):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    # filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    #if config['smoothing'] == 'gaussian_filter':
        #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    #elif config['smoothing'] is None:
        #filtered_convergence = convergence
        
    # Find peaks of convergence
    #peaks = (find_peaks2d(filtered_convergence, threshold=config['threshold'], include_border=False) if config['threshold'] is not None else ([], [], []))

   # ra_peaks = [scaled_boundaries['ra_min'] + (x + 0.5) * (scaled_boundaries['ra_max'] - scaled_boundaries['ra_min']) / filtered_convergence.shape[1] for x in peaks[1]]
   # dec_peaks = [scaled_boundaries['dec_min'] + (y + 0.5) * (scaled_boundaries['dec_max'] - scaled_boundaries['dec_min']) / filtered_convergence.shape[0] for y in peaks[0]]

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=config['vmax'], 
        vmin=config['vmin'],
        extent=[scaled_boundaries['ra_max'], 
                scaled_boundaries['ra_min'], 
                scaled_boundaries['dec_min'], 
                scaled_boundaries['dec_max']],
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    # Mark cluster center if specified
    cluster_center = config['cluster_center']
    ra_center = None
    dec_center = None
    
    if cluster_center == 'auto':
        ra_center = (scaled_boundaries['ra_max'] + scaled_boundaries['ra_min']) / 2
        dec_center = (scaled_boundaries['dec_max'] + scaled_boundaries['dec_min']) / 2
    elif isinstance(cluster_center, dict):
        # Scale the provided coordinates from true to scaled coordinates
        ra_center = np.interp(cluster_center['ra_center'],
                            [true_boundaries['ra_min'], true_boundaries['ra_max']],
                            [scaled_boundaries['ra_min'], scaled_boundaries['ra_max']])
        
        dec_center = np.interp(cluster_center['dec_center'],
                             [true_boundaries['dec_min'], true_boundaries['dec_max']],
                             [scaled_boundaries['dec_min'], scaled_boundaries['dec_max']])
    elif cluster_center is not None:
        print("Unrecognized cluster_center format, skipping marker.")
        ra_center = dec_center = None

    if ra_center is not None:
        ax.plot(ra_center, dec_center, 'wx', markersize=10)

   #ax.scatter(ra_peaks, dec_peaks, s=100, facecolors='none', edgecolors='g', linewidth=1.5)

    # Determine nice step sizes based on the range
    ra_range = true_boundaries['ra_max'] - true_boundaries['ra_min']
    dec_range = true_boundaries['dec_max'] - true_boundaries['dec_min']

    # Choose step size (0.01, 0.05, 0.1, 0.25, 0.5) based on range size
    possible_steps = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    ra_step = possible_steps[np.abs(ra_range/5 - possible_steps).argmin()]
    dec_step = possible_steps[np.abs(dec_range/5 - possible_steps).argmin()]

    # Generate ticks
    x_ticks = np.arange(np.ceil(true_boundaries['ra_min']/ra_step)*ra_step,
                        np.floor(true_boundaries['ra_max']/ra_step)*ra_step + ra_step/2,
                        ra_step)
    y_ticks = np.arange(np.ceil(true_boundaries['dec_min']/dec_step)*dec_step,
                        np.floor(true_boundaries['dec_max']/dec_step)*dec_step + dec_step/2,
                        dec_step)

    # Convert to scaled coordinates
    scaled_x_ticks = np.interp(x_ticks, 
                            [true_boundaries['ra_min'], true_boundaries['ra_max']], 
                            [scaled_boundaries['ra_min'], scaled_boundaries['ra_max']])
    scaled_y_ticks = np.interp(y_ticks, 
                            [true_boundaries['dec_min'], true_boundaries['dec_max']], 
                            [scaled_boundaries['dec_min'], scaled_boundaries['dec_max']])

    # Set the ticks
    ax.set_xticks(scaled_x_ticks)
    ax.set_yticks(scaled_y_ticks)
    ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
      
    ax.set_xlabel(config['xlabel'])
    ax.set_ylabel(config['ylabel'])
    ax.set_title(config['plot_title'])

    # Is there a better way to force something to be a boolean?
    if config['gridlines'] == True:
        ax.grid(color='black')

    # Add colorbar; turn off minor axes first
    plt.rcParams.update({'ytick.minor.visible': False})
    plt.rcParams.update({'xtick.minor.visible': False})

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    fig.colorbar(im, cax=cax)

    # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
    fig.tight_layout() 
    fig.savefig(output_name)
    print(f"Convergence map saved as PNG file: {output_name}")
    plt.close(fig)

def plot_animation(convergence, boundaries, config, output_name='animation.mp4', center_cl=None, smoothing=False, num_frames=50, fps=5):
    """
    Make plot of convergence map and save to file using information passed
    in run configuration file. 

    Arguments
        convergence: XXX raw convergence map XXX
        boundaries: XXX RA/Dec axis limits for plot, set in XXX
        config: overall run configuration file

    """

    # Embiggen font sizes, tick marks, etc.
    fontsize = 15
    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'in'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size':6})
    plt.rcParams.update({'ytick.direction':'in'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
        
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
            extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
            origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
        )  
        if center_cl is not None:
            ra_c, dec_c = center_cl["ra_c"], center_cl["dec_c"]
            ax.plot(ra_c, dec_c, 'wx', markersize=10)

        ax.set_xlabel(config['xlabel'])
        ax.set_ylabel(config['ylabel'])
        #ax.set_title(config['plot_title'])

        # Is there a better way to force something to be a boolean?
        if config['gridlines'] == True:
            ax.grid(color='black')

        # Add colorbar; turn off minor axes first
        plt.rcParams.update({'ytick.minor.visible': False})
        plt.rcParams.update({'xtick.minor.visible': False})

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.07)
        fig.colorbar(im, cax=cax)

        # Save to file and exit, redoing tight_layout b/c sometimes figure gets cut off 
        fig.tight_layout() 

    # Create the animation for the first 20 frames
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(output_name, writer='ffmpeg', fps=fps)
    print(f"Convergence map saved as MP4 file: {output_name}")
    plt.close(fig)