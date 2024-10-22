import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np
from matplotlib.animation import FuncAnimation

from lenspack.utils import bin2d
from lenspack.image.inversion import ks93
from lenspack.peaks import find_peaks2d

def plot_convergence(convergence, boundaries, config, output_name):
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
    filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    # Create an aspect ratio that accounts for the curvature of the sky (pixels are not square)
    # TODO: make this more robust. Currently, this takes an average declination (middle of image)
    # and linearly scales the aspect ratio based on that 'middle' declination.
    aspect_ratio = np.cos(np.deg2rad((boundaries['dec_max'] + boundaries['dec_min']) / 2))
    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=config['vmax'], 
        vmin=config['vmin'],
        extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
        origin='lower', # Sets the origin to bottom left to match the RA/DEC convention
        aspect=(1/aspect_ratio)
    )  

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

def plot_convergence_v2(convergence, boundaries, config, output_name, threshold = None):
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
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    
    if config['smoothing'] == 'gaussian_filter':
        filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    elif config['smoothing'] is None:
        filtered_convergence = convergence
        
    if threshold is not None:
        y, x, h = find_peaks2d(filtered_convergence, threshold=threshold, include_border=False)
    else: # empty lists
        x, y, h = [], [], []

    # Make the plot!
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=config['figsize'], tight_layout=True
    )
    
    im = ax.imshow(
        filtered_convergence[:, ::-1], 
        cmap=config['cmap'],
        vmax=config['vmax'], 
        vmin=config['vmin'],
        extent=[boundaries['ra_max'], 
                    boundaries['ra_min'], 
                    boundaries['dec_min'], 
                    boundaries['dec_max']],
        origin='lower' # Sets the origin to bottom left to match the RA/DEC convention
    )
    
    cluster_center = config['cluster_center']
    # Case 1: Do nothing if cluster_center is None
    if cluster_center is None:
        pass
    # Case 2: Calculate center if cluster_center is 'auto'
    elif cluster_center == 'auto':
        ra_0 = (boundaries['ra_max'] + boundaries['ra_min']) / 2
        dec_0 = (boundaries['dec_max'] + boundaries['dec_min']) / 2
        ax.plot(ra_0, dec_0, 'wx', markersize=10)
    # Case 3: Plot marker if cluster_center is a dictionary
    elif isinstance(cluster_center, dict):
        ra_center, dec_center = cluster_center['ra_center'], cluster_center['dec_center']
        ax.plot(ra_center, dec_center, 'wx', markersize=10)
    else:
        print("Unrecognized cluster_center format, skipping marker.")
        
    # convert x,y to ra,dec
    ra_peak, dec_peak = [], []
    for i in range(len(x)):
        ra_peak.append(boundaries['ra_min'] + (x[i]+0.5) * (boundaries['ra_max'] - boundaries['ra_min']) / filtered_convergence.shape[1])
        dec_peak.append(boundaries['dec_min'] + (y[i]+0.5) * (boundaries['dec_max'] - boundaries['dec_min']) / filtered_convergence.shape[0])
    
    ax.scatter(ra_peak, dec_peak, s=100, facecolors='none', edgecolors='g', linewidth=1.5)
      
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

    
    # Apply Gaussian filter -- is this the right place to do it?
    # We are planning on implementing other filters at some point, right?
    #filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])
    # Set the number of frames to 20 or the total length of kappa_e_stack
    #num_frames = min(50, len(convergence))
    
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
        #plt.show()
        #fig.savefig(config['output_path'])
        #plt.close(fig)
        # Create the animation for the first 20 frames
    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    #ani.save('kappa_e_stack_animation_fixed.gif', writer='imagemagick', fps=2)
    #ani.save('kappa_e_stack_animation_smoothed.mp4', writer='ffmpeg', fps=5)
    ani.save(output_name, writer='ffmpeg', fps=fps)
    print(f"Convergence map saved as MP4 file: {output_name}")
    plt.close(fig)