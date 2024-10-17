import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
import numpy as np

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
