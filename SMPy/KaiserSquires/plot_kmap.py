import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def plot_convergence(convergence, boundaries, config):
    # Apply Gaussian filter 
    filtered_convergence = gaussian_filter(convergence, config['gaussian_kernel'])

    plt.figure(figsize=config['figsize'])
    plt.imshow(filtered_convergence[:, ::-1], 
               cmap=config['cmap'],
               vmax=config['vmax'], 
               vmin=config['vmin'],
               extent=[boundaries['ra_max'], 
                       boundaries['ra_min'], 
                       boundaries['dec_min'], 
                       boundaries['dec_max']],
               origin='lower')  # Sets the origin to bottom left to match the RA/DEC convention
    plt.colorbar()
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    plt.title(config['plot_title'])
    plt.savefig(config['output_path'])
