import yaml
import numpy as np
import pandas as pd

from SMPy import utils
from SMPy.KaiserSquires import kaiser_squires
from SMPy.KaiserSquires import plot_kmap

#read in data
def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def ks_inversion_list(grid_list):
    """
    Iterate through a list of (g1map, g2map) pairs and return a list of kappa_e values.
    Parameters:
    grid_list : list of tuples
        A list where each element is a tuple of (g1map, g2map)
    Returns:
    kappa_e_list, kappa_b_list : list
        A list containing the kappa_e_maps for each (g1map, g2map) pair, likewise for kappa_b_maps
    """
    kappa_e_list = []
    kappa_b_list = []
    
    for g1map, g2map in grid_list:
        # Call the ks_inversion function for each pair
        kappa_e, kappa_b = KaiserSquires.ks_inversion(g1map, -g2map)  
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list


def create_sn_map(config)
    # load in orginial data frame 
    shear_df = utils.load_shear_data(config['input_path'], 
                                          config['ra_col'], 
                                          config['dec_col'], 
                                          config['g1_col'], 
                                          config['g2_col'], 
                                          config['weight_col'])

    # Create a list of shuffled data frames given the original data frame and the number of shuffles 
    shuffled_dfs = utils.generate_multiple_shear_dfs(shear_df, 100)

    # Calculate boundaries with a any given dataframe, 
    # since shuffling the data drame does not affect the boundaries
    first_df = shuffled_dfs[0]
    boundaries = utils.calculate_field_boundaries(first_df['ra'], 
                                                    first_df['dec'], 
                                                    config['resolution'])

    #create a list of tuples (g1map, g2map) for all the shuffled data frames
    g1_g2_map_list = shear_grids_for_shuffled_dfs(shuffled_dfs) 

    # create the grid of convergence values for both E and B mode
    shuff_kappa_e_list, shuff_kappa_b_list = ks_inversion_list(g1_g2_map_list)

    #stacks all the maps into a 3D array (axis = 0 is the depth across all the maps)
    kappa_e_stack = np.stack(shuff_kappa_e_list, axis = 0)
    kappa_b_stack = np.stack(shuff_kappa_b_list, axis = 0)

    #takes the variance across each map for each pixel 
    variance_map_e = np.var(kappa_e_stack, axis = 0)
    variance_map_b = np.var(kappa_b_stack, axis = 0)

    #calculate signal to noise
    signal_to_noise = (og_kappa_e / variance_map_e)

    #plot ***important, need to adjust plot_convergence to include S/N***
    config_copy = config.copy()
    plot_kmap.plot_convergence(signal_to_noise, boundaries, config_copy)

    # Save the convergence map as a FITS file (**not sure if I did this right**)
    if config.get('save_fits', False):
        fits_output_path = config.get('fits_output_path', config['output_path'].replace('.png', '.fits'))
        config_copy['fits_output_path'] = fits_output_path
        utils.save_convergence_fits(convergence, boundaries, config_copy)
    

def run(config_path):
    config = read_config(config_path)
    create_sn_map(config)












                                        
