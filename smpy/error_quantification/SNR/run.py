import yaml
import numpy as np
import pandas as pd

from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot

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
        kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, -g2map)  
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list


def create_sn_map(config, convergence_maps, boundaries):
    # Load shear data
    shear_df = utils.load_shear_data(
        config['input_path'],
        config['ra_col'],
        config['dec_col'],
        config['g1_col'],
        config['g2_col'],
        config['weight_col']
    )

    # Create shuffled dataframes
    shuffled_dfs = utils.generate_multiple_shear_dfs(shear_df, config['num_shuffles'])

    # Create shear grids for shuffled dataframes
    g1_g2_map_list = utils.shear_grids_for_shuffled_dfs(shuffled_dfs, boundaries, config)

    # Calculate kappa for shuffled maps
    kappa_e_list, kappa_b_list = ks_inversion_list(g1_g2_map_list)

    # Calculate variance maps
    variance_map_e = np.var(np.stack(kappa_e_list, axis=0), axis=0) if kappa_e_list else None
    variance_map_b = np.var(np.stack(kappa_b_list, axis=0), axis=0) if kappa_b_list else None

    # Initialize an empty dictionary for signal-to-noise maps
    sn_maps = {}

    # Calculate signal-to-noise maps if the respective mode exists
    if 'E' in convergence_maps and variance_map_e is not None:
        sn_maps['E'] = convergence_maps['E'] / np.sqrt(variance_map_e)

    if 'B' in convergence_maps and variance_map_b is not None:
        sn_maps['B'] = convergence_maps['B'] / np.sqrt(variance_map_b)

    # Plot and save the SNR maps
    modes = config['mode'] if isinstance(config['mode'], list) else [config['mode']]  # Ensure modes is a list

    for mode in modes:
        if mode in sn_maps:  # Check if the mode exists in sn_maps
            plot_config = config.copy()
            plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
            output_name = f"{config['output_directory']}{config['output_base_name']}_snr_{mode.lower()}_mode.png"
            plot.plot_convergence(sn_maps[mode], boundaries, plot_config, output_name)

    

def run(config_path, convergence, boundaries):
    config = read_config(config_path)
    create_sn_map(config, convergence, boundaries)












                                        
