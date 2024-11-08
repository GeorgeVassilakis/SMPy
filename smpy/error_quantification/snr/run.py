import yaml
import numpy as np
import pandas as pd
from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def ks_inversion_list(grid_list):
    """
    Iterate through a list of (g1map, g2map) pairs and return a list of kappa_e values.
    """
    kappa_e_list = []
    kappa_b_list = []
    
    for g1map, g2map in grid_list:
        # Call the ks_inversion function for each pair
        kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, -g2map)
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list

def create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries):
    # Load shear data
    shear_df = utils.load_shear_data(
        config['input_path'],
        config['ra_col'],
        config['dec_col'],
        config['g1_col'],
        config['g2_col'],
        config['weight_col']
    )
    
    # Scale RA and DEC - store scaled coordinates in new columns
    shear_df = utils.scale_ra_dec(shear_df)
    
    # Create shuffled dataframes - this should preserve ra_scaled and dec_scaled columns
    shuffled_dfs = utils.generate_multiple_shear_dfs(shear_df, config['num_shuffles'])
    
    # Create shear grids for shuffled dataframes using scaled coordinates
    g1_g2_map_list = []
    for shuffled_df in shuffled_dfs:
        # Use ra_scaled and dec_scaled for grid creation
        g1map, g2map = utils.create_shear_grid(
            shuffled_df['ra_scaled'],
            shuffled_df['dec_scaled'],
            shuffled_df['g1'],
            shuffled_df['g2'],
            shuffled_df['weight'],
            boundaries=scaled_boundaries,
            resolution=config['resolution']
        )
        g1_g2_map_list.append((g1map, g2map))
    
    # Calculate kappa for shuffled maps
    kappa_e_list, kappa_b_list = ks_inversion_list(g1_g2_map_list)
    
    # Calculate variance maps
    variance_map_e = np.var(np.stack(kappa_e_list, axis=0), axis=0) if kappa_e_list else None
    variance_map_b = np.var(np.stack(kappa_b_list, axis=0), axis=0) if kappa_b_list else None
    
    # Initialize dictionary for signal-to-noise maps
    sn_maps = {}
    
    # Calculate signal-to-noise maps if the respective mode exists
    if 'E' in convergence_maps and variance_map_e is not None:
        sn_maps['E'] = convergence_maps['E'] / np.sqrt(variance_map_e)
    
    if 'B' in convergence_maps and variance_map_b is not None:
        sn_maps['B'] = convergence_maps['B'] / np.sqrt(variance_map_b)
    
    # Plot and save the SNR maps
    modes = config['mode'] if isinstance(config['mode'], list) else [config['mode']]
    
    for mode in modes:
        if mode in sn_maps:
            plot_config = config.copy()
            plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
            output_name = f"{config['output_directory']}{config['output_base_name']}_snr_{mode.lower()}_mode.png"
            plot.plot_convergence(sn_maps[mode], scaled_boundaries, true_boundaries, plot_config, output_name)

def run(config_path, convergence_maps, scaled_boundaries, true_boundaries):
    config = read_config(config_path)
    create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries)