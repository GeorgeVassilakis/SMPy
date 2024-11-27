import yaml
import numpy as np
from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot, filters
from smpy.coordinates import get_coordinate_system

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def ks_inversion_list(grid_list, coord_system_type='radec'):

    kappa_e_list = []
    kappa_b_list = []
    
    # Set g2 sign based on coordinate system
    g2_sign = -1 if coord_system_type == 'radec' else 1
    
    for g1map, g2map in grid_list:
        kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, g2_sign * g2map)
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list

def create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries):

    # Get coordinate system
    coord_system_type = config.get('coordinate_system', 'radec').lower()
    coord_system = get_coordinate_system(coord_system_type)
    coord_config = config.get(coord_system_type, {})
    
    # Load shear data
    shear_df = utils.load_shear_data(
        config['input_path'],
        coord_config['coord1'],
        coord_config['coord2'],
        config['g1_col'],
        config['g2_col'],
        config['weight_col'],
        config['input_hdu']
    )
    
    # Transform coordinates
    shear_df = coord_system.transform_coordinates(shear_df)
    
    # Create shuffled dataframes
    shuffled_dfs = utils.generate_multiple_shear_dfs(
        shear_df,
        config['num_shuffles'],
        config['shuffle_type']
    )
    
    # Create shear grids for shuffled dataframes
    g1_g2_map_list = []
    for shuffled_df in shuffled_dfs:
        g1map, g2map = coord_system.create_grid(
            shuffled_df,
            scaled_boundaries,
            config
        )
        g1_g2_map_list.append((g1map, g2map))
    
    # Calculate kappa for shuffled maps
    kappa_e_list, kappa_b_list = ks_inversion_list(g1_g2_map_list, coord_system_type)

    # Process maps
    filter_config = config.get('smoothing')
    processed_kappa_e_list = [filters.apply_filter(k, filter_config) for k in kappa_e_list]
    processed_kappa_b_list = [filters.apply_filter(k, filter_config) for k in kappa_b_list]
    
    # Calculate variance maps
    variance_map_e = np.var(np.stack(processed_kappa_e_list, axis=0), axis=0)
    variance_map_b = np.var(np.stack(processed_kappa_b_list, axis=0), axis=0)
    
    # Create SNR maps
    sn_maps = {}
    
    if 'E' in convergence_maps:
        convergence_e = convergence_maps['E']
        convergence_e = filters.apply_filter(convergence_e, filter_config)
        sn_maps['E'] = convergence_e / np.sqrt(variance_map_e)
    
    if 'B' in convergence_maps:
        convergence_b = convergence_maps['B']
        convergence_b = filters.apply_filter(convergence_b, filter_config)
        sn_maps['B'] = convergence_b / np.sqrt(variance_map_b)
    
    # Plot SNR maps
    for mode in config['mode']:
        if mode in sn_maps:
            plot_config = config.copy()
            plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
            output_name = f"{config['output_directory']}{config['output_base_name']}_snr_{mode.lower()}_mode.png"
            plot.plot_convergence(sn_maps[mode], scaled_boundaries, true_boundaries, plot_config, output_name)

def run(config_path, convergence_maps, scaled_boundaries, true_boundaries):
    config = read_config(config_path)
    create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries)