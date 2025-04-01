import yaml
import numpy as np
import time
from smpy import utils
from smpy.filters import plotting
from smpy.mapping_methods import KaiserSquiresMapper, ApertureMassMapper
from smpy.plotting import plot
from smpy.coordinates import get_coordinate_system

"""Signal-to-noise map generation module.

Creates SNR maps by generating random realizations through spatial or orientation shuffling
to estimate noise properties of the mass maps.
"""

def read_config(file_path):
    """Read configuration from YAML file.

    Parameters
    ----------
    file_path : `str`
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def perform_mapping(grid_list, config, mapping_method='kaiser_squires'):
    """Perform mass mapping on list of shear grids.

    Parameters
    ----------
    grid_list : `list`
        List of (g1_grid, g2_grid) tuples
    config : `dict`
        Configuration dictionary for the mapper
    mapping_method : `str`
        Mapping method to use ('kaiser_squires' or 'aperture_mass')

    Returns
    -------
    kappa_e_list, kappa_b_list : `list`
        Lists of E-mode and B-mode convergence maps
    """
    kappa_e_list = []
    kappa_b_list = []
    
    # Create the appropriate mapper based on the method
    if mapping_method.lower() == 'kaiser_squires':
        mapper = KaiserSquiresMapper(config)
    elif mapping_method.lower() == 'aperture_mass':
        mapper = ApertureMassMapper(config)
    else:
        raise ValueError(f"Unsupported mapping method: {mapping_method}")
    
    # Apply the mapping to each grid
    for g1map, g2map in grid_list:
        kappa_e, kappa_b = mapper.create_maps(g1map, g2map)
        kappa_e_list.append(kappa_e)
        kappa_b_list.append(kappa_b)
    
    return kappa_e_list, kappa_b_list

def create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries):
    """Create signal-to-noise maps from convergence maps.

    Generates noise realizations through spatial/orientation shuffling and creates
    SNR maps for E and B modes.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
        
    Returns
    -------
    dict
        Dictionary containing E/B mode SNR maps
    """
    # Start timing
    start_time = time.time()

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
    mapping_method = config.get('mapping_method', config.get('method', 'kaiser_squires'))
    kappa_e_list, kappa_b_list = perform_mapping(g1_g2_map_list, config, mapping_method)

    # Process maps
    filter_config = config.get('smoothing')
    processed_kappa_e_list = [plotting.apply_filter(k, filter_config) for k in kappa_e_list]
    processed_kappa_b_list = [plotting.apply_filter(k, filter_config) for k in kappa_b_list]
    
    # Calculate variance maps
    variance_map_e = np.var(np.stack(processed_kappa_e_list, axis=0), axis=0)
    variance_map_b = np.var(np.stack(processed_kappa_b_list, axis=0), axis=0)
    
    # Create SNR maps
    sn_maps = {}
    
    if 'E' in convergence_maps:
        convergence_e = convergence_maps['E']
        convergence_e = plotting.apply_filter(convergence_e, filter_config)
        sn_maps['E'] = convergence_e / np.sqrt(variance_map_e)
    
    if 'B' in convergence_maps:
        convergence_b = convergence_maps['B']
        convergence_b = plotting.apply_filter(convergence_b, filter_config)
        sn_maps['B'] = convergence_b / np.sqrt(variance_map_b)
    
    # Plot SNR maps
    for mode in config['mode']:
        if mode in sn_maps:
            plot_config = config.copy()
            plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
            output_name = f"{config['output_directory']}{config['output_base_name']}_{mapping_method}_snr_{mode.lower()}_mode.png"
            plot.plot_convergence(sn_maps[mode], scaled_boundaries, true_boundaries, plot_config, output_name, map_type='snr')
    
    # End timing
    end_time = time.time()
    if config.get('print_timing', False):
        elapsed_time = end_time - start_time
        print(f"Time taken to create {mapping_method} SNR maps: {elapsed_time:.2f} seconds")
    
    return sn_maps

def run(config_path, convergence_maps, scaled_boundaries, true_boundaries):
    """Run SNR map generation.

    Parameters
    ----------
    config_path : `str`
        Path to configuration file
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
        
    Returns
    -------
    dict
        Dictionary containing E/B mode SNR maps
    """
    config = read_config(config_path)
    return create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries)