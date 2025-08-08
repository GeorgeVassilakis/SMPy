"""SNR map generation through randomized null hypothesis testing.

This module creates signal-to-noise ratio maps by generating random
realizations through spatial or orientation shuffling to estimate noise
properties of the mass maps for statistical significance assessment.
"""

import yaml
import numpy as np
import time
from smpy import utils
from smpy.filters import plotting
from smpy.mapping_methods import KaiserSquiresMapper, ApertureMassMapper, KSPlusMapper
from smpy.plotting import plot
from smpy.coordinates import get_coordinate_system
import os

def read_config(file_path):
    """Read configuration from YAML file.

    Load and parse the YAML configuration file for SNR computation.

    Parameters
    ----------
    file_path : `str`
        Path to configuration file.

    Returns
    -------
    config : `dict`
        Configuration dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def perform_mapping(grid_list, config, mapping_method='kaiser_squires'):
    """Perform mass mapping on list of shear grids.

    Apply the specified mass mapping method to multiple shear grids
    to generate convergence maps for statistical analysis.

    Parameters
    ----------
    grid_list : `list`
        List of (g1_grid, g2_grid) tuples.
    config : `dict`
        Configuration dictionary for the mapper.
    mapping_method : `str`, optional
        Mapping method to use ('kaiser_squires', 'aperture_mass', or
        'ks_plus').

    Returns
    -------
    kappa_e_list : `list`
        List of E-mode convergence maps.
    kappa_b_list : `list`
        List of B-mode convergence maps.

    Raises
    ------
    ValueError
        If unsupported mapping method is specified.
    """
    kappa_e_list = []
    kappa_b_list = []
    
    # Create the appropriate mapper based on the method
    if mapping_method.lower() == 'kaiser_squires':
        mapper = KaiserSquiresMapper(config)
    elif mapping_method.lower() == 'aperture_mass':
        mapper = ApertureMassMapper(config)
    elif mapping_method.lower() == 'ks_plus':
        mapper = KSPlusMapper(config)
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

    Generate noise realizations through spatial or orientation shuffling
    and create SNR maps for E and B modes by comparing signal maps to
    the variance estimated from randomized null realizations.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary.
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries.
    true_boundaries : `dict`
        True coordinate boundaries.

    Returns
    -------
    snr_maps : `dict`
        Dictionary containing E/B mode SNR maps.

    Notes
    -----
    The SNR is computed as: SNR = signal / sqrt(variance_from_shuffles)
    where the variance is estimated from multiple randomized realizations
    of the input data through spatial or orientation shuffling.
    """
    # Start timing
    start_time = time.time()

    # Get coordinate system
    coord_system_type = config['general']['coordinate_system'].lower()
    coord_system = get_coordinate_system(coord_system_type)
    coord_config = config['general'][coord_system_type]
    
    # Load shear data
    shear_df = utils.load_shear_data(
        config['general']['input_path'],
        coord_config['coord1'],
        coord_config['coord2'],
        config['general']['g1_col'],
        config['general']['g2_col'],
        config['general']['weight_col'],
        config['general']['input_hdu']
    )
    
    # Transform coordinates
    shear_df = coord_system.transform_coordinates(shear_df)
    
    # Create shuffled dataframes
    shuffled_dfs = utils.generate_multiple_shear_dfs(
        shear_df,
        config['snr']['num_shuffles'],
        config['snr']['shuffle_type'],
        config['snr'].get('seed', 0)
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
    mapping_method = config['general']['method']
    
    # If the mapping method is KS+, use standard KS for variance calculation
    if mapping_method.lower() == 'ks_plus':
        variance_mapping_method = 'kaiser_squires'
    else:
        variance_mapping_method = mapping_method
    
    kappa_e_list, kappa_b_list = perform_mapping(g1_g2_map_list, config, variance_mapping_method)

    # Process maps
    filter_config = config['snr']['smoothing']
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
    for mode in config['general']['mode']:
        if mode in sn_maps:
            plot_config = config['plotting'].copy()
            # Ensure plotting uses the correct coordinate system
            plot_config['coordinate_system'] = config['general'].get('coordinate_system', 'radec')
            if plot_config['coordinate_system'] == 'pixel':
                plot_config['axis_reference'] = config['general']['pixel'].get('pixel_axis_reference', 'catalog')
            plot_config['plot_title'] = f'{config["snr"]["plot_title"]} ({mode}-mode)'
            
            # Create method-specific output directory
            method_output_dir = f"{config['general']['output_directory']}/{mapping_method}"
            os.makedirs(method_output_dir, exist_ok=True)
            
            output_name = f"{config['general']['output_directory']}/{mapping_method}/{config['general']['output_base_name']}_{mapping_method}_snr_{mode.lower()}_mode.png"
            plot.plot_convergence(sn_maps[mode], scaled_boundaries, true_boundaries, plot_config, output_name, map_type='snr')
    
    # End timing
    end_time = time.time()
    if config['general'].get('print_timing', False):
        elapsed_time = end_time - start_time
        print(f"Time taken to create {mapping_method} SNR maps: {elapsed_time:.2f} seconds")
    
    return sn_maps

def run(config_path, convergence_maps, scaled_boundaries, true_boundaries):
    """Run SNR map generation.

    Main entry point for signal-to-noise ratio map computation from
    convergence maps using randomized null hypothesis testing.

    Parameters
    ----------
    config_path : `str`
        Path to configuration file.
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries.
    true_boundaries : `dict`
        True coordinate boundaries.

    Returns
    -------
    snr_maps : `dict`
        Dictionary containing E/B mode SNR maps.

    Notes
    -----
    This function loads the configuration and delegates to create_sn_map
    for the actual SNR computation and visualization.
    """
    config = read_config(config_path)
    return create_sn_map(config, convergence_maps, scaled_boundaries, true_boundaries)