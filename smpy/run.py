"""Main run module for mass mapping."""

import yaml
import time
from smpy import utils
from smpy.coordinates import get_coordinate_system
from smpy.mapping_methods import KaiserSquiresMapper, ApertureMassMapper, KSPlusMapper
from smpy.error_quantification.snr import run as snr_run
import os
def prepare_method_config(config, method):
    """Prepare method-specific configuration with plotting settings.
    
    Parameters
    ----------
    config : dict
        Full configuration dictionary
    method : str
        Method name
        
    Returns
    -------
    dict
        Combined configuration for specified method
    """
    method_config = config['general'].copy()
    method_config.update(config['methods'].get(method, {}))
    method_config.update(config['plotting'])
    return method_config

def run_mapping(config):
    """Run mass mapping with specified method.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
        
    Returns
    -------
    maps : dict
        Dictionary containing mass maps
    scaled_boundaries : dict
        Scaled coordinate boundaries
    true_boundaries : dict
        True coordinate boundaries
    """
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
    
    # Calculate boundaries
    scaled_boundaries, true_boundaries = coord_system.calculate_boundaries(
        shear_df['coord1'],
        shear_df['coord2']
    )
    
    # Transform coordinates
    shear_df = coord_system.transform_coordinates(shear_df)
    
    # Create shear grid
    g1map, g2map = coord_system.create_grid(
        shear_df,
        scaled_boundaries,
        config
    )
    
    # Get correct g2 sign based on coordinate system
    g2_sign = -1 if coord_system_type == 'radec' else 1
    
    # Create mass mapper instance
    method = config['method']
    if method == 'aperture_mass':
        mapper = ApertureMassMapper(config)
    elif method == 'kaiser_squires':
        mapper = KaiserSquiresMapper(config)
    elif method == 'ks_plus':
        mapper = KSPlusMapper(config)
    else:
        raise ValueError(f"Unknown mapping method: {method}")
    
    # Run mapping with timing
    start_time = time.time()
    maps = mapper.run(g1map, g2_sign * g2map, scaled_boundaries, true_boundaries)
    end_time = time.time()
    
    if config.get('print_timing', False):
        elapsed_time = end_time - start_time
        print(f"Time taken to create {method} maps: {elapsed_time:.2f} seconds")
    
    return maps, scaled_boundaries, true_boundaries

def run(config_path):
    """Run mass mapping workflow.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    """
    # Read configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get method and prepare config
    method = config['general']['method']
    method_config = prepare_method_config(config, method)
    
    # Run mass mapping
    maps, scaled_boundaries, true_boundaries = run_mapping(method_config)
    
    # Save maps as FITS files if requested
    if config['general'].get('save_fits', False):
        import os
        method_output_dir = f"{config['general']['output_directory']}/{method}"
        os.makedirs(method_output_dir, exist_ok=True)
        
        for mode in config['general']['mode']:
            if mode in maps:
                output_path = f"{method_output_dir}/{config['general']['output_base_name']}_{method}_{mode.lower()}_mode.fits"
                # Convert coordinate system to format expected by save_fits
                if config['general']['coordinate_system'].lower() == 'radec':
                    fits_boundaries = {
                        'ra_min': true_boundaries['coord1_min'],
                        'ra_max': true_boundaries['coord1_max'],
                        'dec_min': true_boundaries['coord2_min'],
                        'dec_max': true_boundaries['coord2_max']
                    }
                    utils.save_fits(maps[mode], fits_boundaries, output_path)
    
    # Create SNR map if requested
    if config['general'].get('create_snr', False):
        snr_config = config['general'].copy()
        snr_config.update(config['snr'])
        snr_config.update(config['plotting'])
        if 'print_timing' in config['general']:
            snr_config['print_timing'] = config['general']['print_timing']
        snr_map = snr_run.create_sn_map(snr_config, maps, scaled_boundaries, true_boundaries)
        
        # Save SNR maps as FITS files if requested
        if config['general'].get('save_fits', False) and snr_map:
            method_output_dir = f"{config['general']['output_directory']}/{method}"
            os.makedirs(method_output_dir, exist_ok=True)
            
            for mode in config['general']['mode']:
                if mode in snr_map:
                    output_path = f"{method_output_dir}/{config['general']['output_base_name']}_{method}_snr_{mode.lower()}_mode.fits"
                    # Convert coordinate system to format expected by save_fits
                    if config['general']['coordinate_system'].lower() == 'radec':
                        fits_boundaries = {
                            'ra_min': true_boundaries['coord1_min'],
                            'ra_max': true_boundaries['coord1_max'],
                            'dec_min': true_boundaries['coord2_min'],
                            'dec_max': true_boundaries['coord2_max']
                        }
                        utils.save_fits(snr_map[mode], fits_boundaries, output_path)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])