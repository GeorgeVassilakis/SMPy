import yaml
from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot, filters
from smpy.coordinates import get_coordinate_system

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_convergence_map(config):

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
    
    
    # Calculate true and scaled boundaries
    scaled_boundaries, true_boundaries = coord_system.calculate_boundaries(
        shear_df['coord1'],
        shear_df['coord2']
    )
    
    # Transform coordinates
    shear_df = coord_system.transform_coordinates(shear_df)
    
    # Create shear grid using coordinate system
    g1map, g2map = coord_system.create_grid(
        shear_df,
        scaled_boundaries,
        config
    )
    
    # Determine g2 sign based on coordinate system
    if config['coordinate_system'] == 'radec':
        g2_sign = -1
    else:
        g2_sign = 1
    
    # Calculate the convergence maps
    modes = config['mode']
    kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, g2_sign * g2map)
    
    # Store unfiltered maps
    convergence_maps = {
        'E': kappa_e,
        'B': kappa_b
    }
    
    # Plot and save the convergence maps
    for mode in modes:
        # Get the convergence map for plotting
        plot_map = convergence_maps[mode]
        
        # Apply filtering if configured
        filter_config = config.get('smoothing')
        plot_map = filters.apply_filter(plot_map, filter_config)
        
        plot_config = config.copy()
        plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
        output_name = f"{config['output_directory']}{config['output_base_name']}_kaiser_squires_{mode.lower()}_mode.png"
        plot.plot_convergence(plot_map, scaled_boundaries, true_boundaries, plot_config, output_name)
        
        # Save the convergence map as a FITS file if requested
        if config.get('save_fits', False):
            output_name = f"{config['output_directory']}{config['output_base_name']}_kaiser_squires_{mode.lower()}_mode.fits"
            utils.save_convergence_fits(plot_map, scaled_boundaries, true_boundaries, config, output_name)
    
    return convergence_maps, scaled_boundaries, true_boundaries

def run(config_path):
    config = read_config(config_path)
    return create_convergence_map(config)