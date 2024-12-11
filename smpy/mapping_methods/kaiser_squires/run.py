import yaml
from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot, filters
from smpy.coordinates import get_coordinate_system

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

def create_convergence_map(config):
    """Create convergence maps using Kaiser-Squires method.

    Creates both E-mode and B-mode convergence maps from shear data,
    handles coordinate transformations, and generates visualizations.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary containing all mapping parameters

    Returns
    -------
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
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
    
    return convergence_maps, scaled_boundaries, true_boundaries

def run(config_path):
    """Run Kaiser-Squires mass mapping.

    Parameters
    ----------
    config_path : `str`
        Path to configuration file

    Returns
    -------
    convergence_maps : `dict`
        Dictionary containing E/B mode convergence maps
    scaled_boundaries : `dict`
        Scaled coordinate boundaries
    true_boundaries : `dict`
        True coordinate boundaries
    """
    config = read_config(config_path)
    return create_convergence_map(config)