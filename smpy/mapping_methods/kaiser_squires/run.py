import yaml
from smpy import utils
from smpy.mapping_methods.kaiser_squires import kaiser_squires
from smpy.plotting import plot

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_convergence_map(config):
    # Load shear data 
    shear_df = utils.load_shear_data(config['input_path'], 
                                          config['ra_col'], 
                                          config['dec_col'], 
                                          config['g1_col'], 
                                          config['g2_col'], 
                                          config['weight_col'])

    # Calculate true field boundaries before scaling
    true_boundaries = utils.calculate_field_boundaries(shear_df['ra'], 
                                                     shear_df['dec'])
    # Transform RA/Dec (center and scale)
    shear_df = utils.scale_ra_dec(shear_df)

    # Calculate field boundaries
    scaled_boundaries = utils.calculate_field_boundaries(shear_df['ra_scaled'], 
                                                  shear_df['dec_scaled'])
    
    # Create shear grid
    g1map, g2map = utils.create_shear_grid(shear_df['ra_scaled'], 
                                           shear_df['dec_scaled'], 
                                           shear_df['g1'],
                                           shear_df['g2'], 
                                           shear_df['weight'], 
                                           boundaries=scaled_boundaries,
                                           resolution=config['resolution'])

    # Calculate the convergence maps
    modes = config['mode']
    kappa_e, kappa_b = kaiser_squires.ks_inversion(g1map, -g2map)

    convergence_maps = {}
    if 'E' in modes:
        convergence_maps['E'] = kappa_e
    if 'B' in modes:
        convergence_maps['B'] = kappa_b

    # Plot and save the convergence maps
    for mode, convergence in convergence_maps.items():
        plot_config = config.copy()
        plot_config['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
        output_name = f"{config['output_directory']}{config['output_base_name']}_kaiser_squires_{mode.lower()}_mode.png"
        plot.plot_convergence(convergence, scaled_boundaries, true_boundaries, plot_config, output_name)

        # Save the convergence map as a FITS file
        if config.get('save_fits', False):
            output_name = f"{config['output_directory']}{config['output_base_name']}_kaiser_squires_{mode.lower()}_mode.fits"
            utils.save_convergence_fits(convergence, scaled_boundaries, true_boundaries, config, output_name)

    return convergence_maps, scaled_boundaries, true_boundaries

def run(config_path):
    config = read_config(config_path)
    create_convergence_map(config)