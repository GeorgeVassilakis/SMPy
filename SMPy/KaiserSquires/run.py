import yaml
from SMPy import utils
from SMPy.KaiserSquires import kaiser_squires
from SMPy.KaiserSquires import plot_kmap

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

    # Calculate field boundaries
    boundaries = utils.calculate_field_boundaries(shear_df['ra'], 
                                                  shear_df['dec'], 
                                                  config['resolution'], 
                                                  )

    # Create shear grid
    g1map, g2map = utils.create_shear_grid(shear_df['ra'], 
                                           shear_df['dec'], 
                                           shear_df['g1'],
                                           shear_df['g2'], 
                                           shear_df['weight'], 
                                           boundaries=boundaries,
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
        config_copy = config.copy()
        config_copy['plot_title'] = f'{config["plot_title"]} ({mode}-mode)'
        config_copy['output_path'] = config['output_path'].replace('.png', f'_{mode.lower()}_mode.png')
        plot_kmap.plot_convergence(convergence, boundaries, config_copy)

        # Save the convergence map as a FITS file
        if config.get('save_fits', False):
            fits_output_path = config.get('fits_output_path', config['output_path'].replace('.png', '.fits'))
            fits_output_path = fits_output_path.replace('.fits', f'_{mode.lower()}_mode.fits')
            config_copy['fits_output_path'] = fits_output_path
            utils.save_convergence_fits(convergence, boundaries, config_copy)

def run(config_path):
    config = read_config(config_path)
    create_convergence_map(config)
