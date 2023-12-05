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
                                                  config['width'])

    # Create shear grid
    g1map, g2map = utils.create_shear_grid(shear_df['ra'], 
                                           shear_df['dec'], 
                                           shear_df['g1'],
                                           shear_df['g2'], 
                                           shear_df['weight'], 
                                           boundaries=boundaries,
                                           npix=config['width'])

    # Calculate the convergence map
    convergence = kaiser_squires.ks_inversion(g1map, -g2map, config['width'])

    # Plot the convergence map using the separate plotting function
    plot_kmap.plot_convergence(convergence, boundaries, config)

def run(config_path):
    config = read_config(config_path)
    create_convergence_map(config)
