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

    # Calculate the convergence map
    mode = config['mode']
    if mode == 'E'
        convergence = kaiser_squires.ks_e_mode_inversion(g1map, -g2map)
    elif mode == 'B'
        convergence = kaiser_squires.ks_b_mode_inversion(g1map, -g2map)
    else
        raise ValueError(f"Invalid mode: {mode}. Must be 'E' or 'B'.")

    # Save the convergence map as a FITS file (or not)
    utils.save_convergence_fits(convergence, boundaries, config)

    # Plot the convergence map using the separate plotting function
    plot_kmap.plot_convergence(convergence, boundaries, config)

def run(config_path):
    config = read_config(config_path)
    create_convergence_map(config)
