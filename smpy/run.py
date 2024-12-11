import yaml
from smpy.mapping_methods.kaiser_squires import run as ks_run
from smpy.error_quantification.snr import run as snr_run

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

def run(config_path):
    """Run mass mapping workflow.

    Creates convergence maps using specified method and optionally
    generates SNR maps.

    Parameters
    ----------
    config_path : `str`
        Path to configuration file
        
    Raises
    ------
    ValueError
        If unknown mapping method specified
    """
    config = read_config(config_path)
    
    general_config = config['general']
    
    # Create convergence map
    method = general_config.get('method', 'kaiser_squires')
    if method == 'kaiser_squires':
        ks_config = {**general_config, **config['convergence']}
        convergence_maps, scaled_boundaries, true_boundaries = ks_run.create_convergence_map(ks_config)
    else:
        raise ValueError(f"Unknown convergence method: {method}")
    
    # Create SNR map if requested
    if general_config.get('create_snr', False):
        snr_config = {**general_config, **config['snr']}
        snr_run.create_sn_map(snr_config, convergence_maps, scaled_boundaries, true_boundaries)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])