import yaml
from SMPy.KaiserSquires import run_withv2 as ks_run
from SMPy.SNR import run as snr_run

def read_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def run(config_path):
    config = read_config(config_path)
    
    general_config = config['general']
    
    # Create convergence map
    method = general_config.get('method', 'kaiser_squires')
    if method == 'kaiser_squires':
        ks_config = {**general_config, **config['convergence']}
        convergence_map, boundaries = ks_run.create_convergence_map(ks_config)
    else:
        raise ValueError(f"Unknown convergence method: {method}")
    
    # Create SNR map if requested
    if general_config.get('create_snr', False):
        snr_config = {**general_config, **config['snr']}
        snr_run.create_sn_map(snr_config, convergence_map, boundaries)

if __name__ == "__main__":
    import sys
    run(sys.argv[1])