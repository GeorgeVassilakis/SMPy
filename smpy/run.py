"""Main execution module for mass mapping workflows.

This module provides the primary entry point for running mass mapping
analyses with various reconstruction methods. It handles configuration
processing, data loading, coordinate system management, and output
generation for the complete mass mapping pipeline.
"""

import os
import time
from pathlib import Path

import yaml

from smpy import utils
from smpy.coordinates import get_coordinate_system
from smpy.error_quantification.snr.run import create_sn_map as create_snr_map
from smpy.mapping_methods import ApertureMassMapper, KaiserSquiresMapper, KSPlusMapper
def prepare_method_config(config, method):
    """Validate method configuration and return nested config unchanged.
    
    Verify that the specified mass mapping method exists in the
    configuration dictionary. Returns the full nested config structure
    for use with standardized mapper access patterns.
    
    Parameters
    ----------
    config : `dict`
        Full nested configuration dictionary containing method parameters.
    method : `str`
        Method name to validate ('kaiser_squires', 'ks_plus', or
        'aperture_mass').
        
    Returns
    -------
    config : `dict`
        Full nested configuration dictionary.
        
    Raises
    ------
    ValueError
        If method is not found in configuration.
        
    Notes
    -----
    This function preserves the complete nested configuration
    structure required by the mapping
    method implementations.
    """
    # Validate the method exists in config
    if method not in config.get('methods', {}):
        raise ValueError(f"Method '{method}' not found in config")
    return config  # Return full nested config unchanged

def run_mapping(config):
    """Execute mass mapping using the specified reconstruction method.
    
    Perform the complete mass mapping workflow including data loading,
    coordinate system setup, grid creation, and mass reconstruction.
    Handles timing measurement and coordinate-specific sign corrections.
    
    Parameters
    ----------
    config : `dict`
        Full configuration dictionary containing all method parameters,
        coordinate system settings, and file paths.
        
    Returns
    -------
    maps : `dict`
        Dictionary containing reconstructed mass maps with keys
        corresponding to the requested modes ('E', 'B').
    scaled_boundaries : `dict`
        Coordinate boundaries in the scaled coordinate system used for
        plotting and grid creation.
    true_boundaries : `dict`
        Coordinate boundaries in the original coordinate system for
        astronomical positioning and tick labels.
    counts_grid : `numpy.ndarray` or `None`
        Per-pixel counts accumulated during gridding; ``None`` if not
        available.
        
    Raises
    ------
    ValueError
        If unknown mapping method is specified.
        
    Notes
    -----
    The function automatically applies the correct g2 shear component
    sign convention based on the coordinate system: negative for RA/Dec
    (celestial) coordinates, positive for pixel coordinates.
    """
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
    # Capture counts grid (if accumulated) immediately after gridding for optional overlays
    counts_grid = None
    if hasattr(coord_system, '_last_count_grid'):
        counts_grid = getattr(coord_system, '_last_count_grid')
    
    # Get correct g2 sign based on coordinate system
    g2_sign = -1 if coord_system_type == 'radec' else 1
    
    # Create mass mapper instance
    method = config['general']['method']
    if method == 'aperture_mass':
        mapper = ApertureMassMapper(config)
    elif method == 'kaiser_squires':
        mapper = KaiserSquiresMapper(config)
    elif method == 'ks_plus':
        mapper = KSPlusMapper(config)
    else:
        raise ValueError(f"Unknown mapping method: {method}")
    
    # Provide per-pixel weights to mappers that support masking by data presence
    # Coordinate systems expose the accumulated weights as _last_weight_grid
    if method == 'ks_plus' and hasattr(coord_system, '_last_weight_grid'):
        # KS+ expects mask M=1 for data present, 0 for gaps. Weight>0 defines data presence.
        mapper._weight_grid = coord_system._last_weight_grid
    
    # Run mapping with timing
    start_time = time.time()
    maps = mapper.run(
        g1map,
        g2_sign * g2map,
        scaled_boundaries,
        true_boundaries,
        counts_overlay=counts_grid,
    )
    end_time = time.time()
    
    if config['general'].get('print_timing', False):
        elapsed_time = end_time - start_time
        print(f"Time taken to create {method} maps: {elapsed_time:.2f} seconds")
    
    return maps, scaled_boundaries, true_boundaries, counts_grid

def run(config_input):
    """Execute the complete mass mapping analysis workflow.
    
    Main entry point for mass mapping analysis that handles configuration
    processing, data validation, mass reconstruction, and optional SNR
    analysis. Supports multiple input formats and provides comprehensive
    output including maps and coordinate boundaries.
    
    Parameters
    ----------
    config_input : `str`, `pathlib.Path`, `dict`, or `Config`
        Configuration specification. Supported formats:
        - `str` or `pathlib.Path`: Path to YAML configuration file
        - `dict`: Configuration dictionary with required parameters
        - `Config`: SMPy Config object instance
    
    Returns
    -------
    result : `dict`
        Complete analysis results containing:
        - 'maps': Dictionary of reconstructed mass maps by mode
        - 'scaled_boundaries': Coordinate boundaries for plotting
        - 'true_boundaries': Original coordinate boundaries
        - 'snr_maps': SNR maps (if create_snr=True in config)
        - 'counts_map': Per-pixel counts (if create_counts_map=True)
        
    Raises
    ------
    TypeError
        If config_input is not one of the supported types.
    FileNotFoundError
        If specified input data file does not exist.
    ValueError
        If configuration validation fails or method is not supported.
        
    Notes
    -----
    The function automatically handles:
    - File existence validation before processing
    - FITS file output generation (if save_fits=True)
    - SNR map creation and output (if create_snr=True)
    - Method-specific output directory creation
    - Coordinate system conversions for file output
    
    For command-line usage, this function can be called directly with
    a configuration file path.
    """
    from .config import Config
    
    # Handle different input types
    if isinstance(config_input, (str, Path)):
        # Load from file
        with open(config_input, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config_input, Config):
        # Extract dictionary from Config object
        config = config_input.to_dict()
    elif isinstance(config_input, dict):
        # Use dictionary directly
        config = config_input
    else:
        raise TypeError(f"config_input must be str, Path, dict, or Config object, got {type(config_input)}")
    
    # Get method and prepare config
    method = config['general']['method']
    method_config = prepare_method_config(config, method)
    
    # Check file existence right before we need it
    if isinstance(config_input, Config):
        # If we have a Config object, use its method
        config_input.validate_file_existence()
    else:
        # If we have a dict, check manually
        input_path = config['general'].get('input_path')
        
        # Check if we have a non-empty path
        if input_path and input_path != "":
            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"Input file not found: {input_path}\n"
                    f"Please check that the file exists and the path is correct."
                )
    
    # Run mass mapping
    maps, scaled_boundaries, true_boundaries, counts_grid = run_mapping(method_config)
    
    # Save maps as FITS files if requested
    if config['general'].get('save_fits', False):

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
        # Pass full nested config to SNR function
        snr_map = create_snr_map(config, maps, scaled_boundaries, true_boundaries, counts_overlay=counts_grid)
        
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
    
    # Return results
    result = {
        'maps': maps,
        'scaled_boundaries': scaled_boundaries,
        'true_boundaries': true_boundaries
    }
    
    # Add SNR map if created
    if config['general'].get('create_snr', False) and 'snr_map' in locals():
        result['snr_maps'] = snr_map

    # Create and save counts map PNG if requested, using counts from gridding
    if config['general'].get('create_counts_map', False) and counts_grid is not None:
        result['counts_map'] = counts_grid
        if config['general'].get('save_plots', True):
            from smpy.plotting import plot as plot_mod
            plot_cfg = config.get('plotting', {}).copy()
            plot_cfg['coordinate_system'] = config['general'].get('coordinate_system', 'radec')
            if plot_cfg['coordinate_system'] == 'pixel':
                plot_cfg['axis_reference'] = config['general']['pixel'].get('pixel_axis_reference', 'catalog')
            # Set title and ensure linear scaling for counts
            plot_cfg['plot_title'] = f"Counts Map"
            sc = (plot_cfg.get('scaling') or {}).copy()
            sc['type'] = 'linear'
            sc.pop('percentile', None)
            plot_cfg['scaling'] = sc
            method = config['general']['method']
            method_output_dir = f"{config['general']['output_directory']}/{method}"
            os.makedirs(method_output_dir, exist_ok=True)
            output_name = f"{method_output_dir}/{config['general']['output_base_name']}_{method}_counts.png"
            # Use mass_map plotter with counts category to disable peak overlays
            plot_mod.plot_mass_map(counts_grid, scaled_boundaries, true_boundaries, plot_cfg, output_name, map_category="counts")
    
    return result

if __name__ == "__main__":
    import sys
    run(sys.argv[1])
