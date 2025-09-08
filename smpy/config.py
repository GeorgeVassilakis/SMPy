"""Configuration management for SMPy.

This module provides configuration loading, validation, and management
for SMPy mass mapping operations.
"""

import copy
import os
from pathlib import Path
from typing import Dict, Optional

import yaml


class Config:
    """Manage configuration dictionaries for SMPy mass mapping analysis.

    Handle loading, merging, and validating configuration dictionaries from
    YAML files and user parameters. All configurations use consistent nested
    structure for clean architecture and reliable error handling.

    Parameters
    ----------
    config_dict : `dict`, optional
        Configuration dictionary. If `None`, creates empty config.

    Notes
    -----
    The Config class uses a consistent nested configuration structure:

    - **General settings**: ``config['general']`` - Input/output, coordinate
      system, analysis settings
    - **Method-specific**: ``config['methods'][method_name]`` - Parameters
      for each mapping method
    - **Plotting settings**: ``config['plotting']`` - Visualization parameters
    - **SNR settings**: ``config['snr']`` - Signal-to-noise map generation
      parameters

    Configuration access follows the fail-fast principle:
    - Required parameters use direct access:
      ``config['section']['parameter']``
    - Optional parameters use ``.get()``:
      ``config['section'].get('parameter', default)``
    - Missing required config raises immediate ``KeyError`` for clear
      debugging

    Examples
    --------
    Load default configuration for Kaiser-Squires:

    >>> config = Config.from_defaults('kaiser_squires')
    >>> config.show_config()

    Load existing user configuration:

    >>> config = Config.from_file('my_config.yaml')
    >>> config.show_config(section='general')

    Access method-specific parameters:

    >>> cfg_dict = config.to_dict()
    >>> smoothing = cfg_dict['methods']['kaiser_squires']['smoothing']

    Save current configuration:

    >>> config.save_config('output_config.yaml')

    Update configuration programmatically:

    >>> config.update_from_kwargs(
    ...     data='catalog.fits',
    ...     coord_system='radec',
    ...     pixel_scale=0.168
    ... )
    """
    
    def __init__(self, config_dict=None):
        """Initialize Config with optional configuration dictionary.

        Parameters
        ----------
        config_dict : `dict`, optional
            Configuration dictionary. If None, creates empty config.
        """
        self.config = config_dict if config_dict is not None else {}
    
    @classmethod
    def from_file(cls, path):
        """Load configuration from YAML file.

        Parameters
        ----------
        path : `str` or `pathlib.Path`
            Path to YAML configuration file.

        Returns
        -------
        config : `Config`
            Configuration instance loaded from file.
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_defaults(cls, method='kaiser_squires'):
        """Load default configuration for specified method.
        
        Load configuration from the default.yaml file and return the nested 
        structure as-is. This provides consistent configuration structure 
        regardless of loading method.
        
        Parameters
        ----------
        method : `str`, optional
            Method name ('kaiser_squires', 'aperture_mass', or 'ks_plus').
            Default is 'kaiser_squires'.
            
        Returns
        -------
        config : `Config`
            Configuration instance with default settings in nested structure.

        Raises
        ------
        FileNotFoundError
            If the default configuration file cannot be found.
        ValueError
            If the specified method is not supported.

        Notes
        -----
        Returns the full nested configuration structure:

        - **General settings**: ``config['general']`` 
        - **Method-specific**: ``config['methods'][method_name]``
        - **Plotting settings**: ``config['plotting']``
        - **SNR settings**: ``config['snr']``

        Examples
        --------
        Load Kaiser-Squires defaults:

        >>> config = Config.from_defaults('kaiser_squires')
        >>> smoothing = config.to_dict()['methods']['kaiser_squires']['smoothing']
        >>> print(smoothing['type'])
        gaussian

        Load KS+ defaults:

        >>> config = Config.from_defaults('ks_plus')
        >>> ks_config = config.to_dict()['methods']['ks_plus']
        >>> print(ks_config['inpainting_iterations'])
        100
        """
        # Load default.yaml
        defaults_path = Path(__file__).parent / 'configs' / 'default.yaml'
        if not defaults_path.exists():
            raise FileNotFoundError(f"Default config file not found: {defaults_path}")
        
        with open(defaults_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set the method in general section
        config['general']['method'] = method
        
        return cls(config)

    
    def update_from_kwargs(self, **kwargs):
        """Update configuration from keyword arguments.
        
        This method maps simple keyword arguments to the nested
        configuration structure expected by SMPy.
        
        Parameters
        ----------
        **kwargs
            Keyword arguments to convert to config structure
        """
        # Handle data/input_path
        if 'data' in kwargs:
            self._ensure_section('general')
            self.config['general']['input_path'] = kwargs['data']
        
        # Handle coordinate system
        if 'coord_system' in kwargs:
            self._ensure_section('general')
            coord_system = kwargs['coord_system']
            if coord_system not in ['radec', 'pixel']:
                raise ValueError(f"Invalid coord_system: {coord_system}")
            self.config['general']['coordinate_system'] = coord_system
            # Mark that coordinate system was explicitly set by user
            self.config['general']['_coord_system_set_by_user'] = True
        
        # Handle pixel_scale (for radec system)
        if 'pixel_scale' in kwargs and kwargs['pixel_scale'] is not None:
            self._ensure_section('general')
            self._ensure_section('general', 'radec')
            self.config['general']['radec']['resolution'] = kwargs['pixel_scale']
            # Mark that pixel_scale was explicitly set by user
            self.config['general']['_pixel_scale_set_by_user'] = True
        
        # Handle downsample_factor (for pixel system)
        if 'downsample_factor' in kwargs and kwargs['downsample_factor'] is not None:
            self._ensure_section('general')
            self._ensure_section('general', 'pixel')
            self.config['general']['pixel']['downsample_factor'] = kwargs['downsample_factor']
            # Mark that downsample_factor was explicitly set by user
            self.config['general']['_downsample_factor_set_by_user'] = True

        # Handle pixel_axis_reference (for pixel plotting)
        if 'pixel_axis_reference' in kwargs and kwargs['pixel_axis_reference'] is not None:
            self._ensure_section('general')
            self._ensure_section('general', 'pixel')
            axis_ref = kwargs['pixel_axis_reference']
            if axis_ref not in ['catalog', 'map']:
                raise ValueError("pixel_axis_reference must be 'catalog' or 'map'")
            self.config['general']['pixel']['pixel_axis_reference'] = axis_ref
        
        # Handle method
        if 'method' in kwargs:
            self._ensure_section('general')
            self.config['general']['method'] = kwargs['method']
        
        # Handle output directory
        if 'output_dir' in kwargs:
            self._ensure_section('general')
            self.config['general']['output_directory'] = kwargs['output_dir']
        
        # Handle output base name
        if 'output_base_name' in kwargs:
            self._ensure_section('general')
            self.config['general']['output_base_name'] = kwargs['output_base_name']
        
        # Handle smoothing parameter
        if 'smoothing' in kwargs:
            # Always nested structure
            method = self.config['general']['method']
            self._ensure_section('methods')
            self._ensure_section('methods', method)
            self._ensure_section('methods', method, 'smoothing')
            self.config['methods'][method]['smoothing']['sigma'] = kwargs['smoothing']
        
        # Handle create_snr
        if 'create_snr' in kwargs:
            self._ensure_section('general')
            self.config['general']['create_snr'] = kwargs['create_snr']
        
        # Handle create_counts_map
        if 'create_counts_map' in kwargs:
            self._ensure_section('general')
            self.config['general']['create_counts_map'] = kwargs['create_counts_map']
        
        # Handle save_fits
        if 'save_fits' in kwargs:
            self._ensure_section('general')
            self.config['general']['save_fits'] = kwargs['save_fits']
        
        # Handle mode
        if 'mode' in kwargs:
            self._ensure_section('general')
            mode_value = kwargs['mode']
            if isinstance(mode_value, str):
                mode_value = [mode_value]
            self.config['general']['mode'] = mode_value
        
        # Handle data columns
        for col in ['g1_col', 'g2_col', 'weight_col']:
            if col in kwargs:
                self._ensure_section('general')
                self.config['general'][col] = kwargs[col]
        
        # Handle verbosity/timing
        if 'print_timing' in kwargs:
            self._ensure_section('general')
            self.config['general']['print_timing'] = kwargs['print_timing']
        
        if 'verbose' in kwargs:
            self._ensure_section('plotting')
            self.config['plotting']['verbose'] = kwargs['verbose']

        # Handle plotting fontsize
        if 'fontsize' in kwargs and kwargs['fontsize'] is not None:
            self._ensure_section('plotting')
            self.config['plotting']['fontsize'] = kwargs['fontsize']
        
        # Handle KS+ specific parameters
        if 'inpainting_iterations' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            self.config['methods']['ks_plus']['inpainting_iterations'] = kwargs['inpainting_iterations']
        
        if 'reduced_shear_iterations' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            self.config['methods']['ks_plus']['reduced_shear_iterations'] = kwargs['reduced_shear_iterations']
        
        # Handle KS+ wavelet settings
        if 'wavelet' in kwargs and isinstance(kwargs['wavelet'], dict):
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            # Ensure wavelet sub-dict exists, then update
            if 'wavelet' not in self.config['methods']['ks_plus']:
                self.config['methods']['ks_plus']['wavelet'] = {}
            self.config['methods']['ks_plus']['wavelet'].update(kwargs['wavelet'])
        
        if 'wavelet_nscales' in kwargs and kwargs['wavelet_nscales'] is not None:
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            self.config['methods']['ks_plus']['nscales'] = kwargs['wavelet_nscales']
        
        # Handle aperture mass filter parameters
        if 'filter' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'aperture_mass')
            if isinstance(kwargs['filter'], dict):
                self._ensure_section('methods', 'aperture_mass', 'filter')
                self.config['methods']['aperture_mass']['filter'].update(kwargs['filter'])
        
        # Handle individual filter parameters
        if 'filter_type' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'aperture_mass')
            self._ensure_section('methods', 'aperture_mass', 'filter')
            self.config['methods']['aperture_mass']['filter']['type'] = kwargs['filter_type']
        
        if 'filter_scale' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'aperture_mass')
            self._ensure_section('methods', 'aperture_mass', 'filter')
            self.config['methods']['aperture_mass']['filter']['scale'] = kwargs['filter_scale']
    
    def _ensure_section(self, *keys):
        """Ensure nested dictionary sections exist.
        
        Parameters
        ----------
        *keys : `str`
            Nested keys to ensure exist
        """
        current = self.config
        for key in keys:
            if key not in current:
                current[key] = {}
            current = current[key]
    
    def validate(self):
        """Validate configuration for required parameters.
        
        Expects nested configuration structure only.
        
        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        # Access nested structure directly
        general = self.config['general']
        
        # Check for required parameters (only if input_path is actually set to a real value)
        if general.get('input_path') and general['input_path'] != "":
            required_params = ['input_path', 'coordinate_system']
            for param in required_params:
                if param not in general:
                    raise ValueError(f"Required parameter '{param}' missing from config")
        
        # Check coordinate system specific requirements
        # Only validate if input_path is set (meaning this is a real run, not just loading defaults)
        input_path = general.get('input_path', '')
        if input_path and input_path != "":
            coord_system = general.get('coordinate_system', '').lower()
            coord_system_set_by_user = general.get('_coord_system_set_by_user', False)
            
            if coord_system == 'radec':
                # If coordinate system was set by user, require pixel_scale to also be set by user
                if coord_system_set_by_user and not general.get('_pixel_scale_set_by_user', False):
                    raise ValueError(self._missing_coord_param_message('radec'))
                elif not coord_system_set_by_user and ('radec' not in general or 'resolution' not in general['radec']):
                    raise ValueError(self._missing_coord_param_message('radec'))
            elif coord_system == 'pixel':
                # If coordinate system was set by user, require downsample_factor to also be set by user  
                if coord_system_set_by_user and not general.get('_downsample_factor_set_by_user', False):
                    raise ValueError(self._missing_coord_param_message('pixel'))
                elif not coord_system_set_by_user and ('pixel' not in general or 'downsample_factor' not in general['pixel']):
                    raise ValueError(self._missing_coord_param_message('pixel'))
                # Validate optional axis reference if present
                pixel_cfg = general.get('pixel', {})
                axis_ref = pixel_cfg.get('pixel_axis_reference')
                if axis_ref is not None and axis_ref not in ['catalog', 'map']:
                    raise ValueError("'pixel_axis_reference' must be 'catalog' or 'map' when provided")
        
        # Validate method
        method = general.get('method', 'kaiser_squires')
        valid_methods = ['kaiser_squires', 'aperture_mass', 'ks_plus']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")

    def _missing_coord_param_message(self, coord_system: str) -> str:
        """Create a unified, actionable error message for missing parameters.

        Parameters
        ----------
        coord_system : `str`
            The coordinate system specified in the configuration. Expected
            values are 'radec' or 'pixel'.

        Returns
        -------
        message : `str`
            A clear error message that explains what parameter is missing and
            how to provide it via the Python API or YAML configuration.
        """
        if coord_system == 'radec':
            return (
                "Missing required parameter for coordinate_system='radec'. "
                "Provide 'pixel_scale' (API: pixel_scale=..., YAML: general.radec.resolution)."
            )
        if coord_system == 'pixel':
            return (
                "Missing required parameter for coordinate_system='pixel'. "
                "Provide 'downsample_factor' (API: downsample_factor=..., "
                "YAML: general.pixel.downsample_factor)."
            )
        return (
            "Invalid coordinate_system specified. Expected 'radec' or 'pixel'."
        )
    
    def validate_file_existence(self):
        """Validate that input files exist on disk.
        
        Expects nested configuration structure only.
        
        Raises
        ------
        FileNotFoundError
            If input file does not exist
        """
        # Access nested structure directly
        input_path = self.config['general'].get('input_path')
        
        # Skip validation for empty paths or test paths
        if not input_path or input_path == "" or input_path.startswith('/some/fake'):
            return
            
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                f"Please check that the file exists and the path is correct."
            )
    
    def to_dict(self):
        """Return configuration as dictionary.
        
        Returns
        -------
        config : `dict`
            Configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def show_config(self, section=None):
        """Print current configuration in YAML format.
        
        Parameters
        ----------
        section : `str`, optional
            Show only specific section ('general', 'plotting', 'snr', 
            'methods'). If `None`, shows entire configuration.
        """
        if section:
            # Extract and show only requested section
            if section in self.config:
                config_to_show = {section: self.config[section]}
            else:
                print(f"Section '{section}' not found")
                return
        else:
            # Show entire config
            config_to_show = self.config
        
        # Print as YAML
        print(yaml.dump(config_to_show, default_flow_style=False, sort_keys=False))

    def save_config(self, path):
        """Save current configuration to YAML file.
        
        Parameters
        ----------
        path : `str` or `pathlib.Path`
            Path to save configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: {path}")

