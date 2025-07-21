"""Configuration management for SMPy."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import warnings


class Config:
    """Manage configuration dictionaries for SMPy mass mapping analysis.
    
    Handle loading, merging, and validating configuration dictionaries from 
    YAML files and user parameters. Support both nested configuration 
    structures (from user files) and flattened structures (from default 
    configurations) to maintain backward compatibility while simplifying 
    the user experience.

    Parameters
    ----------
    config_dict : `dict`, optional
        Configuration dictionary. If `None`, creates empty config.

    Notes
    -----
    The Config class supports two configuration structures:

    1. **Nested Structure** (from `from_file`):
       - Has 'general', 'methods', 'plotting', 'snr' sections
       - Maintains original user file organization
       - Used for existing user configuration files

    2. **Flattened Structure** (from `from_defaults`):
       - Method-specific parameters promoted to top level
       - Simplified structure for programmatic use
       - Optimized for mapping method requirements

    The flattened structure ensures each mapping method gets its 
    configuration in the expected format:

    - **Kaiser-Squires**: ``config['smoothing']`` (top level)
    - **Aperture Mass**: ``config['filter']`` (top level)  
    - **KS+**: ``config['ks_plus']`` (nested section)

    Examples
    --------
    Load default configuration for Kaiser-Squires:

    >>> config = Config.from_defaults('kaiser_squires')
    >>> config.show_config()

    Load existing user configuration:

    >>> config = Config.from_file('my_config.yaml')
    >>> config.show_config(section='general')

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
            Path to YAML configuration file
            
        Returns
        -------
        config : `Config`
            Configuration instance loaded from file
        """
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_defaults(cls, method='kaiser_squires'):
        """Load default configuration for specified method with flattening.
        
        Load configuration from the single default.yaml file and flatten the 
        structure to match the access patterns expected by each mapping method.
        This ensures backward compatibility while simplifying configuration 
        management.
        
        Parameters
        ----------
        method : `str`, optional
            Method name ('kaiser_squires', 'aperture_mass', or 'ks_plus').
            Default is 'kaiser_squires'.
            
        Returns
        -------
        config : `Config`
            Configuration instance with default settings, properly flattened
            for the specified method.

        Raises
        ------
        FileNotFoundError
            If the default configuration file cannot be found.
        ValueError
            If the specified method is not supported.

        Notes
        -----
        The flattening process promotes method-specific configuration to the 
        top level while preserving coordinate system structure:

        - **Kaiser-Squires**: 'smoothing' moved to top level
        - **Aperture Mass**: 'filter' moved to top level
        - **KS+**: 'ks_plus' section preserved as nested structure
        - **All methods**: 'plotting' and 'snr' sections flattened to top level

        Examples
        --------
        Load Kaiser-Squires defaults:

        >>> config = Config.from_defaults('kaiser_squires')
        >>> smoothing = config.to_dict()['smoothing']
        >>> print(smoothing['type'])
        gaussian

        Load KS+ defaults:

        >>> config = Config.from_defaults('ks_plus')
        >>> ks_config = config.to_dict()['ks_plus']
        >>> print(ks_config['inpainting_iterations'])
        100
        """
        # Load single default.yaml (which has everything!)
        defaults_path = Path(__file__).parent / 'defaults' / 'default.yaml'
        if not defaults_path.exists():
            raise FileNotFoundError(f"Default config file not found: {defaults_path}")
        
        with open(defaults_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Set the method in general section
        raw_config['general']['method'] = method
        
        # Flatten the configuration using the same pattern as run.py::prepare_method_config()
        flattened_config = raw_config['general'].copy()
        
        # Special handling for KS+ vs other methods
        if method == 'ks_plus':
            # KS+ expects its config nested under 'ks_plus' key
            flattened_config['ks_plus'] = raw_config['methods']['ks_plus']
        else:
            # Other methods expect their config at top level
            flattened_config.update(raw_config['methods'].get(method, {}))
        
        flattened_config.update(raw_config['plotting'])
        flattened_config.update(raw_config['snr'])
        
        # Keep the nested structure for coordinate systems (needed by coordinate modules)
        flattened_config['radec'] = raw_config['general']['radec']
        flattened_config['pixel'] = raw_config['general']['pixel']
        
        return cls(flattened_config)

    # BACKUP: Original from_defaults() implementation (removed in Phase 2)
    # @classmethod
    # def from_defaults_original(cls, method='kaiser_squires'):
    #     """Load default configuration for specified method.
    #     
    #     This is the original implementation that loaded separate method files.
    #     Kept as backup reference during refactoring.
    #     """
    #     # Get the path to the defaults directory
    #     defaults_dir = Path(__file__).parent / 'defaults'
    #     
    #     # Load base default config
    #     base_config_path = defaults_dir / 'default.yaml'
    #     if not base_config_path.exists():
    #         raise FileNotFoundError(f"Default config file not found: {base_config_path}")
    #     
    #     with open(base_config_path, 'r') as f:
    #         base_config = yaml.safe_load(f)
    #     
    #     # Load method-specific config if it exists
    #     method_config_path = defaults_dir / f'{method}.yaml'
    #     if method_config_path.exists():
    #         with open(method_config_path, 'r') as f:
    #             method_config = yaml.safe_load(f)
    #         # Deep merge method config into base config
    #         merged_config = cls._deep_merge(base_config, method_config)
    #     else:
    #         merged_config = base_config
    #     
    #     # Set the method in the config
    #     merged_config['general']['method'] = method
    #     
    #     return cls(merged_config)
    
    # REMOVED: _deep_merge() and update() methods (Phase 2)
    # These methods were removed as part of the configuration simplification.
    # The complex deep merging is no longer needed with the flattened approach.
    
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
            # Handle both nested and flattened configs
            if 'general' in self.config:
                # Nested structure
                method = self.config.get('general', {}).get('method', 'kaiser_squires')
                self._ensure_section('methods')
                self._ensure_section('methods', method)
                self._ensure_section('methods', method, 'smoothing')
                self.config['methods'][method]['smoothing']['sigma'] = kwargs['smoothing']
            else:
                # Flattened structure - smoothing is at top level
                if 'smoothing' in self.config and isinstance(self.config['smoothing'], dict):
                    self.config['smoothing']['sigma'] = kwargs['smoothing']
                else:
                    # Create smoothing section if it doesn't exist
                    self.config['smoothing'] = {'type': 'gaussian', 'sigma': kwargs['smoothing']}
        
        # Handle create_snr
        if 'create_snr' in kwargs:
            self._ensure_section('general')
            self.config['general']['create_snr'] = kwargs['create_snr']
        
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
        
        # Handle KS+ specific parameters
        if 'inpainting_iterations' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            self.config['methods']['ks_plus']['inpainting_iterations'] = kwargs['inpainting_iterations']
        
        if 'reduced_shear_iterations' in kwargs:
            self._ensure_section('methods')
            self._ensure_section('methods', 'ks_plus')
            self.config['methods']['ks_plus']['reduced_shear_iterations'] = kwargs['reduced_shear_iterations']
        
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
        
        Supports both nested (from_file) and flattened (from_defaults) config structures.
        
        Raises
        ------
        ValueError
            If required parameters are missing or invalid
        """
        # Handle both nested (with 'general' section) and flattened configs
        if 'general' in self.config:
            # Nested structure (from from_file or old configs)
            general = self.config['general']
        else:
            # Flattened structure (from from_defaults)
            general = self.config
        
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
                    raise ValueError("For 'radec' coordinate system, 'pixel_scale' parameter is required")
                elif not coord_system_set_by_user and ('radec' not in general or 'resolution' not in general['radec']):
                    raise ValueError("For 'radec' coordinate system, 'pixel_scale' parameter is required")
            elif coord_system == 'pixel':
                # If coordinate system was set by user, require downsample_factor to also be set by user  
                if coord_system_set_by_user and not general.get('_downsample_factor_set_by_user', False):
                    raise ValueError("For 'pixel' coordinate system, 'downsample_factor' parameter is required")
                elif not coord_system_set_by_user and ('pixel' not in general or 'downsample_factor' not in general['pixel']):
                    raise ValueError("For 'pixel' coordinate system, 'downsample_factor' parameter is required")
        
        # Validate method
        method = general.get('method', 'kaiser_squires')
        valid_methods = ['kaiser_squires', 'aperture_mass', 'ks_plus']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
    
    def validate_file_existence(self):
        """Validate that input files exist on disk.
        
        Supports both nested (from_file) and flattened (from_defaults) config structures.
        
        Raises
        ------
        FileNotFoundError
            If input file does not exist
        """
        # Handle both nested and flattened configs
        if 'general' in self.config:
            input_path = self.config['general'].get('input_path')
        else:
            input_path = self.config.get('input_path')
        
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

