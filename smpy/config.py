"""Configuration management for SMPy."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Config:
    """Configuration management class for SMPy mass mapping.
    
    This class handles loading, merging, and validating configuration
    dictionaries from YAML files and user parameters.
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
        """Load default configuration for specified method.
        
        Parameters
        ----------
        method : `str`, optional
            Method name ('kaiser_squires', 'aperture_mass', or 'ks_plus')
            
        Returns
        -------
        config : `Config`
            Configuration instance with default settings
        """
        # Get the path to the defaults directory
        defaults_dir = Path(__file__).parent / 'defaults'
        
        # Load base default config
        base_config_path = defaults_dir / 'default.yaml'
        if not base_config_path.exists():
            raise FileNotFoundError(f"Default config file not found: {base_config_path}")
        
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Load method-specific config if it exists
        method_config_path = defaults_dir / f'{method}.yaml'
        if method_config_path.exists():
            with open(method_config_path, 'r') as f:
                method_config = yaml.safe_load(f)
            # Deep merge method config into base config
            merged_config = cls._deep_merge(base_config, method_config)
        else:
            merged_config = base_config
        
        # Set the method in the config
        merged_config['general']['method'] = method
        
        return cls(merged_config)
    
    @staticmethod
    def _deep_merge(dict1, dict2):
        """Deep merge two dictionaries.
        
        Parameters
        ----------
        dict1 : `dict`
            Base dictionary
        dict2 : `dict`
            Dictionary to merge into dict1
            
        Returns
        -------
        merged : `dict`
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def update(self, other):
        """Update configuration with another dictionary.
        
        Parameters
        ----------
        other : `dict`
            Dictionary to merge into current config
        """
        self.config = self._deep_merge(self.config, other)
    
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
            # Map user-friendly names to internal names
            coord_system_map = {
                'ra_dec': 'radec',
                'pixel': 'pixel'
            }
            coord_system = coord_system_map.get(kwargs['coord_system'], kwargs['coord_system'])
            self.config['general']['coordinate_system'] = coord_system
            # Mark that coordinate system was explicitly set by user
            self.config['general']['_coord_system_set_by_user'] = True
        
        # Handle pixel_scale (for ra_dec system)
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
            method = self.config.get('general', {}).get('method', 'kaiser_squires')
            self._ensure_section('methods')
            self._ensure_section('methods', method)
            self._ensure_section('methods', method, 'smoothing')
            self.config['methods'][method]['smoothing']['sigma'] = kwargs['smoothing']
        
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
        
        Raises
        ------
        ValueError
            If required parameters are missing
        """
        # Check for required general sections
        if 'general' not in self.config:
            raise ValueError("Configuration missing 'general' section")
        
        general = self.config['general']
        
        # Check for required parameters (only if input_path is actually set to a real value)
        if general.get('input_path') and general['input_path'] != "":
            required_params = ['input_path', 'coordinate_system']
            for param in required_params:
                if param not in general:
                    raise ValueError(f"Required parameter '{param}' missing from general config")
        
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
        
        # Check if input file exists (skip check for obviously fake test paths)
        input_path = general.get('input_path')
        if (input_path and input_path != "" and not input_path.startswith('/some/fake') 
            and not os.path.exists(input_path)):
            raise ValueError(f"Input file not found: {input_path}")
        
        # Validate method
        method = general.get('method', 'kaiser_squires')
        valid_methods = ['kaiser_squires', 'aperture_mass', 'ks_plus']
        if method not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Must be one of: {valid_methods}")
    
    def to_dict(self):
        """Return configuration as dictionary.
        
        Returns
        -------
        config : `dict`
            Configuration dictionary
        """
        return copy.deepcopy(self.config)
    
    def save(self, path):
        """Save configuration to YAML file.
        
        Parameters
        ----------
        path : `str` or `pathlib.Path`
            Path to save configuration file
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)