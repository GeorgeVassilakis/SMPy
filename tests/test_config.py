"""Tests for SMPy configuration system."""

import unittest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smpy.config import Config


class TestConfig(unittest.TestCase):
    """Test the Config class functionality."""
    
    def test_from_defaults_kaiser_squires(self):
        """Test loading default config for Kaiser-Squires method."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Check basic structure
        self.assertIn('general', config_dict)
        self.assertIn('methods', config_dict)
        self.assertIn('plotting', config_dict)
        self.assertIn('snr', config_dict)
        
        # Check method is set correctly
        self.assertEqual(config_dict['general']['method'], 'kaiser_squires')
        
        # Check KS-specific settings exist
        self.assertIn('kaiser_squires', config_dict['methods'])
        self.assertIn('smoothing', config_dict['methods']['kaiser_squires'])
    
    def test_from_defaults_aperture_mass(self):
        """Test loading default config for aperture mass method."""
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        
        # Check method is set correctly
        self.assertEqual(config_dict['general']['method'], 'aperture_mass')
        
        # Check aperture mass specific settings exist
        self.assertIn('aperture_mass', config_dict['methods'])
        self.assertIn('filter', config_dict['methods']['aperture_mass'])
    
    def test_from_defaults_ks_plus(self):
        """Test loading default config for KS+ method."""
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        
        # Check method is set correctly
        self.assertEqual(config_dict['general']['method'], 'ks_plus')
        
        # Check KS+ specific settings exist
        self.assertIn('ks_plus', config_dict['methods'])
        self.assertIn('inpainting_iterations', config_dict['methods']['ks_plus'])
    
    def test_update_from_kwargs_basic(self):
        """Test updating config from basic kwargs."""
        config = Config.from_defaults('kaiser_squires')
        
        config.update_from_kwargs(
            data='/path/to/catalog.fits',
            coord_system='radec',
            pixel_scale=0.168,
            output_dir='/output'
        )
        
        config_dict = config.to_dict()
        
        # Check basic parameters are set
        self.assertEqual(config_dict['general']['input_path'], '/path/to/catalog.fits')
        self.assertEqual(config_dict['general']['coordinate_system'], 'radec')
        self.assertEqual(config_dict['general']['radec']['resolution'], 0.168)
        self.assertEqual(config_dict['general']['output_directory'], '/output')
    
    def test_update_from_kwargs_pixel_system(self):
        """Test updating config for pixel coordinate system."""
        config = Config.from_defaults('kaiser_squires')
        
        config.update_from_kwargs(
            data='/path/to/catalog.fits',
            coord_system='pixel',
            downsample_factor=2
        )
        
        config_dict = config.to_dict()
        
        # Check pixel system parameters
        self.assertEqual(config_dict['general']['coordinate_system'], 'pixel')
        self.assertEqual(config_dict['general']['pixel']['downsample_factor'], 2)
    
    def test_update_from_kwargs_method_specific(self):
        """Test updating method-specific parameters."""
        config = Config.from_defaults('kaiser_squires')
        
        config.update_from_kwargs(
            smoothing=3.0,
            inpainting_iterations=200,  # KS+ parameter
            filter_type='schirmer'      # Aperture mass parameter
        )
        
        config_dict = config.to_dict()
        
        # Check smoothing is applied to current method
        self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['sigma'], 3.0)
        
        # Check KS+ parameter is set
        self.assertEqual(config_dict['methods']['ks_plus']['inpainting_iterations'], 200)
        
        # Check aperture mass parameter is set
        self.assertEqual(config_dict['methods']['aperture_mass']['filter']['type'], 'schirmer')
    
    def test_validation_success(self):
        """Test successful validation with all required parameters."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fits', delete=False) as f:
            temp_file = f.name
        
        try:
            config = Config.from_defaults('kaiser_squires')
            config.update_from_kwargs(
                data=temp_file,
                coord_system='radec',
                pixel_scale=0.168
            )
            
            # Should not raise any exception
            config.validate()
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validation_missing_required_params(self):
        """Test validation fails when required parameters are missing."""
        config = Config.from_defaults('kaiser_squires')
        
        # Set a data path and coord system but missing pixel_scale for radec system
        config.update_from_kwargs(data='/some/fake/path.fits', coord_system='radec')
        
        with self.assertRaises(ValueError):
            config.validate()
    
    def test_validation_missing_pixel_scale_for_radec(self):
        """Test validation fails when pixel_scale missing for radec system."""
        config = Config.from_defaults('kaiser_squires')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fits', delete=False) as f:
            temp_file = f.name
        
        try:
            config.update_from_kwargs(
                data=temp_file,
                coord_system='radec'
                # Missing pixel_scale
            )
            
            with self.assertRaises(ValueError) as context:
                config.validate()
            
            self.assertIn('pixel_scale', str(context.exception))
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validation_missing_downsample_for_pixel(self):
        """Test validation fails when downsample_factor missing for pixel system."""
        config = Config.from_defaults('kaiser_squires')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fits', delete=False) as f:
            temp_file = f.name
        
        try:
            config.update_from_kwargs(
                data=temp_file,
                coord_system='pixel'
                # Missing downsample_factor
            )
            
            with self.assertRaises(ValueError) as context:
                config.validate()
            
            self.assertIn('downsample_factor', str(context.exception))
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_validation_invalid_method(self):
        """Test validation fails for invalid method."""
        config = Config({'general': {'method': 'invalid_method'}})
        
        with self.assertRaises(ValueError) as context:
            config.validate()
        
        self.assertIn('Invalid method', str(context.exception))
    
    def test_deep_merge(self):
        """Test deep merging of configuration dictionaries."""
        dict1 = {
            'general': {'method': 'kaiser_squires', 'output_dir': '.'},
            'methods': {'kaiser_squires': {'smoothing': {'sigma': 2.0}}}
        }
        
        dict2 = {
            'general': {'create_snr': True},
            'methods': {'kaiser_squires': {'smoothing': {'type': 'gaussian'}}}
        }
        
        merged = Config._deep_merge(dict1, dict2)
        
        # Check both values are preserved
        self.assertEqual(merged['general']['method'], 'kaiser_squires')
        self.assertEqual(merged['general']['create_snr'], True)
        
        # Check nested merge worked
        ks_smoothing = merged['methods']['kaiser_squires']['smoothing']
        self.assertEqual(ks_smoothing['sigma'], 2.0)
        self.assertEqual(ks_smoothing['type'], 'gaussian')


if __name__ == '__main__':
    unittest.main()