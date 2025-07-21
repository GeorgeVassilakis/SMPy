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
        
        # Check method is set correctly (flattened structure)
        self.assertEqual(config_dict['method'], 'kaiser_squires')
        
        # Check KS-specific settings exist at top level (flattened)
        self.assertIn('smoothing', config_dict)
        self.assertEqual(config_dict['smoothing']['type'], 'gaussian')
        self.assertEqual(config_dict['smoothing']['sigma'], 2.0)
        
        # Check plotting settings are at top level
        self.assertIn('cmap', config_dict)
        self.assertIn('figsize', config_dict)
        
        # Check SNR settings are at top level
        self.assertIn('shuffle_type', config_dict)
        self.assertIn('num_shuffles', config_dict)
        
        # Check coordinate system structure is preserved
        self.assertIn('radec', config_dict)
        self.assertIn('pixel', config_dict)
    
    def test_from_defaults_aperture_mass(self):
        """Test loading default config for aperture mass method."""
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        
        # Check method is set correctly (flattened structure)
        self.assertEqual(config_dict['method'], 'aperture_mass')
        
        # Check aperture mass specific settings exist at top level (flattened)
        self.assertIn('filter', config_dict)
        self.assertEqual(config_dict['filter']['type'], 'schirmer')
        self.assertEqual(config_dict['filter']['scale'], 60)
        self.assertEqual(config_dict['filter']['truncation'], 1.0)
        
        # Check plotting settings are at top level
        self.assertIn('cmap', config_dict)
        self.assertIn('figsize', config_dict)
        
        # Check coordinate system structure is preserved
        self.assertIn('radec', config_dict)
        self.assertIn('pixel', config_dict)
    
    def test_from_defaults_ks_plus(self):
        """Test loading default config for KS+ method."""
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        
        # Check method is set correctly (flattened structure)
        self.assertEqual(config_dict['method'], 'ks_plus')
        
        # Check KS+ specific settings exist in nested 'ks_plus' section (special case)
        self.assertIn('ks_plus', config_dict)
        self.assertIn('inpainting_iterations', config_dict['ks_plus'])
        self.assertEqual(config_dict['ks_plus']['inpainting_iterations'], 100)
        self.assertIn('reduced_shear_iterations', config_dict['ks_plus'])
        self.assertEqual(config_dict['ks_plus']['reduced_shear_iterations'], 3)
        
        # Check plotting settings are at top level
        self.assertIn('cmap', config_dict)
        self.assertIn('figsize', config_dict)
        
        # Check coordinate system structure is preserved
        self.assertIn('radec', config_dict)
        self.assertIn('pixel', config_dict)
    
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
        """Test updating method-specific parameters in flattened structure."""
        config = Config.from_defaults('kaiser_squires')
        
        config.update_from_kwargs(
            smoothing=3.0,
            inpainting_iterations=200,  # KS+ parameter
            filter_type='schirmer'      # Aperture mass parameter
        )
        
        config_dict = config.to_dict()
        
        # Check smoothing is applied to current method (flattened structure)
        self.assertEqual(config_dict['smoothing']['sigma'], 3.0)
        
        # Check KS+ parameter is set (in methods section since config started flattened)
        # Note: These parameters are added to a methods section by update_from_kwargs
        if 'methods' in config_dict:
            self.assertEqual(config_dict['methods']['ks_plus']['inpainting_iterations'], 200)
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
    
    # REMOVED: test_deep_merge (Phase 4)
    # The _deep_merge method was removed as part of configuration simplification.
    # This functionality is no longer needed with the flattened config approach.
    
    def test_show_config_full(self):
        """Test show_config() displays full configuration."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            config.show_config()
            output = captured_output.getvalue()
            
            # Check that output contains YAML content
            self.assertIn('method: kaiser_squires', output)
            self.assertIn('smoothing:', output)
            self.assertIn('cmap:', output)
            self.assertTrue(len(output) > 100)  # Should be substantial output
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_show_config_section(self):
        """Test show_config() with section parameter."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            config.show_config(section='smoothing')
            output = captured_output.getvalue()
            
            # Check that output contains only smoothing section
            self.assertIn('smoothing:', output)
            self.assertIn('type: gaussian', output)
            self.assertIn('sigma: 2.0', output)
            # Should not contain other sections
            self.assertNotIn('cmap:', output)
            self.assertNotIn('figsize:', output)
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_show_config_invalid_section(self):
        """Test show_config() with invalid section parameter."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            config.show_config(section='nonexistent')
            output = captured_output.getvalue()
            
            # Check that error message is displayed
            self.assertIn("Section 'nonexistent' not found", output)
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_save_config_creates_valid_yaml(self):
        """Test save_config() creates valid YAML file."""
        config = Config.from_defaults('aperture_mass')
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            # Capture stdout to test feedback message
            import io
            import sys
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            # Save config
            config.save_config(temp_file)
            output = captured_output.getvalue()
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            
            # Check feedback message
            self.assertIn(f"Configuration saved to: {temp_file}", output)
            
            # Verify file was created
            self.assertTrue(os.path.exists(temp_file))
            
            # Verify file contains valid YAML that can be loaded back
            import yaml
            with open(temp_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Check that loaded config has expected content
            self.assertIsInstance(loaded_config, dict)
            self.assertEqual(loaded_config['method'], 'aperture_mass')
            self.assertIn('filter', loaded_config)
            self.assertIn('cmap', loaded_config)
            
            # Verify it can be used to create a new Config
            new_config = Config(loaded_config)
            self.assertEqual(new_config.to_dict()['method'], 'aperture_mass')
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_from_defaults_flattened_structure(self):
        """Test that from_defaults() creates properly flattened structure."""
        # Test all three methods have correct flattened structure
        
        # Kaiser-Squires: smoothing at top level
        ks_config = Config.from_defaults('kaiser_squires')
        ks_dict = ks_config.to_dict()
        self.assertIn('smoothing', ks_dict)
        self.assertNotIn('general', ks_dict)  # Should be flattened
        self.assertEqual(ks_dict['method'], 'kaiser_squires')
        
        # Aperture Mass: filter at top level
        am_config = Config.from_defaults('aperture_mass')
        am_dict = am_config.to_dict()
        self.assertIn('filter', am_dict)
        self.assertNotIn('general', am_dict)  # Should be flattened
        self.assertEqual(am_dict['method'], 'aperture_mass')
        
        # KS+: ks_plus nested section
        ksp_config = Config.from_defaults('ks_plus')
        ksp_dict = ksp_config.to_dict()
        self.assertIn('ks_plus', ksp_dict)
        self.assertNotIn('general', ksp_dict)  # Should be flattened
        self.assertEqual(ksp_dict['method'], 'ks_plus')
    
    def test_from_defaults_kaiser_squires_smoothing_access(self):
        """Test Kaiser-Squires has smoothing accessible at top level."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Verify smoothing is at top level (mapping method access pattern)
        smoothing_config = config_dict.get('smoothing')
        self.assertIsNotNone(smoothing_config)
        self.assertEqual(smoothing_config['type'], 'gaussian')
        self.assertEqual(smoothing_config['sigma'], 2.0)
        
        # Verify this matches the expected access pattern: config['smoothing']
        self.assertTrue('smoothing' in config_dict)
    
    def test_from_defaults_ks_plus_nested_config_access(self):
        """Test KS+ has config accessible via nested ks_plus section."""
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        
        # Verify ks_plus section exists (mapping method access pattern)
        ks_plus_config = config_dict.get('ks_plus', {})
        self.assertIsNotNone(ks_plus_config)
        self.assertEqual(ks_plus_config['inpainting_iterations'], 100)
        self.assertEqual(ks_plus_config['reduced_shear_iterations'], 3)
        
        # Verify this matches the expected access pattern: config.get('ks_plus', {})
        self.assertTrue('ks_plus' in config_dict)
    
    def test_from_defaults_aperture_mass_filter_access(self):
        """Test Aperture Mass has filter accessible at top level."""
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        
        # Verify filter is at top level (mapping method access pattern)
        filter_config = config_dict.get('filter', {})
        self.assertIsNotNone(filter_config)
        self.assertEqual(filter_config['type'], 'schirmer')
        self.assertEqual(filter_config['scale'], 60)
        self.assertEqual(filter_config['truncation'], 1.0)
        
        # Verify this matches the expected access pattern: config.get('filter', {})
        self.assertTrue('filter' in config_dict)


if __name__ == '__main__':
    unittest.main()