"""Tests for SMPy configuration system."""

import unittest
import tempfile
import os
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from smpy.config import Config


class TestConfig(unittest.TestCase):
    """Test the Config class functionality."""
    
    def test_from_defaults_kaiser_squires(self):
        """Test loading default config for Kaiser-Squires method."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Check nested structure exists
        self.assertIn('general', config_dict)
        self.assertIn('methods', config_dict)
        self.assertIn('plotting', config_dict)
        self.assertIn('snr', config_dict)
        
        # Check method is set correctly (nested structure)
        self.assertEqual(config_dict['general']['method'], 'kaiser_squires')
        
        # Check KS-specific settings exist in nested methods section
        self.assertIn('kaiser_squires', config_dict['methods'])
        self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['type'], 'gaussian')
        self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['sigma'], 2.0)
        
        # Check plotting settings are in plotting section
        self.assertIn('cmap', config_dict['plotting'])
        self.assertIn('figsize', config_dict['plotting'])
        
        # Check SNR settings are in SNR section
        self.assertIn('shuffle_type', config_dict['snr'])
        self.assertIn('num_shuffles', config_dict['snr'])
        
        # Check coordinate system structure is in general section
        self.assertIn('radec', config_dict['general'])
        self.assertIn('pixel', config_dict['general'])
    
    def test_from_defaults_aperture_mass(self):
        """Test loading default config for aperture mass method."""
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        
        # Check nested structure exists
        self.assertIn('general', config_dict)
        self.assertIn('methods', config_dict)
        self.assertIn('plotting', config_dict)
        
        # Check method is set correctly (nested structure)
        self.assertEqual(config_dict['general']['method'], 'aperture_mass')
        
        # Check aperture mass specific settings exist in nested methods section
        self.assertIn('aperture_mass', config_dict['methods'])
        self.assertEqual(config_dict['methods']['aperture_mass']['filter']['type'], 'schirmer')
        self.assertEqual(config_dict['methods']['aperture_mass']['filter']['scale'], 60)
        self.assertEqual(config_dict['methods']['aperture_mass']['filter']['truncation'], 1.0)
        
        # Check plotting settings are in plotting section
        self.assertIn('cmap', config_dict['plotting'])
        self.assertIn('figsize', config_dict['plotting'])
        
        # Check coordinate system structure is in general section
        self.assertIn('radec', config_dict['general'])
        self.assertIn('pixel', config_dict['general'])
    
    def test_from_defaults_ks_plus(self):
        """Test loading default config for KS+ method."""
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        
        # Check nested structure exists
        self.assertIn('general', config_dict)
        self.assertIn('methods', config_dict)
        self.assertIn('plotting', config_dict)
        
        # Check method is set correctly (nested structure)
        self.assertEqual(config_dict['general']['method'], 'ks_plus')
        
        # Check KS+ specific settings exist in nested methods section
        self.assertIn('ks_plus', config_dict['methods'])
        self.assertIn('inpainting_iterations', config_dict['methods']['ks_plus'])
        self.assertEqual(config_dict['methods']['ks_plus']['inpainting_iterations'], 100)
        self.assertIn('reduced_shear_iterations', config_dict['methods']['ks_plus'])
        self.assertEqual(config_dict['methods']['ks_plus']['reduced_shear_iterations'], 3)
        
        # Check plotting settings are in plotting section
        self.assertIn('cmap', config_dict['plotting'])
        self.assertIn('figsize', config_dict['plotting'])
        
        # Check coordinate system structure is in general section
        self.assertIn('radec', config_dict['general'])
        self.assertIn('pixel', config_dict['general'])
    
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
        """Test updating method-specific parameters in nested structure."""
        config = Config.from_defaults('kaiser_squires')
        
        config.update_from_kwargs(
            smoothing=3.0,
            inpainting_iterations=200,  # KS+ parameter
            filter_type='schirmer'      # Aperture mass parameter
        )
        
        config_dict = config.to_dict()
        
        # Check smoothing is applied to current method (nested structure)
        self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['sigma'], 3.0)
        
        # Check KS+ parameter is set in methods section
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
    # This functionality is no longer needed with the nested config approach.
    
    def test_show_config_full(self):
        """Test show_config() displays full configuration."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            config.show_config()
            output = captured_output.getvalue()
            
            # Check that output contains YAML content (nested structure)
            self.assertIn('general:', output)
            self.assertIn('method: kaiser_squires', output)
            self.assertIn('methods:', output)
            self.assertIn('plotting:', output)
            self.assertTrue(len(output) > 100)  # Should be substantial output
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_show_config_section(self):
        """Test show_config() with section parameter."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            config.show_config(section='methods')
            output = captured_output.getvalue()
            
            # Check that output contains only methods section
            self.assertIn('methods:', output)
            self.assertIn('kaiser_squires:', output)
            self.assertIn('type: gaussian', output)
            self.assertIn('sigma: 2.0', output)
            # Should not contain other sections
            self.assertNotIn('general:', output)
            self.assertNotIn('plotting:', output)
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_show_config_invalid_section(self):
        """Test show_config() with invalid section parameter."""
        config = Config.from_defaults('kaiser_squires')
        
        # Capture stdout to test output
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
            
            # Check that loaded config has expected content (nested structure)
            self.assertIsInstance(loaded_config, dict)
            self.assertEqual(loaded_config['general']['method'], 'aperture_mass')
            self.assertIn('filter', loaded_config['methods']['aperture_mass'])
            self.assertIn('cmap', loaded_config['plotting'])
            
            # Verify it can be used to create a new Config
            new_config = Config(loaded_config)
            self.assertEqual(new_config.to_dict()['general']['method'], 'aperture_mass')
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_from_defaults_nested_structure(self):
        """Test that from_defaults() creates consistent nested structure."""
        # Test all three methods have correct nested structure
        
        # Kaiser-Squires: method-specific config in methods section
        ks_config = Config.from_defaults('kaiser_squires')
        ks_dict = ks_config.to_dict()
        self.assertIn('general', ks_dict)  # Should be nested
        self.assertIn('methods', ks_dict)
        self.assertIn('kaiser_squires', ks_dict['methods'])
        self.assertEqual(ks_dict['general']['method'], 'kaiser_squires')
        
        # Aperture Mass: method-specific config in methods section
        am_config = Config.from_defaults('aperture_mass')
        am_dict = am_config.to_dict()
        self.assertIn('general', am_dict)  # Should be nested
        self.assertIn('methods', am_dict)
        self.assertIn('aperture_mass', am_dict['methods'])
        self.assertEqual(am_dict['general']['method'], 'aperture_mass')
        
        # KS+: method-specific config in methods section
        ksp_config = Config.from_defaults('ks_plus')
        ksp_dict = ksp_config.to_dict()
        self.assertIn('general', ksp_dict)  # Should be nested
        self.assertIn('methods', ksp_dict)
        self.assertIn('ks_plus', ksp_dict['methods'])
        self.assertEqual(ksp_dict['general']['method'], 'ks_plus')
    
    def test_from_defaults_kaiser_squires_smoothing_access(self):
        """Test Kaiser-Squires has smoothing accessible via nested structure."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Verify smoothing is in nested methods section (new access pattern)
        smoothing_config = config_dict['methods']['kaiser_squires']['smoothing']
        self.assertIsNotNone(smoothing_config)
        self.assertEqual(smoothing_config['type'], 'gaussian')
        self.assertEqual(smoothing_config['sigma'], 2.0)
        
        # Verify this matches the expected access pattern: config['methods']['kaiser_squires']['smoothing']
        self.assertTrue('smoothing' in config_dict['methods']['kaiser_squires'])
    
    def test_from_defaults_ks_plus_nested_config_access(self):
        """Test KS+ has config accessible via nested methods section."""
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        
        # Verify ks_plus section exists in methods (new access pattern)
        ks_plus_config = config_dict['methods']['ks_plus']
        self.assertIsNotNone(ks_plus_config)
        self.assertEqual(ks_plus_config['inpainting_iterations'], 100)
        self.assertEqual(ks_plus_config['reduced_shear_iterations'], 3)
        
        # Verify this matches the expected access pattern: config['methods']['ks_plus']
        self.assertTrue('ks_plus' in config_dict['methods'])
    
    def test_from_defaults_aperture_mass_filter_access(self):
        """Test Aperture Mass has filter accessible via nested structure."""
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        
        # Verify filter is in nested methods section (new access pattern)
        filter_config = config_dict['methods']['aperture_mass']['filter']
        self.assertIsNotNone(filter_config)
        self.assertEqual(filter_config['type'], 'schirmer')
        self.assertEqual(filter_config['scale'], 60)
        self.assertEqual(filter_config['truncation'], 1.0)
        
        # Verify this matches the expected access pattern: config['methods']['aperture_mass']['filter']
        self.assertTrue('filter' in config_dict['methods']['aperture_mass'])
    
    def test_nested_structure_consistency(self):
        """Test that all methods produce consistent nested structure."""
        methods = ['kaiser_squires', 'aperture_mass', 'ks_plus']
        
        for method in methods:
            with self.subTest(method=method):
                config = Config.from_defaults(method)
                config_dict = config.to_dict()
                
                # All configs should have these top-level sections
                required_sections = ['general', 'methods', 'plotting', 'snr']
                for section in required_sections:
                    self.assertIn(section, config_dict, f"Section '{section}' missing for method '{method}'")
                
                # General section should have method set correctly
                self.assertEqual(config_dict['general']['method'], method)
                
                # Methods section should contain the specific method
                self.assertIn(method, config_dict['methods'], f"Method '{method}' not found in methods section")
    
    def test_fail_fast_config_access(self):
        """Test that missing required config raises KeyError immediately."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Test that direct access works for existing keys
        self.assertEqual(config_dict['general']['method'], 'kaiser_squires')
        self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['type'], 'gaussian')
        
        # Test that direct access fails fast for missing keys
        with self.assertRaises(KeyError):
            _ = config_dict['nonexistent_section']
        
        with self.assertRaises(KeyError):
            _ = config_dict['general']['nonexistent_param']
        
        with self.assertRaises(KeyError):
            _ = config_dict['methods']['nonexistent_method']
    
    def test_optional_vs_required_parameters(self):
        """Test distinction between optional and required parameters."""
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        
        # Required parameters should exist and be directly accessible
        required_params = [
            ('general', 'method'),
            ('general', 'coordinate_system'),
            ('methods', 'kaiser_squires', 'smoothing', 'type'),
            ('plotting', 'figsize'),
            ('snr', 'num_shuffles')
        ]
        
        for *sections, param in required_params:
            current = config_dict
            for section in sections:
                current = current[section]
            self.assertIn(param, current, f"Required parameter missing: {' -> '.join(sections + [param])}")
        
        # Optional parameters should use .get() and have reasonable behavior
        plotting_config = config_dict['plotting']
        optional_params = ['verbose', 'vmin', 'vmax', 'threshold', 'cluster_center']
        
        for param in optional_params:
            # Test that .get() works without KeyError and returns reasonable values
            value_with_default = plotting_config.get(param, 'TEST_DEFAULT')
            value_without_default = plotting_config.get(param)
            
            # Either the parameter exists with a real value, or .get() returns None/default
            if param in plotting_config:
                # If parameter exists, it should have a meaningful value (not our test default)
                self.assertNotEqual(value_with_default, 'TEST_DEFAULT', 
                                  f"Parameter '{param}' exists but .get() returned default")
                self.assertEqual(value_with_default, value_without_default,
                               f"Parameter '{param}' .get() inconsistency")
            else:
                # If parameter doesn't exist, .get() should return None or default
                self.assertIsNone(value_without_default, 
                                f"Missing parameter '{param}' should return None via .get()")
                self.assertEqual(value_with_default, 'TEST_DEFAULT',
                               f"Missing parameter '{param}' should return default via .get()")
    
    def test_config_actually_works_with_mappers(self):
        """Test that config structure actually works with real mapper classes."""
        # This test prevents "reward hacking" by ensuring the config works in practice
        from smpy.mapping_methods import KaiserSquiresMapper, ApertureMassMapper, KSPlusMapper
        
        methods_and_mappers = [
            ('kaiser_squires', KaiserSquiresMapper),
            ('aperture_mass', ApertureMassMapper), 
            ('ks_plus', KSPlusMapper)
        ]
        
        for method_name, mapper_class in methods_and_mappers:
            with self.subTest(method=method_name):
                # Load config and create mapper
                config = Config.from_defaults(method_name)
                config_dict = config.to_dict()
                
                # This will fail if the nested structure is wrong
                mapper = mapper_class(config_dict)
                
                # Test that mapper can access its configuration correctly
                self.assertEqual(mapper.name, method_name)
                
                # Test that helper properties work (fail if structure is wrong)
                general_config = mapper.general_config
                method_config = mapper.method_config  
                plotting_config = mapper.plotting_config
                
                # Validate actual values, not just existence
                self.assertEqual(general_config['method'], method_name)
                self.assertEqual(general_config['coordinate_system'], 'radec')
                self.assertIsInstance(plotting_config['figsize'], list)
                self.assertGreater(len(plotting_config['figsize']), 0)
                # cmap should be a string colormap name
                self.assertIsInstance(plotting_config['cmap'], str)
                
                # Method-specific validations to ensure configs are meaningful
                if method_name == 'kaiser_squires':
                    smoothing = method_config['smoothing']
                    self.assertEqual(smoothing['type'], 'gaussian')
                    self.assertIsInstance(smoothing['sigma'], (int, float))
                    self.assertGreater(smoothing['sigma'], 0)
                    
                    # Test that the config actually works in practice by creating fake grids
                    import numpy as np
                    fake_g1 = np.random.rand(10, 10)
                    fake_g2 = np.random.rand(10, 10) 
                    # This should not raise an exception if config is correct
                    try:
                        kappa_e, kappa_b = mapper.create_maps(fake_g1, fake_g2)
                        self.assertIsInstance(kappa_e, np.ndarray)
                        self.assertIsInstance(kappa_b, np.ndarray)
                    except Exception as e:
                        self.fail(f"Kaiser-Squires mapper failed with valid config: {e}")
                    
                elif method_name == 'aperture_mass':
                    filter_config = method_config['filter']
                    self.assertIn(filter_config['type'], ['schirmer', 'schneider'])
                    self.assertIsInstance(filter_config['scale'], (int, float))
                    self.assertGreater(filter_config['scale'], 0)
                    
                    # Test actual functionality
                    import numpy as np
                    fake_g1 = np.random.rand(50, 50)  # Larger for aperture mass
                    fake_g2 = np.random.rand(50, 50)
                    try:
                        map_e, map_b = mapper.create_maps(fake_g1, fake_g2)
                        self.assertIsInstance(map_e, np.ndarray)
                        self.assertIsInstance(map_b, np.ndarray)
                    except Exception as e:
                        self.fail(f"Aperture mass mapper failed with valid config: {e}")
                    
                elif method_name == 'ks_plus':
                    self.assertIsInstance(method_config['inpainting_iterations'], int)
                    self.assertGreater(method_config['inpainting_iterations'], 0)
                    self.assertIsInstance(method_config['reduced_shear_iterations'], int)
                    self.assertGreater(method_config['reduced_shear_iterations'], 0)
                    
                    # Test actual functionality 
                    import numpy as np
                    fake_g1 = np.random.rand(20, 20)
                    fake_g2 = np.random.rand(20, 20)
                    try:
                        kappa_e, kappa_b = mapper.create_maps(fake_g1, fake_g2)
                        self.assertIsInstance(kappa_e, np.ndarray)
                        self.assertIsInstance(kappa_b, np.ndarray)
                    except Exception as e:
                        self.fail(f"KS+ mapper failed with valid config: {e}")
    
    def test_config_fails_with_missing_required_sections(self):
        """Test that mappers fail appropriately when required config sections are missing."""
        import numpy as np
        fake_g1 = np.random.rand(10, 10)
        fake_g2 = np.random.rand(10, 10)
        
        # Test Kaiser-Squires with missing smoothing section
        config = Config.from_defaults('kaiser_squires')
        config_dict = config.to_dict()
        del config_dict['methods']['kaiser_squires']  # Remove entire method section
        
        from smpy.mapping_methods import KaiserSquiresMapper
        with self.assertRaises(KeyError) as context:
            mapper = KaiserSquiresMapper(config_dict)
        self.assertIn('kaiser_squires', str(context.exception))
        
        # Test Aperture Mass with missing filter section  
        config = Config.from_defaults('aperture_mass')
        config_dict = config.to_dict()
        del config_dict['methods']['aperture_mass']['filter']
        
        from smpy.mapping_methods import ApertureMassMapper
        mapper = ApertureMassMapper(config_dict)
        with self.assertRaises(KeyError) as context:
            mapper.create_maps(fake_g1, fake_g2)
        self.assertIn('filter', str(context.exception))
        
        # Test with missing general section
        config = Config.from_defaults('ks_plus')
        config_dict = config.to_dict()
        del config_dict['general']
        
        from smpy.mapping_methods import KSPlusMapper
        with self.assertRaises(KeyError) as context:
            mapper = KSPlusMapper(config_dict)
        self.assertIn('general', str(context.exception))


if __name__ == '__main__':
    unittest.main()