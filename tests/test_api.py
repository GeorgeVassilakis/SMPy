"""Tests for SMPy API functions."""

import unittest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smpy.config import Config
from smpy.api import map_mass, map_kaiser_squires, map_aperture_mass, map_ks_plus


class TestAPIConfigGeneration(unittest.TestCase):
    """Test that API functions generate correct configurations."""
    
    def setUp(self):
        """Set up temporary files for testing."""
        # Create a temporary FITS file for testing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.fits', delete=False)
        self.temp_file.close()
        self.temp_file_path = self.temp_file.name
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)
    
    def test_map_mass_generates_correct_config(self):
        """Test that map_mass generates correct configuration structure."""
        # We'll test config generation by catching the config before it's executed
        # This is done by mocking the run function
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            # Call the API function
            map_mass(
                data=self.temp_file_path,
                method='kaiser_squires',
                coord_system='ra_dec',
                pixel_scale=0.168,
                output_dir='/test',
                create_snr=True,
                smoothing=3.0
            )
            
            # Check that run was called with a Config object
            self.assertTrue(mock_run.called)
            config_arg = mock_run.call_args[0][0]
            self.assertIsInstance(config_arg, Config)
            
            # Convert to dict and check structure
            config_dict = config_arg.to_dict()
            
            # Check general parameters
            self.assertEqual(config_dict['general']['input_path'], self.temp_file_path)
            self.assertEqual(config_dict['general']['method'], 'kaiser_squires')
            self.assertEqual(config_dict['general']['coordinate_system'], 'radec')
            self.assertEqual(config_dict['general']['radec']['resolution'], 0.168)
            self.assertEqual(config_dict['general']['output_directory'], '/test')
            self.assertEqual(config_dict['general']['create_snr'], True)
            
            # Check method-specific parameters
            self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['sigma'], 3.0)
    
    def test_map_kaiser_squires_sets_correct_method(self):
        """Test that map_kaiser_squires sets method correctly."""
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            map_kaiser_squires(
                data=self.temp_file_path,
                coord_system='ra_dec',
                pixel_scale=0.168,
                smoothing=1.5
            )
            
            config_arg = mock_run.call_args[0][0]
            config_dict = config_arg.to_dict()
            
            # Check method is set correctly
            self.assertEqual(config_dict['general']['method'], 'kaiser_squires')
            self.assertEqual(config_dict['methods']['kaiser_squires']['smoothing']['sigma'], 1.5)
    
    def test_map_aperture_mass_sets_correct_method(self):
        """Test that map_aperture_mass sets method and parameters correctly."""
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            map_aperture_mass(
                data=self.temp_file_path,
                coord_system='ra_dec',
                pixel_scale=0.168,
                filter_type='schneider',
                filter_scale=80
            )
            
            config_arg = mock_run.call_args[0][0]
            config_dict = config_arg.to_dict()
            
            # Check method is set correctly
            self.assertEqual(config_dict['general']['method'], 'aperture_mass')
            
            # Check aperture mass specific parameters
            filter_config = config_dict['methods']['aperture_mass']['filter']
            self.assertEqual(filter_config['type'], 'schneider')
            self.assertEqual(filter_config['scale'], 80)
    
    def test_map_ks_plus_sets_correct_method(self):
        """Test that map_ks_plus sets method and parameters correctly."""
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            map_ks_plus(
                data=self.temp_file_path,
                coord_system='ra_dec',
                pixel_scale=0.168,
                smoothing=2.5,
                inpainting_iterations=150,
                reduced_shear_iterations=5
            )
            
            config_arg = mock_run.call_args[0][0]
            config_dict = config_arg.to_dict()
            
            # Check method is set correctly
            self.assertEqual(config_dict['general']['method'], 'ks_plus')
            
            # Check KS+ specific parameters
            ks_plus_config = config_dict['methods']['ks_plus']
            self.assertEqual(ks_plus_config['smoothing']['sigma'], 2.5)
            self.assertEqual(ks_plus_config['inpainting_iterations'], 150)
            self.assertEqual(ks_plus_config['reduced_shear_iterations'], 5)
    
    def test_pixel_coordinate_system(self):
        """Test API functions work correctly with pixel coordinate system."""
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            map_mass(
                data=self.temp_file_path,
                coord_system='pixel',
                downsample_factor=3,
                method='kaiser_squires'
            )
            
            config_arg = mock_run.call_args[0][0]
            config_dict = config_arg.to_dict()
            
            # Check coordinate system settings
            self.assertEqual(config_dict['general']['coordinate_system'], 'pixel')
            self.assertEqual(config_dict['general']['pixel']['downsample_factor'], 3)
    
    def test_additional_parameters(self):
        """Test that additional parameters are handled correctly."""
        from unittest.mock import patch
        
        with patch('smpy.run.run') as mock_run:
            map_mass(
                data=self.temp_file_path,
                coord_system='ra_dec',
                pixel_scale=0.168,
                g1_col='custom_g1',
                g2_col='custom_g2',
                weight_col='weight_custom',
                mode=['E', 'B'],
                save_fits=True,
                print_timing=True
            )
            
            config_arg = mock_run.call_args[0][0]
            config_dict = config_arg.to_dict()
            
            # Check additional parameters
            self.assertEqual(config_dict['general']['g1_col'], 'custom_g1')
            self.assertEqual(config_dict['general']['g2_col'], 'custom_g2')
            self.assertEqual(config_dict['general']['weight_col'], 'weight_custom')
            self.assertEqual(config_dict['general']['mode'], ['E', 'B'])
            self.assertEqual(config_dict['general']['save_fits'], True)
            self.assertEqual(config_dict['general']['print_timing'], True)
    
    def test_config_validation_called(self):
        """Test that config validation is called."""
        from unittest.mock import patch
        
        # Mock the validation to track if it's called
        with patch('smpy.config.Config.validate') as mock_validate:
            with patch('smpy.run.run'):
                map_mass(
                    data=self.temp_file_path,
                    coord_system='ra_dec',
                    pixel_scale=0.168
                )
                
                # Check that validation was called
                self.assertTrue(mock_validate.called)
    
    def test_invalid_parameters_raise_error(self):
        """Test that invalid parameters raise appropriate errors during validation."""
        # Test missing required parameter
        with self.assertRaises(ValueError):
            map_mass(
                data=self.temp_file_path,
                coord_system='ra_dec'
                # Missing pixel_scale
            )
        
        # Test missing downsample_factor for pixel system
        with self.assertRaises(ValueError):
            map_mass(
                data=self.temp_file_path,
                coord_system='pixel'
                # Missing downsample_factor
            )


if __name__ == '__main__':
    unittest.main()