"""Tests for optional per-pixel counts map feature.

Verifies configuration toggle presence/propagation and basic accumulation of
per-pixel counts during gridding in pixel coordinates.
"""

import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smpy.config import Config
from smpy.api import map_mass
from smpy.coordinates import PixelSystem
import pandas as pd
import numpy as np


class TestCountsMap(unittest.TestCase):
    def test_default_config_includes_counts_toggle(self):
        cfg = Config.from_defaults('kaiser_squires').to_dict()
        # Present and defaults to False
        self.assertIn('create_counts_map', cfg['general'])
        self.assertFalse(cfg['general']['create_counts_map'])

    def test_api_propagates_counts_toggle(self):
        from unittest.mock import patch

        with patch('smpy.run.run') as mock_run:
            map_mass(
                data='/some/fake/path.fits',  # avoid IO during config generation
                method='kaiser_squires',
                coord_system='pixel',
                downsample_factor=1,
                create_counts_map=True,
            )

            self.assertTrue(mock_run.called)
            cfg_obj = mock_run.call_args[0][0]
            cfg = cfg_obj.to_dict()
            self.assertTrue(cfg['general']['create_counts_map'])

    def test_pixel_counts_accumulate(self):
        # Construct a minimal shear DataFrame with known duplicate binning
        df = pd.DataFrame({
            'coord1': [0.2, 0.7, 1.4],
            'coord2': [0.2, 0.2, 0.6],
            'g1': [0.0, 0.0, 0.0],
            'g2': [0.0, 0.0, 0.0],
            'weight': [1.0, 1.0, 1.0],
        })

        # Pixel system boundaries and config
        pix = PixelSystem()
        scaled_bounds, _ = pix.calculate_boundaries(df['coord1'].values, df['coord2'].values)
        cfg = {
            'general': {
                'coordinate_system': 'pixel',
                'pixel': {
                    'downsample_factor': 1,
                    'coord1': 'X_IMAGE',
                    'coord2': 'Y_IMAGE',
                    'pixel_axis_reference': 'catalog',
                }
            }
        }
        df = pix.transform_coordinates(df)
        g1, g2 = pix.create_grid(df, scaled_bounds, cfg)
        self.assertEqual(g1.shape, g2.shape)
        # Check that the count grid was recorded and sums to 3
        self.assertTrue(hasattr(pix, '_last_count_grid'))
        counts = pix._last_count_grid
        self.assertEqual(int(np.sum(counts)), 3)
        # Expect 2 counts in the first x-bin, first y-bin; 1 count in second x-bin, first y-bin
        # PixelSystem creates arrays as (ny, nx); our data should make ny==1 or 2 depending on range
        ny, nx = counts.shape
        self.assertGreaterEqual(nx, 2)
        self.assertGreaterEqual(ny, 1)
        # Sum per-bin expectations
        self.assertGreaterEqual(counts[0, 0], 2)
        self.assertGreaterEqual(counts[0, 1], 1)


if __name__ == '__main__':
    unittest.main()

