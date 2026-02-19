"""Unit tests for DS9 x-ray contour overlays in plotting."""

import os
import tempfile
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from smpy.plotting.plot import plot_mass_map, plot_snr_map
from smpy.plotting.utils import read_ds9_ctr


class TestXRayContourPlotting(unittest.TestCase):
    """Test DS9 contour parsing and map-type overlay toggles."""

    def setUp(self):
        """Set up reusable test maps and boundaries."""
        self.data = np.linspace(0.0, 1.0, 100).reshape(10, 10)
        self.scaled_boundaries = {
            "coord1_min": 0.0,
            "coord1_max": 10.0,
            "coord2_min": 0.0,
            "coord2_max": 10.0,
        }
        self.true_boundaries = {
            "coord1_min": 0.0,
            "coord1_max": 10.0,
            "coord2_min": 0.0,
            "coord2_max": 10.0,
        }
        self._temp_files = []

    def tearDown(self):
        """Remove temporary files created by tests."""
        for path in self._temp_files:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_ds9_ctr_parses_line_blocks(self):
        """Parser should split contours by ``line`` directives."""
        ctr_path = self._write_ctr_file(
            """
            # DS9 contour export
            fk5
            1 1
            2 2
            line
            3 3
            4 4
            """
        )

        contours, coord_type = read_ds9_ctr(ctr_path)

        self.assertEqual(coord_type, "fk5")
        self.assertEqual(len(contours), 2)
        self.assertEqual(contours[0].shape, (2, 2))
        self.assertEqual(contours[1].shape, (2, 2))

    def test_convergence_toggle_only(self):
        """Contours should draw on convergence maps when enabled."""
        ctr_path = self._write_ctr_file(
            """
            fk5
            2 2
            4 4
            line
            6 6
            8 8
            """
        )
        config = self._radec_config(
            ctr_path=ctr_path,
            show_on_convergence=True,
            show_on_snr=False,
        )

        fig, ax, _ = plot_mass_map(
            data=self.data,
            scaled_boundaries=self.scaled_boundaries,
            true_boundaries=self.true_boundaries,
            config=config,
            return_handles=True,
        )
        self.assertGreater(len(ax.lines), 0)
        plt.close(fig)

        fig, ax, _ = plot_snr_map(
            data=self.data,
            scaled_boundaries=self.scaled_boundaries,
            true_boundaries=self.true_boundaries,
            config=config,
            return_handles=True,
        )
        self.assertEqual(len(ax.lines), 0)
        plt.close(fig)

    def test_snr_toggle_only(self):
        """Contours should draw on SNR maps when enabled."""
        ctr_path = self._write_ctr_file(
            """
            fk5
            1 9
            5 5
            9 1
            """
        )
        config = self._radec_config(
            ctr_path=ctr_path,
            show_on_convergence=False,
            show_on_snr=True,
        )

        fig, ax, _ = plot_mass_map(
            data=self.data,
            scaled_boundaries=self.scaled_boundaries,
            true_boundaries=self.true_boundaries,
            config=config,
            return_handles=True,
        )
        self.assertEqual(len(ax.lines), 0)
        plt.close(fig)

        fig, ax, _ = plot_snr_map(
            data=self.data,
            scaled_boundaries=self.scaled_boundaries,
            true_boundaries=self.true_boundaries,
            config=config,
            return_handles=True,
        )
        self.assertGreater(len(ax.lines), 0)
        plt.close(fig)

    def test_pixel_image_contours_map_reference(self):
        """Image-space contours should convert for pixel ``axis_reference=map``."""
        ctr_path = self._write_ctr_file(
            """
            image
            1 1
            10 10
            """
        )
        config = self._pixel_config(
            ctr_path=ctr_path,
            show_on_convergence=True,
            show_on_snr=False,
            axis_reference="map",
        )

        fig, ax, _ = plot_mass_map(
            data=self.data,
            scaled_boundaries=self.scaled_boundaries,
            true_boundaries=self.true_boundaries,
            config=config,
            return_handles=True,
        )

        self.assertEqual(len(ax.lines), 1)
        x_data = ax.lines[0].get_xdata()
        y_data = ax.lines[0].get_ydata()
        self.assertAlmostEqual(float(x_data[0]), 0.5)
        self.assertAlmostEqual(float(y_data[0]), 0.5)
        plt.close(fig)

    def _write_ctr_file(self, content):
        """Write contour text to a temporary file."""
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".ctr", delete=False)
        handle.write(content.strip() + "\n")
        handle.close()
        self._temp_files.append(handle.name)
        return handle.name

    def _radec_config(self, ctr_path, show_on_convergence, show_on_snr):
        """Build minimal RA/Dec plotting config for contour tests."""
        return {
            "coordinate_system": "radec",
            "figsize": (4, 3),
            "fontsize": 10,
            "cmap": "viridis",
            "xlabel": "auto",
            "ylabel": "auto",
            "plot_title": "Test",
            "gridlines": False,
            "vmin": None,
            "vmax": None,
            "threshold": None,
            "verbose": False,
            "cluster_center": None,
            "scaling": {"type": "linear"},
            "xray_contours": {
                "ctr_file": ctr_path,
                "show_on_convergence": show_on_convergence,
                "show_on_snr": show_on_snr,
                "color": "cyan",
                "linewidth": 0.8,
                "alpha": 0.7,
            },
        }

    def _pixel_config(self, ctr_path, show_on_convergence, show_on_snr, axis_reference):
        """Build minimal pixel plotting config for contour tests."""
        config = self._radec_config(ctr_path, show_on_convergence, show_on_snr)
        config["coordinate_system"] = "pixel"
        config["axis_reference"] = axis_reference
        return config


if __name__ == "__main__":
    unittest.main()
