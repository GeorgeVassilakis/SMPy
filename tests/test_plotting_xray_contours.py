"""Behaviour tests for DS9 x-ray contour overlays on SMPy plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import smpy

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_annular.fits"
PIXEL = dict(coord_system="pixel", downsample_factor=170, coord1_col="X_IMAGE_se", coord2_col="Y_IMAGE_se",
             g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")

@pytest.fixture
def result():
    """Pixel-coordinate maps of the example catalog, with SNR maps."""
    return smpy.map_mass(CATALOG, create_snr=True, snr={"num_shuffles": 2}, **PIXEL)

@pytest.mark.parametrize("on_convergence, on_snr", [(True, False), (False, True)])
def test_contours_follow_their_switches(result, tmp_path, on_convergence, on_snr):
    """Each contour in the file is drawn on exactly the maps its switches
    select."""
    ctr_file = tmp_path / "xray.ctr"
    ctr_file.write_text("# DS9 contour export\nimage\n10 10\n20 20\nline\n30 30\n40 40\n")
    contours = {"ctr_file": str(ctr_file), "show_on_convergence": on_convergence, "show_on_snr": on_snr}
    for snr, shown in ((False, on_convergence), (True, on_snr)):
        fig, ax, _ = result.plot("E", snr=snr, xray_contours=contours)
        assert len(ax.lines) == (2 if shown else 0)
        plt.close(fig)

@pytest.mark.parametrize("axis_reference", ["catalog", "map"])
def test_image_contours_land_on_their_pixels(result, tmp_path, axis_reference):
    """A contour point at DS9 image pixel (i, j) is drawn at the centre of map
    pixel (i, j)."""
    ctr_file = tmp_path / "xray.ctr"
    ctr_file.write_text("image\n1 1\n3 2\n")
    contours = {"ctr_file": str(ctr_file), "show_on_convergence": True}
    fig, ax, image = result.plot("E", axis_reference=axis_reference, xray_contours=contours)
    left, right, bottom, top = image.get_extent()
    ny, nx = result.maps["E"].shape
    pixels = np.array([[1, 1], [3, 2]])
    expected = np.column_stack((left + (pixels[:, 0] - 0.5) * (right - left) / nx,
                                bottom + (pixels[:, 1] - 0.5) * (top - bottom) / ny))
    np.testing.assert_allclose(ax.lines[0].get_xydata(), expected)
    plt.close(fig)
