"""Tests that SMPy's maps obey known physics.

Each test compares SMPy with an answer known independently of it: exact
algebra, a change to the input with a known effect on the maps, or the
noiseless NFW cluster of the truth catalogue.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table
from scipy.ndimage import gaussian_filter

import smpy
from smpy.coordinates import ShearGrid
from smpy.mapping_methods import METHODS

TRUTH = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_truth.fits"
SHEAR = dict(g1_col="nfw_g1", g2_col="nfw_g2")
COORDINATES = {
    "radec": dict(pixel_scale=0.4),
    "pixel": dict(coord_system="pixel", downsample_factor=170, coord1_col="x_image", coord2_col="y_image"),
}
# The default aperture-mass filter is wider than the truth map
METHOD_SETTINGS = {"kaiser_squires": {}, "aperture_mass": {"filter_scale": 10}, "ks_plus": {}}

@pytest.fixture
def truth():
    """The truth catalogue with its convergence and a column of zeros.

    The catalogue holds reduced shear g and magnification mu, so the
    convergence follows from mu = 1 / ((1 - kappa)^2 (1 - |g|^2)).
    """
    table = Table.read(TRUTH, hdu=1)
    g_squared = table["nfw_g1"] ** 2 + table["nfw_g2"] ** 2
    table["kappa"] = 1 - 1 / np.sqrt(table["nfw_mu"] * (1 - g_squared))
    table["zero"] = 0.0
    return table

def binned_kappa(truth, coordinates):
    """Return the true convergence averaged over each map pixel.

    The convergence passes through SMPy's gridding in place of g1, so it is
    binned onto exactly the pixels of the maps.
    """
    return smpy.map_mass(truth, g1_col="kappa", g2_col="zero", **coordinates).grid.g1

def test_kaiser_squires_inverts_shear():
    """Kaiser-Squires returns the E and B convergence that made a shear field.

    The grid has odd sides. On an even side the highest frequency stands for
    both +1/2 and -1/2 cycles per pixel, and the shear term 2 k1 k2 / k^2
    changes sign between them, so a convergence mode there has no real shear.
    """
    rng = np.random.default_rng(0)
    kappa = rng.normal(size=(47, 63)) + 1j * rng.normal(size=(47, 63))
    kappa -= kappa.mean()  # A uniform convergence makes no shear
    k1, k2 = np.meshgrid(np.fft.fftfreq(63), np.fft.fftfreq(47))
    k_squared = np.where((k1 == 0) & (k2 == 0), 1.0, k1**2 + k2**2)
    gamma = np.fft.ifft2((k1**2 - k2**2 + 2j * k1 * k2) / k_squared * np.fft.fft2(kappa))
    ones = np.ones(kappa.shape)
    grid = ShearGrid(gamma.real, gamma.imag, ones, ones)

    map_e, map_b = METHODS["kaiser_squires"](smoothing=0).create_maps(grid)
    np.testing.assert_allclose(map_e, kappa.real, rtol=0, atol=1e-12)
    np.testing.assert_allclose(map_b, kappa.imag, rtol=0, atol=1e-12)

@pytest.mark.parametrize("coord_system", COORDINATES)
@pytest.mark.parametrize("method", METHOD_SETTINGS)
def test_truth_catalog_recovers_cluster(truth, method, coord_system):
    """Each method finds the NFW cluster in E, where it is, with little B.

    Aperture mass is a filtered statistic, not convergence, so only its peak
    and B are checked.
    """
    coordinates = COORDINATES[coord_system]
    result = smpy.map_mass(truth, method, **SHEAR, **coordinates, **METHOD_SETTINGS[method])
    map_e, map_b = result.maps["E"], result.maps["B"]
    kappa = gaussian_filter(binned_kappa(truth, coordinates), 2.0)  # Smoothed like the maps

    peak = np.unravel_index(np.argmax(map_e), map_e.shape)
    true_peak = np.unravel_index(np.argmax(kappa), kappa.shape)
    assert max(abs(np.subtract(peak, true_peak))) <= 1
    assert np.std(map_b) < 0.3 * np.std(map_e)
    if method != "aperture_mass":
        assert np.corrcoef(map_e.ravel(), kappa.ravel())[0, 1] > 0.9

@pytest.mark.parametrize("method", ["kaiser_squires", "aperture_mass"])
def test_rotating_shear_swaps_e_and_b(truth, method):
    """Turning every shear by 45 degrees swaps the E and B maps.

    (g1, g2) -> (-g2, g1) multiplies the complex shear, and so the complex
    convergence E + iB, by i: (E, B) -> (-B, E). On RA/Dec maps g2 changes
    sign because RA increases to the left on the sky, which reverses the
    turn: (E, B) -> (B, -E).
    """
    turned = truth.copy()
    turned["nfw_g1"], turned["nfw_g2"] = -truth["nfw_g2"], truth["nfw_g1"]
    settings = dict(**SHEAR, **COORDINATES["radec"], **METHOD_SETTINGS[method])
    before = smpy.map_mass(truth, method, **settings)
    after = smpy.map_mass(turned, method, **settings)

    np.testing.assert_allclose(after.maps["E"], before.maps["B"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(after.maps["B"], -before.maps["E"], rtol=0, atol=1e-12)

def test_pixel_wcs_places_galaxies_in_their_pixels(truth):
    """The WCS puts every galaxy in the map pixel that counted it."""
    result = smpy.map_mass(truth, **SHEAR, **COORDINATES["pixel"])
    x, y = result.wcs.world_to_pixel_values(truth["x_image"], truth["y_image"])
    ny, nx = result.grid.counts.shape
    counts, _, _ = np.histogram2d(y, x, bins=(ny, nx), range=((-0.5, ny - 0.5), (-0.5, nx - 0.5)))
    np.testing.assert_array_equal(counts, result.grid.counts)
