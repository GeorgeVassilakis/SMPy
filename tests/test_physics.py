"""Tests that SMPy's maps obey known physics.

Each test compares SMPy with an answer known independently of it: exact
algebra, a change to the input with a known effect on the maps, the noiseless
NFW cluster of the truth catalogue, or a lens whose shear is built on the
sphere by astropy.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.ndimage import gaussian_filter

import smpy
from smpy.coordinates import ShearGrid
from smpy.filters.starlet import inverse_starlet_transform_2d, starlet_transform_2d
from smpy.mapping_methods import METHODS
from smpy.mapping_methods.kaiser_squires import shear_from_convergence

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

def wide_field_lens():
    """A 12-degree field at Dec -60 with pure tangential shear around a lens
    4.7 degrees from its centre.

    Each galaxy's shear is tangential to the great circle towards the lens,
    in the galaxy's own west-north frame, so no map projection is involved.
    """
    rng = np.random.default_rng(0)
    dec = rng.uniform(-66, -54, 60000)
    ra = 30 + rng.uniform(-6, 6, dec.size) / np.cos(np.deg2rad(dec))
    galaxies, lens = SkyCoord(ra, dec, unit="deg"), SkyCoord(37, -57, unit="deg")
    angle = galaxies.position_angle(lens).rad + np.pi / 2  # Towards the lens, from west to north
    gamma_t = 0.05 * np.exp(-galaxies.separation(lens).deg ** 2 / (2 * 0.6**2))
    shear = -gamma_t * np.exp(2j * angle)
    return Table({"ra": ra, "dec": dec, "g1": shear.real, "g2": shear.imag})

def counts_from_wcs(result, x, y):
    """Bin galaxies at WCS pixel positions (x, y) onto the result's map."""
    ny, nx = result.grid.counts.shape
    counts, _, _ = np.histogram2d(y, x, bins=(ny, nx), range=((-0.5, ny - 0.5), (-0.5, nx - 0.5)))
    return counts

def binned_kappa(truth, coordinates):
    """Return the true convergence averaged over each map pixel.

    The convergence passes through SMPy's gridding in place of g1, so it is
    binned onto exactly the pixels of the maps.
    """
    return smpy.map_mass(truth, g1_col="kappa", g2_col="zero", **coordinates).grid.g1

def test_kaiser_squires_relations_are_exact():
    """SMPy's convergence-to-shear relation and Kaiser-Squires inversion
    match the textbook relations, for E and B alike.

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

    np.testing.assert_allclose(shear_from_convergence(kappa.real, kappa.imag), [gamma.real, gamma.imag],
                               rtol=0, atol=1e-12)
    map_e, map_b = METHODS["kaiser_squires"](smoothing=0).create_maps(grid)
    np.testing.assert_allclose(map_e, kappa.real, rtol=0, atol=1e-12)
    np.testing.assert_allclose(map_b, kappa.imag, rtol=0, atol=1e-12)

def test_inverse_starlet_transform_restores_image():
    """The inverse starlet transform returns the image that was transformed."""
    image = np.random.default_rng(0).normal(size=(48, 64))
    restored = inverse_starlet_transform_2d(starlet_transform_2d(image, 4))
    np.testing.assert_allclose(restored, image, rtol=0, atol=1e-12)

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
    convergence E + iB, by i: (E, B) -> (-B, E).
    """
    turned = truth.copy()
    turned["nfw_g1"], turned["nfw_g2"] = -truth["nfw_g2"], truth["nfw_g1"]
    settings = dict(**SHEAR, **COORDINATES["radec"], **METHOD_SETTINGS[method])
    before = smpy.map_mass(truth, method, **settings)
    after = smpy.map_mass(turned, method, **settings)

    np.testing.assert_allclose(after.maps["E"], -before.maps["B"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(after.maps["B"], before.maps["E"], rtol=0, atol=1e-12)

def test_lens_far_from_field_centre_makes_no_b_mode():
    """Tangential shear around a lens far from the centre of a wide field
    maps to E with almost no B.

    Away from the field centre the map's y axis is no longer north, so this
    holds only if each galaxy's shear is turned into the map frame.
    """
    result = smpy.map_mass(wide_field_lens(), pixel_scale=6.0)
    assert np.std(result.maps["B"]) < 0.1 * np.std(result.maps["E"])

def test_radec_wcs_places_every_galaxy_in_its_pixel():
    """Every galaxy of a wide field is counted, in the map pixel where the
    TAN WCS puts it."""
    table = wide_field_lens()
    result = smpy.map_mass(table, pixel_scale=6.0)
    x, y = result.wcs.world_to_pixel(SkyCoord(table["ra"], table["dec"], unit="deg"))
    counts = counts_from_wcs(result, x, y)
    assert counts.sum() == len(table)
    np.testing.assert_array_equal(counts, result.grid.counts)

def test_pixel_wcs_places_every_galaxy_in_its_pixel(truth):
    """Every galaxy is counted, in the map pixel where the WCS puts it."""
    result = smpy.map_mass(truth, **SHEAR, **COORDINATES["pixel"])
    counts = counts_from_wcs(result, *result.wcs.world_to_pixel_values(truth["x_image"], truth["y_image"]))
    assert counts.sum() == len(truth)
    np.testing.assert_array_equal(counts, result.grid.counts)
