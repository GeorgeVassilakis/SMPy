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
from scipy import fft
from scipy.ndimage import convolve, gaussian_filter

import smpy
from smpy.coordinates import ShearGrid
from smpy.filters.starlet import inverse_starlet_transform_2d, starlet_transform_2d
from smpy.mapping_methods import METHODS
from smpy.mapping_methods.kaiser_squires import convergence_from_shear, shear_from_convergence

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
    changes sign between them, so a mode there that also varies along the
    other axis has no real shear.
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

def test_shear_relations_agree_on_even_grids():
    """A Kaiser-Squires map, turned back into shear and inverted again, is
    unchanged on a grid with even sides, and a convergence mode at the
    highest frequency of one axis only, whose shear is unambiguous, is
    recovered."""
    rng = np.random.default_rng(0)
    ones = np.ones((48, 64))
    mapper = METHODS["kaiser_squires"](smoothing=0)
    maps = mapper.create_maps(ShearGrid(*rng.normal(size=(2, 48, 64)), ones, ones))
    again = mapper.create_maps(ShearGrid(*shear_from_convergence(*maps), ones, ones))
    np.testing.assert_allclose(again, maps, rtol=0, atol=1e-12)

    # kappa_E = gamma1 = cos(pi x), with gamma2 = 0
    alternating = np.cos(np.pi * np.arange(64)) * ones
    map_e, map_b = mapper.create_maps(ShearGrid(alternating, 0 * ones, ones, ones))
    np.testing.assert_allclose(map_e, alternating, rtol=0, atol=1e-12)
    np.testing.assert_allclose(map_b, 0, rtol=0, atol=1e-12)

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

@pytest.mark.parametrize("method, method_settings, degrees", [
    ("kaiser_squires", {}, 15),
    ("aperture_mass", METHOD_SETTINGS["aperture_mass"], 15),
    # KS+'s reduced-shear correction divides by 1 - kappa_E alone, and its
    # wavelet constraint rescales E and B separately, which commutes only
    # with quarter turns
    ("ks_plus", {"reduced_shear_iterations": 1}, 45),
    ("ks_plus", {"reduced_shear_iterations": 1, "use_wavelet_constraints": False}, 15),
], ids=["kaiser_squires", "aperture_mass", "ks_plus", "ks_plus without wavelet constraint"])
def test_rotating_shear_rotates_e_and_b(truth, method, method_settings, degrees):
    """Turning every shear by an angle turns E + iB by twice that angle.

    Turning a shear by phi multiplies g1 + i g2 by exp(2i phi). Kaiser-Squires
    and aperture mass are linear in the shear, and KS+ thresholds E + iB by
    its magnitude, so E + iB is multiplied by the same factor. At 45 degrees
    this swaps E and B: (E, B) -> (-B, E).
    """
    turn = np.exp(2j * np.deg2rad(degrees))
    shear = (truth["nfw_g1"] + 1j * truth["nfw_g2"]) * turn
    turned = truth.copy()
    turned["nfw_g1"], turned["nfw_g2"] = shear.real, shear.imag
    settings = dict(**SHEAR, **COORDINATES["radec"], **method_settings)
    before = smpy.map_mass(truth, method, **settings)
    after = smpy.map_mass(turned, method, **settings)

    expected = (before.maps["E"] + 1j * before.maps["B"]) * turn
    np.testing.assert_allclose(after.maps["E"], expected.real, rtol=0, atol=1e-12)
    np.testing.assert_allclose(after.maps["B"], expected.imag, rtol=0, atol=1e-12)

def test_ks_plus_fills_a_hole_in_a_sparse_field():
    """Inside a hole, KS+ recovers a convergence field made of a few cosine
    modes as Kaiser-Squires does from the complete shear, weakest modes
    included.

    The weakest modes return only if the DCT threshold falls to zero. The
    wavelet constraint and field extension are off, so this checks the
    sparse inpainting alone.
    """
    rng = np.random.default_rng(2)
    coefficients = np.zeros((48, 64), complex)
    for amplitude in (1.0, 0.3, 0.1, 0.03, 0.01, 0.005):
        coefficients[tuple(rng.integers(1, 12, 2))] += amplitude * np.exp(2j * np.pi * rng.random())
    gamma1, gamma2 = shear_from_convergence(fft.idctn(coefficients.real, norm="ortho"),
                                            fft.idctn(coefficients.imag, norm="ortho"))
    complete = np.array(convergence_from_shear(gamma1, gamma2))
    y, x = np.mgrid[:48, :64]
    weights = np.where((y - 24) ** 2 + (x - 30) ** 2 < 36, 0.0, 1.0)
    mapper = METHODS["ks_plus"](smoothing=0, inpainting_iterations=100, reduced_shear_iterations=1,
                                nscales=None, extension_size=0, use_wavelet_constraints=False)
    maps = np.array(mapper.create_maps(ShearGrid(gamma1 * weights, gamma2 * weights, weights, weights)))
    hole = weights == 0
    assert np.abs(maps[:, hole] - complete[:, hole]).max() < 1e-3 * np.abs(complete).max()

def test_ks_plus_beats_kaiser_squires_in_holes(truth):
    """Inside holes in the truth catalogue, KS+ recovers the true convergence
    better than Kaiser-Squires.

    Galaxies in the holes get zero weight, so the map grid is unchanged.
    Each map is compared after removing its mean over the observed pixels,
    which shear does not determine.
    """
    rng = np.random.default_rng(3)
    truth["weight"] = 1.0
    for ra, dec in zip(rng.uniform(truth["ra"].min(), truth["ra"].max(), 8),
                       rng.uniform(truth["dec"].min(), truth["dec"].max(), 8)):
        offset = np.hypot((truth["ra"] - ra) * np.cos(np.deg2rad(dec)), truth["dec"] - dec)
        truth["weight"][offset < 0.02] = 0.0
    kappa = binned_kappa(truth, COORDINATES["radec"])
    settings = dict(weight_col="weight", smoothing=0, **SHEAR, **COORDINATES["radec"])
    errors = {}
    for method in ("kaiser_squires", "ks_plus"):
        result = smpy.map_mass(truth, method, **settings)
        observed, holes = result.grid.weights > 0, result.grid.weights == 0
        difference = (result.maps["E"] - result.maps["E"][observed].mean()) - (kappa - kappa[observed].mean())
        errors[method] = np.sqrt(np.mean(difference[holes] ** 2))
    assert errors["ks_plus"] < 0.8 * errors["kaiser_squires"]

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

def test_ks_plus_snr_is_kaiser_squires_snr_where_their_maps_agree(truth):
    """Where KS+ makes the Kaiser-Squires map, it makes the same SNR map.

    Without empty pixels, field extension or reduced-shear correction, KS+
    inpaints nothing and returns the Kaiser-Squires map, also for every
    shuffled catalogue, as shuffles keep the galaxy positions.
    """
    settings = dict(**SHEAR, **COORDINATES["radec"], smoothing=3.0, create_snr=True, snr={"num_shuffles": 3})
    ks = smpy.map_mass(truth, "kaiser_squires", **settings)
    ks_plus = smpy.map_mass(truth, "ks_plus", extension_size=0, reduced_shear_iterations=1,
                            inpainting_iterations=5, **settings)
    np.testing.assert_array_equal(ks_plus.maps["E"], ks.maps["E"])
    for mode in "EB":
        np.testing.assert_allclose(ks_plus.snr[mode], ks.snr[mode], rtol=1e-12)

def test_aperture_mass_snr_noise_is_the_shape_noise_of_each_pixel():
    """The noise behind an aperture-mass SNR map is the exact shape noise of
    each pixel, in a field where it differs between the two halves.

    Turning each shear g_j by a random angle gives the aperture mass of
    filter Q a variance sum_j Q_j^2 |g_j|^2 / 2, with Q the Schneider et al.
    (1998) filter of order 3.
    """
    y, x = np.mgrid[:81, :81]
    amplitude = np.where(x < 40, 0.06, 0.3)
    shear = amplitude * np.exp(2j * np.random.default_rng(4).uniform(0, np.pi, x.shape))
    table = Table({"x": x.ravel(), "y": y.ravel(), "g1": shear.real.ravel(), "g2": shear.imag.ravel()})
    result = smpy.map_mass(table, "aperture_mass", coord_system="pixel", downsample_factor=1, coord1_col="x",
                           coord2_col="y", filter_type="schneider", filter_scale=4, create_snr=True,
                           snr={"num_shuffles": 200})
    noise = result.maps["E"] / result.snr["E"]

    radius = np.hypot(*np.mgrid[-4:5, -4:5]) / 4
    q = np.where(radius < 1, 20 / (16 * np.pi) * radius**2 * (1 - radius**2) ** 3, 0)
    variance = convolve(amplitude**2, q**2, mode="constant") / 2
    ratio = noise / np.sqrt(variance)
    for half in (x < 40, x >= 40):
        assert abs(ratio[half].mean() - 1) < 0.03
