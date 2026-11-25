"""JAX transform tests for the public array mapping API."""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

import smpy

jax = pytest.importorskip("jax")
from smpy.jax import ArrayMapper  # noqa: E402

jnp = jax.numpy
enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_annular.fits"
CATALOG_OPTIONS = dict(pixel_scale=0.4, g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")

@pytest.fixture(autouse=True)
def _double_precision():
    """Use double precision for analytic and finite-difference checks."""
    with enable_x64(True):
        yield

def test_sky_catalog_position_differentiates_shear_rotation():
    """Moving a source off the TAN meridian changes its shear frame."""
    mapper = ArrayMapper()
    ra0, dec0, shear = 41.0, -30.0, 0.2

    def projected_g2(ra):
        result = mapper.map_catalog(
            jnp.array([ra]), jnp.array([dec0]), jnp.array([shear]), jnp.zeros(1),
            shape=(3, 3), origin=(-1.5, -1.5), tangent_point=(ra0, dec0),
        )
        return result.grid.g2.sum()

    value, derivative = jax.jit(jax.value_and_grad(projected_g2))(ra0)
    # At the central meridian the local north rotates at sin(dec0)
    # radians per radian of RA. The shear rotates as a spin-two field.
    expected = -2 * shear * np.sin(np.deg2rad(dec0)) * np.pi / 180
    np.testing.assert_allclose(value, 0.0, atol=1e-14)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)

def test_kaiser_squires_smoothing_derivative():
    """A traced Gaussian sigma changes the gain of an axial Fourier mode."""
    mapper = ArrayMapper("kaiser_squires", smoothing=2.0)
    frequency = 4 * np.pi / 17
    mode = np.cos(frequency * (np.arange(17) - 8))
    g1 = jnp.broadcast_to(jnp.asarray(mode), (9, 17))
    g2 = -0.3 * g1
    sigma = 1.1

    def objective(width):
        maps = mapper.map_grid(g1, g2, smoothing=width)
        return maps["E"][4, 8], maps["B"][4, 8]

    (value_e, value_b), derivative = jax.jit(jax.value_and_grad(objective, has_aux=True))(sigma)

    # At the central pixel the Gaussian footprint lies inside the grid.
    offsets = np.arange(-4, 5)
    profile = np.exp(-0.5 * (offsets / sigma)**2)
    profile /= profile.sum()
    cosine = np.cos(frequency * offsets)
    gain = profile @ cosine
    expected_derivative = profile @ ((cosine - gain) * offsets**2) / sigma**3
    np.testing.assert_allclose([value_e, value_b], [gain, -0.3 * gain], rtol=1e-12)
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12)

@pytest.mark.parametrize("gap_width", [1, 3], ids=["zero-gap-variance", "wavelet-rescaling"])
def test_sparse_ks_plus_derivatives(gap_width):
    """Sparse KS+ differentiates shear and sigma through wavelet inpainting.

    Parameters
    ----------
    gap_width : `int`
        Side length of the square hole. A one-pixel hole has zero variance.
    """
    mapper = ArrayMapper(
        "ks_plus", smoothing=1.0, inpainting_iterations=5, reduced_shear_iterations=1,
        nscales=2, extension_size=0, use_wavelet_constraints=True,
    )
    rng = np.random.default_rng(13)
    shape = (9, 11)
    weights = np.ones(shape)
    weights[3:3 + gap_width, 4:4 + gap_width] = 0
    shear = rng.normal(scale=0.03, size=(3, *shape))
    shear[:, weights == 0] = 0
    g1, g2, direction = jnp.asarray(shear)
    probe = jnp.asarray(np.where(weights == 0, rng.normal(size=shape), 0))

    def objective(amplitude, sigma):
        maps = mapper.map_grid(g1 + amplitude * direction, g2, weights, smoothing=sigma)
        return jnp.sum(probe * (maps["E"] + 0.6 * maps["B"]))

    evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))
    amplitude, sigma, step = 0.0, 0.7, 1e-5
    _, gradient = evaluate(amplitude, sigma)
    expected = np.array([
        (evaluate(amplitude + step, sigma)[0] - evaluate(amplitude - step, sigma)[0]) / (2 * step),
        (evaluate(amplitude, sigma + step)[0] - evaluate(amplitude, sigma - step)[0]) / (2 * step),
    ])
    assert np.all(np.abs(expected) > 1e-6)
    np.testing.assert_allclose(gradient, expected, rtol=1e-5, atol=1e-8)

def test_aperture_scale_and_cutoff_derivatives():
    """An axial point shear follows the Schneider profile and its slope."""
    mapper = ArrayMapper(
        "aperture_mass", filter_type="schneider", filter_scale=4.0, truncation=1.0, ell=3,
    )
    source = jnp.zeros((9, 11)).at[4, 5].set(1.0)

    def objective(scale, cutoff, amplitude):
        g1 = amplitude * source
        maps = mapper.map_grid(g1, -0.4 * g1, filter_scale=scale, truncation=cutoff)
        return maps["E"][4, 3], maps["B"][4, 3]

    evaluate = jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2), has_aux=True))
    scale, cutoff, amplitude = 3.4, 0.67, 0.3
    values, gradient = evaluate(scale, cutoff, amplitude)

    # The source is two pixels along the x axis, where cos(2 phi) is one.
    q = (2 / scale)**2
    profile = 20 * q * (1 - q)**3 / (np.pi * scale**2)
    derivative = profile * (-4 + 6 * q / (1 - q)) / scale
    np.testing.assert_allclose(values, [-amplitude * profile, 0.4 * amplitude * profile], rtol=1e-12)
    np.testing.assert_allclose(gradient, [-amplitude * derivative, 0.0, -profile], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(evaluate(scale, 0.55, amplitude)[0], [0.0, 0.0])
    np.testing.assert_allclose(evaluate(scale, cutoff, 0.0)[1][2], -profile, rtol=1e-12)

def test_weighted_bilinear_catalog_position_derivative():
    """Moving one galaxy changes its fractional share of the binned shear."""
    mapper = ArrayMapper("kaiser_squires", smoothing=0)
    origin, pixel_size = (-3.0, 4.0), 0.5
    x = origin[0] + pixel_size * jnp.array([1.75, 2.10])
    y = origin[1] + pixel_size * jnp.array([1.90, 2.15])
    g1, g2 = jnp.array([0.2, -0.4]), jnp.array([0.1, 0.3])
    weights = jnp.array([2.0, 3.0])

    def objective(position):
        result = mapper.map_catalog(
            x.at[0].set(position), y, g1, g2, weights, shape=(5, 7),
            origin=origin, pixel_size=pixel_size, gridding="bilinear",
        )
        # Summing along y selects ky = 0, where KS is the identity except
        # for the mean. The column difference cancels that mean.
        column_difference = jnp.sum(result.maps["E"][:, 1] - result.maps["E"][:, 0])
        return column_difference, result

    (_, result), derivative = jax.jit(jax.value_and_grad(objective, has_aux=True))(x[0])

    # Rows and columns 1 and 2 have centers at 1.5 and 2.5 pixels.
    first = 2.0 * np.outer([0.60, 0.40], [0.75, 0.25])
    second = 3.0 * np.outer([0.35, 0.65], [0.40, 0.60])
    total = first + second
    expected_g1 = (0.2 * first - 0.4 * second) / total
    expected_g2 = (0.1 * first + 0.3 * second) / total
    for actual, expected in zip((result.grid.g1, result.grid.g2, result.grid.weights),
                                (expected_g1, expected_g2, total)):
        np.testing.assert_allclose(actual, np.pad(expected, ((1, 2), (1, 4))), rtol=1e-12, atol=1e-14)
    weight_derivative = -2.0 * np.array([0.60, 0.40]) / pixel_size
    expected_derivative = np.sum(weight_derivative * (0.2 - expected_g1[:, 0]) / total[:, 0])
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-12)

def test_catalog_snr_uses_explicit_key():
    """The key fixes the null maps, and SNR is invariant to shear units."""
    mapper = ArrayMapper("kaiser_squires", smoothing=0)
    x, y = jnp.array([0.5, 2.5, 4.5, 1.5]), jnp.array([0.5, 1.5, 2.5, 2.5])
    g1, g2 = jnp.array([0.03, -0.07, 0.10, 0.02]), jnp.array([0.08, 0.06, -0.04, 0.01])

    @jax.jit
    def map_catalog(key, amplitude=1.0):
        return mapper.map_catalog(x, y, amplitude * g1, amplitude * g2,
                                  shape=(3, 5), key=key, num_shuffles=8)

    key = jax.random.key(5)
    first, repeated, changed = map_catalog(key), map_catalog(key), map_catalog(jax.random.key(9))
    for mode in "EB":
        np.testing.assert_array_equal(first.maps[mode], changed.maps[mode])
        np.testing.assert_array_equal(first.snr[mode], repeated.snr[mode])
        assert np.isfinite(first.snr[mode][1, 2])
        assert np.isfinite(changed.snr[mode][1, 2])
        assert not np.isclose(first.snr[mode][1, 2], changed.snr[mode][1, 2])

    def snr(amplitude):
        return map_catalog(key, amplitude).snr["E"][1, 2]

    # Scaling every shear by the same positive factor scales the signal
    # and the null-map standard deviation equally.
    np.testing.assert_allclose(jax.grad(snr)(1.0), 0.0, atol=1e-12)

@pytest.mark.parametrize("method", ["kaiser_squires", "ks_plus", "aperture_mass"])
@pytest.mark.parametrize("shuffle_type", ["orientation", "spatial"])
def test_mapping_backends_agree(method, shuffle_type):
    """Both backends reconstruct the same gaps and seeded null maps."""
    catalog = Table.read(CATALOG, hdu=1)
    ra, dec = np.asarray(catalog["ra"]), np.asarray(catalog["dec"])
    ra0, dec0 = np.median(ra), np.median(dec)
    gap = np.hypot((ra - ra0) * np.cos(np.deg2rad(dec0)), dec - dec0) < 0.03
    catalog["weight"][gap] = 0
    options = dict(CATALOG_OPTIONS, create_snr=True,
                   snr=dict(shuffle_type=shuffle_type, num_shuffles=5, seed=13))
    if method == "ks_plus":
        options["inpainting_iterations"] = 8

    reference = smpy.map_mass(catalog, method, **options)
    with enable_x64(False):
        result = smpy.map_mass(catalog, method, backend="jax", **options)
        assert not jax.config.jax_enable_x64

    assert result.settings["general"]["backend"] == "jax"
    for actual, expected, atol in ((result.maps, reference.maps, 1e-12), (result.snr, reference.snr, 1e-9)):
        for mode in "EB":
            assert actual[mode].dtype == np.float64 and actual[mode].flags.writeable
            np.testing.assert_allclose(actual[mode], expected[mode], rtol=0, atol=atol, equal_nan=True)

def test_jax_projection_matches_numpy_grid_and_maps():
    """The TAN projection and shear rotation agree on the same sky grid."""
    catalog = Table.read(CATALOG, hdu=1)
    reference = smpy.map_mass(catalog, **CATALOG_OPTIONS)
    edges = reference.scaled_boundaries
    sky = reference.true_boundaries
    mapper = ArrayMapper()
    result = mapper.map_catalog(
        *(np.asarray(catalog[name]) for name in ("ra", "dec", "g1_Rinv", "g2_Rinv", "weight")),
        shape=reference.maps["E"].shape, origin=(edges["coord1_min"], edges["coord2_min"]),
        pixel_size=edges["pixel_size"], tangent_point=(sky["coord1_center"], sky["coord2_center"]),
    )

    np.testing.assert_array_equal(result.grid.counts, reference.grid.counts)
    for name in ("weights", "g1", "g2"):
        np.testing.assert_allclose(
            getattr(result.grid, name), getattr(reference.grid, name), rtol=0, atol=1e-12,
        )
    for mode in "EB":
        np.testing.assert_allclose(result.maps[mode], reference.maps[mode], rtol=0, atol=1e-12)

@pytest.mark.parametrize("gridding", ["hard", "bilinear"])
def test_map_catalog_drops_unplaceable_galaxies(gridding):
    """Unplaceable galaxies contribute nothing to a fixed grid under JIT."""
    mapper = ArrayMapper(smoothing=0)

    @jax.jit
    def map_sky(ra, dec, g1, g2, weights):
        return mapper.map_catalog(ra, dec, g1, g2, weights, shape=(4, 5), origin=(-2.5, -2.0),
                                  tangent_point=(0.0, 0.0), gridding=gridding)

    columns = jnp.array([[0.2, -0.4], [-0.3, 0.5], [0.03, -0.07], [0.08, 0.06], [2.0, 3.0]])
    # Outside x, outside y, non-finite positions and the antipode.
    extra = jnp.array([[6.0, 0.0, jnp.nan, 0.0, 180.0], [0.0, -6.0, 0.0, jnp.inf, 0.0],
                       [0.9] * 5, [0.7] * 5, [6.0] * 5])
    expected = map_sky(*columns)
    result = map_sky(*jnp.concatenate((columns, extra), axis=1))
    for name in ("counts", "weights", "g1", "g2"):
        np.testing.assert_array_equal(getattr(result.grid, name), getattr(expected.grid, name), err_msg=name)
    for mode in "EB":
        np.testing.assert_array_equal(result.maps[mode], expected.maps[mode])

    @jax.jit
    def map_pixel(x, y):
        return mapper.map_catalog(x, y, jnp.array([0.1, 0.2]), jnp.zeros(2), shape=(4, 5), gridding=gridding)

    edge = map_pixel(jnp.array([5.0, 5.01]), jnp.array([4.0, 4.0]))
    expected_counts = np.zeros((4, 5))
    expected_counts[-1, -1] = 1
    np.testing.assert_array_equal(edge.grid.counts, expected_counts)

def test_spatial_shuffle_ignores_dropped_galaxies():
    """Galaxies with one shear trade places without changing a map, so their
    spatial-shuffle noise is zero and the SNR undefined, whatever the dropped
    galaxies carry."""
    mapper = ArrayMapper(smoothing=0)
    # Six galaxies inside a 4 x 5 grid, then four that cannot be placed
    x = jnp.array([1.5, 2.5, 3.5, 1.5, 2.5, 3.5, -3.0, 9.0, 2.5, jnp.nan])
    y = jnp.array([1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 1.5, 2.5, 7.0, 1.5])
    placed = jnp.arange(10) < 6
    result = mapper.map_catalog(x, y, jnp.where(placed, 0.25, 0.75), jnp.where(placed, -0.5, 0.125),
                                shape=(4, 5), key=jax.random.key(3), num_shuffles=4, shuffle_type="spatial")
    assert result.grid.counts.sum() == 6
    for mode in "EB":
        assert np.abs(result.maps[mode]).max() > 0.1
        assert np.isnan(result.snr[mode]).all()

def test_dropped_galaxy_leaves_pixel_size_gradient_unchanged():
    """A galaxy that cannot be placed changes neither the binned shear nor its
    derivative with respect to the pixel size."""
    mapper = ArrayMapper(smoothing=0)

    def shared_pixel_shear(pixel_size, x):
        shear = 0.1 * (1 + jnp.arange(x.size))
        result = mapper.map_catalog(x, jnp.full_like(x, 1.4), shear, jnp.zeros_like(x), shape=(4, 5),
                                    gridding="bilinear", pixel_size=pixel_size)
        return result.grid.g1[1, 1]

    evaluate = jax.jit(jax.value_and_grad(shared_pixel_shear))
    expected = evaluate(1.0, jnp.array([1.1, 1.8]))
    assert abs(expected[1]) > 0.1
    for position in (jnp.nan, jnp.inf, 99.0):
        np.testing.assert_allclose(evaluate(1.0, jnp.array([1.1, 1.8, position])), expected, rtol=1e-12)

def test_ks_plus_ignores_rounding_in_gaps_without_power():
    """After one inpainting iteration the map is a single DCT mode, equal in
    two mirrored gap pixels, so the wavelet constraint has no gap power to
    rescale and changes nothing."""
    shear = np.array([2, -2, -3, 5, -4, -2, 2, 3, 2, 4, -5, -1, 1, -1, -1, -1, -5, -4,
                      0, 0, 5, -3, 4, -3, -4, -3, -1, -3, 4, 3]).reshape(5, 6) / 100
    weights = np.ones((5, 6))
    weights[0, [1, 4]] = 0
    settings = dict(smoothing=0, inpainting_iterations=1, reduced_shear_iterations=1, extension_size=0)
    constrained, free = (
        ArrayMapper("ks_plus", use_wavelet_constraints=use, **settings).map_grid(
            shear * weights, np.zeros((5, 6)), weights)
        for use in (True, False))
    for mode in "EB":
        assert np.abs(free[mode]).max() > 0.01
        np.testing.assert_allclose(constrained[mode], free[mode], rtol=0, atol=1e-12)

def test_aperture_mass_is_exactly_zero_on_nodal_lines():
    """A shear with no g2 has no cross component about a pixel on its own row
    or column and no tangential component about a pixel on its diagonals, so
    B and E are exactly zero there."""
    mapper = ArrayMapper("aperture_mass", filter_type="schneider", filter_scale=3.2, truncation=1.0)
    g1 = jnp.zeros((7, 9)).at[3, 4].set(0.125)
    maps = mapper.map_grid(g1, jnp.zeros_like(g1))
    steps = np.arange(-3, 4)
    np.testing.assert_array_equal(maps["B"][3, :], 0)
    np.testing.assert_array_equal(maps["B"][:, 4], 0)
    np.testing.assert_array_equal(maps["E"][3 + steps, 4 + steps], 0)
    np.testing.assert_array_equal(maps["E"][3 + steps, 4 - steps], 0)
    assert abs(maps["E"][3, 6]) > 1e-3 and abs(maps["B"][4, 6]) > 1e-3

def test_aperture_mass_snr_is_undefined_in_the_same_pixels():
    """A pixel that only the far tail of a Schirmer filter connects to the
    one sheared galaxy has no signal and no noise with the NumPy backend,
    whose convolution skips weights of rounding size, so its SNR is
    undefined. The JAX backend leaves the same pixels undefined."""
    corners = Table(dict(x=[0.0, 24.0, 0.0, 24.0], y=[0.0, 0.0, 24.0, 24.0],
                         g1=[0.125, 0.0, 0.0, 0.0], g2=np.zeros(4)))
    results = [
        smpy.map_mass(corners, "aperture_mass", backend=backend, coord_system="pixel", downsample_factor=1,
                      coord1_col="x", coord2_col="y", g1_col="g1", g2_col="g2", filter_type="schirmer",
                      filter_scale=10, truncation=2.0, create_snr=True, snr=dict(num_shuffles=8, seed=13))
        for backend in ("numpy", "jax")]
    for mode in "EB":
        undefined = [np.isnan(result.snr[mode]) for result in results]
        assert 0 < undefined[0].sum() < undefined[0].size
        np.testing.assert_array_equal(undefined[1], undefined[0])

def test_ks_plus_first_threshold_keeps_the_largest_mode():
    """The first inpainting threshold is the largest DCT coefficient of the
    first reconstruction, which survives it, so one iteration fills the two
    empty pixels with that mode on both backends."""
    shear = np.array([0, -2, 1, 4, 5, -3, 1, -3, 1, 4, 5, -4, -2, -2, -1, -4, 3, 0,
                      5, 5, 1, 0, 2, -3, 4, 5, 3, -1, 4, -3]).reshape(5, 6) / 100
    y, x = np.indices((5, 6))
    observed = (y > 0) | ((x != 1) & (x != 4))
    catalog = Table(dict(x=x[observed].astype(float), y=y[observed].astype(float),
                         g1=shear[observed], g2=np.zeros(28)))
    numpy_maps, jax_maps = (
        smpy.map_mass(catalog, "ks_plus", backend=backend, coord_system="pixel", downsample_factor=1,
                      coord1_col="x", coord2_col="y", g1_col="g1", g2_col="g2", smoothing=0,
                      inpainting_iterations=1, reduced_shear_iterations=1, extension_size=0,
                      use_wavelet_constraints=False).maps
        for backend in ("numpy", "jax"))
    assert numpy_maps["E"].shape == (5, 6) and np.abs(numpy_maps["E"]).max() > 0.01
    for mode in "EB":
        np.testing.assert_allclose(jax_maps[mode], numpy_maps[mode], rtol=0, atol=1e-12)
