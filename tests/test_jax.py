"""JAX transform tests for the public array mapping API."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import smpy
from smpy.coordinates.pixel import PixelSystem
from smpy.filters.processing import schirmer_filter, schneider_filter
from smpy.filters.starlet import starlet_transform_2d
from smpy.mapping_methods.kaiser_squires import KaiserSquiresMapper
from smpy.snr import compute_snr

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

@pytest.fixture(autouse=True)
def _double_precision():
    """Use double precision for analytic and finite-difference checks."""
    with enable_x64(True):
        yield

def test_kaiser_squires_smoothing_derivative():
    """A traced Gaussian sigma changes the gain of an axial Fourier mode."""
    mapper = smpy.ArrayMapper("kaiser_squires", smoothing=2.0)
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
    mapper = smpy.ArrayMapper(
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
    mapper = smpy.ArrayMapper(
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
    mapper = smpy.ArrayMapper("kaiser_squires", smoothing=0)
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
    mapper = smpy.ArrayMapper("kaiser_squires", smoothing=0)
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

def test_legacy_snr_respects_the_grid_estimator():
    """Rescaling the grid's shear scales the signal and null maps equally."""
    class DoubleGrid(PixelSystem):
        """A coordinate system with twice the binned shear."""

        def create_grid(self, data_df, boundaries):
            """Bin the catalog and double both shear components."""
            grid = super().create_grid(data_df, boundaries)
            return replace(grid, g1=2 * grid.g1, g2=2 * grid.g2)

    catalog = pd.DataFrame(dict(coord1=[0., 2.], coord2=[0., 0.],
                                g1=[0.2, 0.], g2=[0., 0.], weight=[1., 1.]))
    mapper = KaiserSquiresMapper(smoothing=0)
    settings = dict(num_shuffles=3, seed=0, shuffle_type="orientation")
    estimates = []
    for system in (PixelSystem(), DoubleGrid()):
        transformed = system.transform_coordinates(catalog)
        boundaries, _ = system.calculate_boundaries(transformed, {"downsample_factor": 1})
        grid = system.create_grid(transformed, boundaries)
        maps = dict(zip("EB", mapper.create_maps(grid)))
        estimates.append(compute_snr(transformed, system, boundaries, mapper, maps, settings)["E"])
    np.testing.assert_allclose(*estimates, rtol=1e-12)
    for array in (grid.g1, *estimates):
        assert isinstance(array, np.ndarray)
        assert array.flags.writeable

def test_legacy_snr_prepares_each_null_catalog():
    """An additive gridding calibration acts on the shuffled shear."""
    class CalibratedGrid(PixelSystem):
        """Subtract an additive calibration before binning."""

        def prepare_data(self, data_df):
            """Apply the calibration in the map frame."""
            prepared = super().prepare_data(data_df)
            return prepared.assign(g1=prepared["g1"] - 0.1)

    catalog = pd.DataFrame(dict(coord1=[0., 2.], coord2=[0., 0.],
                                g1=[0.2, 0.], g2=[0., 0.], weight=[1., 1.]))
    system = CalibratedGrid()
    data = system.transform_coordinates(catalog)
    boundaries, _ = system.calculate_boundaries(data, {"downsample_factor": 1})
    mapper = KaiserSquiresMapper(smoothing=0)
    signal = dict(zip("EB", mapper.create_maps(system.create_grid(data, boundaries))))
    settings = dict(num_shuffles=3, seed=0, shuffle_type="orientation")
    measured = compute_snr(data, system, boundaries, mapper, signal, settings)

    # KS on the three-pixel row subtracts the mean. Only the first galaxy
    # has nonzero shear before calibration, so only it adds shape noise.
    phases = np.random.default_rng(0).uniform(0, 2 * np.pi, (3, 2))
    std_first = (2 / 3 * 0.2 * np.cos(phases[:, 0])).std(ddof=1)
    np.testing.assert_allclose(measured["E"][0, 0], 0.1 / std_first, rtol=1e-12)

@pytest.mark.parametrize("filter_function", [schirmer_filter, schneider_filter])
def test_legacy_numpy_filters_keep_double_precision(filter_function):
    """NumPy input produces writable double precision without a JAX flag."""
    with enable_x64(False):
        profile = filter_function(np.array([0.1], dtype=np.float64), 1.0)
    assert isinstance(profile, np.ndarray)
    assert profile.dtype == np.float64
    assert profile.flags.writeable

def test_legacy_starlet_preserves_the_midpoint_of_float32_pixels():
    """The reflected filter can resolve a midpoint between float32 values."""
    image = np.array([[1, 1 + 2**-23]], dtype=np.float32)
    bands = starlet_transform_2d(image, 2)
    np.testing.assert_array_equal(bands[-1], [[1 + 2**-24, 1 + 2**-24]])

def test_array_mapper_accepts_numpy_method_scalars():
    """A NumPy scalar setting has the same meaning as a Python scalar."""
    shear = jnp.arange(21, dtype=jnp.float64).reshape(3, 7)
    expected = smpy.ArrayMapper("kaiser_squires", smoothing=1.0).map_grid(shear, -shear)
    actual = smpy.ArrayMapper("kaiser_squires", smoothing=np.float32(1.0)).map_grid(shear, -shear)
    for mode in "EB":
        np.testing.assert_array_equal(actual[mode], expected[mode])
