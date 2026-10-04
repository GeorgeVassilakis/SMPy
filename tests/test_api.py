"""Behaviour tests for the Python API."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import smpy
from smpy import Config
from smpy.cli import main
from smpy.mapping_methods import METHODS

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_annular.fits"
COLUMNS = dict(g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")
RADEC = dict(pixel_scale=0.4, **COLUMNS)
PIXEL = dict(coord_system="pixel", downsample_factor=170, coord1_col="X_IMAGE_se", coord2_col="Y_IMAGE_se",
             **COLUMNS)

@pytest.fixture
def table():
    """The example catalog as an astropy Table."""
    return Table.read(CATALOG, hdu=1)

def test_file_and_table_inputs_agree(table):
    """A file path and the same catalog as a Table give identical maps."""
    from_file = smpy.map_mass(CATALOG, **RADEC)
    from_table = smpy.map_mass(table, **RADEC)
    for mode in "EB":
        np.testing.assert_array_equal(from_file.maps[mode], from_table.maps[mode])

def test_grid_reproduces_maps():
    """The grid on a result is the one its maps were made from."""
    result = smpy.map_mass(CATALOG, **RADEC)
    mapper = METHODS["kaiser_squires"](**result.settings["methods"]["kaiser_squires"])
    for mode, grid_map in zip("EB", mapper.create_maps(result.grid)):
        np.testing.assert_array_equal(grid_map, result.maps[mode])

@pytest.mark.parametrize("shuffle_type", ["spatial", "orientation"])
def test_map_mass_has_no_side_effects(table, tmp_path, monkeypatch, shuffle_type):
    """map_mass writes no files and leaves its input and global RNGs alone."""
    monkeypatch.chdir(tmp_path)
    before = table.copy()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    smpy.map_mass(table, create_snr=True, snr={"shuffle_type": shuffle_type, "num_shuffles": 2}, **RADEC)

    assert list(tmp_path.iterdir()) == []
    for name in table.colnames:
        np.testing.assert_array_equal(table[name], before[name])
    after = np.random.get_state()
    assert random.getstate() == python_state
    assert np.array_equal(after[1], numpy_state[1]) and after[2:] == numpy_state[2:]

def test_smpy_command_matches_map_mass(tmp_path):
    """The smpy command and map_mass give the same maps from settings of the
    same names, and the settings record the command writes repeats its run."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        f"general:\n  input_path: {CATALOG}\n  output_directory: out\n  mode: [E, B]\n"
        "  save_plots: false\n  save_fits: true\n  radec: {pixel_scale: 0.4}\n"
        "  g1_col: g1_Rinv\n  g2_col: g2_Rinv\n  weight_col: weight\n"
        "methods:\n  kaiser_squires: {smoothing: 1.5}\n")
    main(["-c", str(config_path)])
    from_python = smpy.map_mass(CATALOG, smoothing=1.5, **RADEC)

    output = tmp_path / "out" / "kaiser_squires"
    for mode in "EB":
        written = fits.getdata(output / f"smpy_output_kaiser_squires_{mode.lower()}_mode.fits")
        np.testing.assert_array_equal(written, from_python.maps[mode])
    record = Config.from_file(output / "smpy_output_kaiser_squires_settings.yaml").resolved()
    assert record == Config.from_file(config_path).resolved()
    assert record["methods"] == from_python.settings["methods"]

INVALID = {
    "other method's setting": ({"inpainting_iterations": 5, **RADEC},
                               "Unknown setting 'inpainting_iterations'"),
    "unknown method": ({"method": "kaiser_squirez", **RADEC}, "general.method must be one of"),
    "missing pixel_scale": (COLUMNS, "general.radec.pixel_scale is required"),
    "negative smoothing": ({"smoothing": -1.0, **RADEC}, "smoothing must be at least 0"),
    "unknown filter": ({"method": "aperture_mass", "filter_type": "gauss", **RADEC}, "filter_type must be"),
    "misspelled YAML key": ("plotting:\n  cmapp: magma\n", "Unknown setting 'plotting.cmapp'"),
    "invalid YAML value": ("general:\n  radec: {pixel_scale: 0.4}\n  mode: [E, C]\n",
                           "general.mode may contain only"),
}

@pytest.mark.parametrize("settings, name", INVALID.values(), ids=INVALID.keys())
def test_invalid_settings_name_the_setting(settings, name, tmp_path):
    """Invalid keywords and YAML settings raise errors that name them."""
    with pytest.raises(ValueError, match=name):
        if isinstance(settings, str):
            (tmp_path / "run.yaml").write_text(settings)
            smpy.run(tmp_path / "run.yaml")
        else:
            smpy.map_mass(CATALOG, **settings)

def test_ks_plus_clips_too_many_scales():
    """KS+ reduces an nscales too large for the map to the largest that fits,
    and warns."""
    # 8 is just above the limit for this map, so an unclipped run stays fast
    settings = dict(inpainting_iterations=2, reduced_shear_iterations=1, **RADEC)
    with pytest.warns(UserWarning, match="nscales=8 was clipped"):
        clipped = smpy.map_mass(CATALOG, "ks_plus", nscales=8, **settings)
    largest = smpy.map_mass(CATALOG, "ks_plus", **settings)
    for mode in "EB":
        np.testing.assert_array_equal(clipped.maps[mode], largest.maps[mode])

@pytest.mark.parametrize("settings", [RADEC, PIXEL], ids=["radec", "pixel"])
def test_plot_draws_on_given_axes(settings):
    """plot draws on the axes it is given and leaves the figure open."""
    result = smpy.map_mass(CATALOG, **settings)
    fig, ax = plt.subplots()
    returned_fig, returned_ax, image = result.plot("E", ax=ax)
    assert returned_fig is fig and returned_ax is ax and image in ax.images
    assert plt.fignum_exists(fig.number)
    plt.close(fig)

def test_pixel_maps_export_to_fits(tmp_path):
    """Pixel-coordinate maps can be written to FITS."""
    result = smpy.map_mass(CATALOG, **PIXEL)
    result.to_fits(tmp_path / "map.fits")
    np.testing.assert_array_equal(fits.getdata(tmp_path / "map.fits"), result.maps["E"])
