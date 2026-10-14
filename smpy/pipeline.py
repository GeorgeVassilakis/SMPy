"""Mass mapping pipeline: compute maps from a catalog, then write outputs.

`compute` makes the maps and writes nothing. `run` computes the maps a
configuration asks for and writes its plots, FITS files and settings record.
"""

import logging
import secrets
import time
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from smpy import __version__
from smpy.config import Config
from smpy.coordinates import get_coordinate_system
from smpy.mapping_methods import METHODS
from smpy.result import MassMap
from smpy.snr import compute_snr
from smpy.utils import load_shear_data

logger = logging.getLogger(__name__)

def compute(catalog, settings):
    """Compute mass maps from a catalog without writing anything.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Shear catalog, or the path to a FITS file.
    settings : `dict`
        Complete, validated settings, such as ``Config.resolved()``.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used, with an SNR ``seed`` of
        'random' replaced by the seed drawn.
    """
    general = settings["general"]
    method = general["method"]
    grid_settings = general[general["coord_system"]]
    coord_system = get_coordinate_system(general["coord_system"])

    shear_df = load_shear_data(catalog, grid_settings["coord1_col"], grid_settings["coord2_col"],
                               general["g1_col"], general["g2_col"], general["weight_col"],
                               general["input_hdu"], "deg" if general["coord_system"] == "radec" else None)
    shear_df = coord_system.transform_coordinates(shear_df)
    scaled_boundaries, true_boundaries = coord_system.calculate_boundaries(shear_df, grid_settings)
    grid = coord_system.create_grid(shear_df, scaled_boundaries)

    start = time.perf_counter()
    mapper = METHODS[method](**settings["methods"][method])
    map_e, map_b = mapper.create_maps(grid)
    maps = {"E": map_e, "B": map_b}
    logger.info("Created %s maps in %.2f s", method, time.perf_counter() - start)

    snr = None
    if general["create_snr"]:
        if settings["snr"]["seed"] == "random":
            settings = {**settings, "snr": {**settings["snr"], "seed": secrets.randbits(128)}}
        snr = compute_snr(shear_df, coord_system, scaled_boundaries, mapper, maps, settings["snr"])
    return MassMap(maps, snr, grid, scaled_boundaries, true_boundaries, settings)

def run(config):
    """Compute mass maps as configured and write the requested outputs.

    Parameters
    ----------
    config : `str`, `pathlib.Path`, or `smpy.Config`
        YAML configuration file, or a `Config`.

    Returns
    -------
    result : `smpy.MassMap`
        The maps, their grid and the settings used.

    Raises
    ------
    ValueError
        Raised if a setting is invalid or ``general.input_path`` is empty.

    Notes
    -----
    Outputs go to ``<output_directory>/<method>/``: PNG plots if
    ``save_plots``, FITS maps if ``save_fits``, and a YAML record of every
    setting used. The ``smpy`` command-line tool calls this function.
    """
    if not isinstance(config, Config):
        config = Config.from_file(config)
    config.validate()
    settings = config.resolved()
    general = settings["general"]
    if not general["input_path"]:
        raise ValueError("general.input_path is required")

    result = compute(general["input_path"], settings)

    method = general["method"]
    output_dir = Path(general["output_directory"]) / method
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{general['output_base_name']}_{method}"
    kinds = (False, True) if result.snr is not None else (False,)
    for mode in general["mode"]:
        for snr in kinds:
            name = f"{prefix}{'_snr' if snr else ''}_{mode.lower()}_mode"
            if general["save_plots"]:
                _save_figure(result.plot(mode, snr=snr)[0], f"{name}.png")
            if general["save_fits"]:
                result.to_fits(f"{name}.fits", mode, snr=snr)
    if general["create_counts_map"] and general["save_plots"]:
        _save_figure(result.plot_counts()[0], f"{prefix}_counts.png")

    with open(f"{prefix}_settings.yaml", "w", encoding="utf-8") as stream:
        stream.write(f"# Settings used by SMPy {__version__}\n")
        yaml.safe_dump(result.settings, stream, sort_keys=False)
    return result

def _save_figure(fig, path):
    """Save a figure to ``path`` and close it."""
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved plot: %s", path)
