"""Regression test comparing SMPy outputs with stored reference values.

The reference file holds maps, SNR maps, counts and FITS output for the
example catalogue. Regenerate it with ``python tests/test_reference.py`` only
when a commit changes results on purpose, and say why in that commit.
"""

import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from smpy import Config, run

CATALOG = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_annular.fits"
REFERENCE = Path(__file__).resolve().parent / "data" / "reference.npz"
METHODS = {
    "ks": ("kaiser_squires", {}),
    "schirmer": ("aperture_mass", {"filter_type": "schirmer"}),
    "schneider": ("aperture_mass", {"filter_type": "schneider"}),
    "ksplus": ("ks_plus", {}),
}
COORDINATES = {
    "radec": {"coord_system": "radec", "pixel_scale": 0.4},
    "pixel": {"coord_system": "pixel", "downsample_factor": 170, "coord1_col": "X_IMAGE_se",
              "coord2_col": "Y_IMAGE_se"},
}
FITS_KEYS = ("CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2")
RTOL = 1e-9
ATOL = 1e-12

def run_case(output_dir, method, coordinates, snr=None, **settings):
    """Run SMPy on the example catalogue and return its result."""
    config = Config.from_defaults(method, input_path=str(CATALOG), g1_col="g1_Rinv", g2_col="g2_Rinv",
                                  weight_col="weight", output_directory=str(output_dir), save_plots=False,
                                  mode=["E", "B"], snr=snr, **coordinates, **settings)
    return run(config)

def compute_outputs(output_dir):
    """Return every reference quantity, keyed by name."""
    outputs = {}
    for coords_name, coordinates in COORDINATES.items():
        for name, (method, settings) in METHODS.items():
            maps = run_case(output_dir, method, coordinates, **settings).maps
            for mode in ("E", "B"):
                outputs[f"{coords_name}_{name}_{mode}"] = maps[mode]

    for shuffle_type in ("spatial", "orientation"):
        snr = {"shuffle_type": shuffle_type, "num_shuffles": 20}
        snr_maps = run_case(output_dir, "kaiser_squires", COORDINATES["radec"], snr=snr,
                            create_snr=True).snr
        for mode in ("E", "B"):
            outputs[f"snr_{shuffle_type}_{mode}"] = snr_maps[mode]

    outputs["counts"] = run_case(output_dir, "kaiser_squires", COORDINATES["radec"]).grid.counts

    run_case(output_dir, "kaiser_squires", COORDINATES["radec"], save_fits=True)
    with fits.open(Path(output_dir) / "kaiser_squires" / "smpy_output_kaiser_squires_e_mode.fits") as hdul:
        outputs["fits_data"] = hdul[0].data
        outputs["fits_header"] = np.array([hdul[0].header[key] for key in FITS_KEYS])
    return outputs

def test_outputs_match_reference(tmp_path):
    """Every output matches the stored reference within tolerance."""
    reference = np.load(REFERENCE)
    outputs = compute_outputs(tmp_path)
    assert sorted(outputs) == sorted(reference.files)
    for key in reference.files:
        np.testing.assert_allclose(outputs[key], reference[key], rtol=RTOL, atol=ATOL, err_msg=key)

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as output_dir:
        np.savez_compressed(REFERENCE, **compute_outputs(output_dir))
    print(f"Wrote {REFERENCE}")
