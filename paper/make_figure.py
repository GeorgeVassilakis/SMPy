"""Make the paper's figure: the three methods' maps of the example catalogue.

Run from anywhere with ``python paper/make_figure.py``; it writes
``paper/mass_maps.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import smpy

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "examples" / "data" / "forecast_lum_annular.fits"
METHODS = {"kaiser_squires": "Kaiser-Squires", "ks_plus": "KS+", "aperture_mass": "Aperture mass"}

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(24, 5.5), gridspec_kw={"wspace": 0.45})
    for ax, (method, name) in zip(axes, METHODS.items()):
        result = smpy.map_mass(CATALOG, method, pixel_scale=0.4, g1_col="g1_Rinv", g2_col="g2_Rinv",
                               weight_col="weight")
        result.plot("E", ax=ax, cmap="magma", plot_title=name)
    fig.savefig(ROOT / "paper" / "mass_maps.png", dpi=150, bbox_inches="tight")
