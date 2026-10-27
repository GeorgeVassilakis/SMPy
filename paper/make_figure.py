"""Make the paper's figure: maps of the example catalog by each method.

Run from anywhere with ``python paper/make_figure.py``; it writes
``paper/mass_maps.png`` at the width of a JOSS page.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import smpy

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "examples" / "data" / "forecast_lum_annular.fits"
SETTINGS = dict(pixel_scale=0.4, g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")
STYLE = dict(cmap="magma", fontsize=8)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 4.6), gridspec_kw={"wspace": 0.55, "hspace": 0.45})
    kaiser_squires = smpy.map_mass(CATALOG, create_snr=True, **SETTINGS)
    kaiser_squires.plot("E", ax=axes[0, 0], plot_title="Kaiser-Squires", **STYLE)
    smpy.map_mass(CATALOG, "ks_plus", **SETTINGS).plot("E", ax=axes[0, 1], plot_title="KS+", **STYLE)
    smpy.map_mass(CATALOG, "aperture_mass", **SETTINGS).plot("E", ax=axes[1, 0], plot_title="Aperture mass",
                                                             **STYLE)
    kaiser_squires.plot("E", snr=True, ax=axes[1, 1], plot_title="Kaiser-Squires SNR", **STYLE)
    fig.savefig(ROOT / "paper" / "mass_maps.png", dpi=200, bbox_inches="tight")
