"""Make the figures and numbers of the validation page of the documentation
and the residual figure of the paper.

Run from anywhere with ``python validation/accuracy.py``. The figures of the
page are written to ``docs/source/_static/validation/``, the paper's to
``paper/residuals.png``, and the numbers quoted on the page are printed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

import smpy

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "examples" / "data" / "forecast_lum_truth.fits"
CLUSTER_REDSHIFT = 0.45
CATALOG = ROOT / "examples" / "data" / "forecast_lum_annular.fits"
OUTPUT = ROOT / "docs" / "source" / "_static" / "validation"
NAMES = {"kaiser_squires": "Kaiser-Squires", "ks_plus": "KS+", "aperture_mass": "Aperture mass"}
TRUTH_SETTINGS = dict(g1_col="nfw_g1", g2_col="nfw_g2", pixel_scale=0.4)

def truth_catalog():
    """Return the galaxies of the truth catalog behind the cluster, as a
    shear catalog would hold them, with their convergence and a column of
    zeros.

    The catalog holds reduced shear g and magnification mu, so the
    convergence follows from mu = 1 / ((1 - kappa)^2 (1 - |g|^2)).
    """
    table = Table.read(TRUTH, hdu=1)
    table = table[table["redshift"] > CLUSTER_REDSHIFT]
    g_squared = table["nfw_g1"] ** 2 + table["nfw_g2"] ** 2
    table["kappa"] = 1 - 1 / np.sqrt(table["nfw_mu"] * (1 - g_squared))
    table["zero"] = 0.0
    return table

def binned_kappa(table, **settings):
    """Return the map grid with the true convergence, averaged over each
    pixel, in place of g1."""
    return smpy.map_mass(table, g1_col="kappa", g2_col="zero", pixel_scale=0.4, **settings).grid

def show(ax, image, title, **style):
    """Draw a map with north up and east to the left."""
    im = ax.imshow(image, origin="lower", **style)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im

def truth_recovery(table):
    """Compare each method's E map of the noiseless truth catalog with the
    true convergence, smoothed like the maps, and return that convergence
    and the error of each convergence map.

    Every map is shown and compared after removing its mean, which shear does
    not determine.
    """
    kappa = gaussian_filter(binned_kappa(table).g1, 2.0)
    kappa -= kappa.mean()
    true_peak = np.unravel_index(np.argmax(kappa), kappa.shape)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.2), layout="constrained")
    style = dict(cmap="magma", vmin=kappa.min(), vmax=kappa.max())
    show(axes[0], kappa, "True convergence", **style)
    errors = {}
    for ax, method in zip(axes[1:], NAMES):
        result = smpy.map_mass(table, method, **TRUTH_SETTINGS)
        map_e, map_b = result.maps["E"] - result.maps["E"].mean(), result.maps["B"]
        peak = np.unravel_index(np.argmax(map_e), map_e.shape)
        line = (f"{NAMES[method]}: peak offset {np.subtract(peak, true_peak)} pixels, "
                f"B/E spread {np.std(map_b) / np.std(map_e):.3f}")
        if method == "aperture_mass":
            fig.colorbar(show(ax, map_e, NAMES[method], cmap="magma"), ax=ax, label="Aperture mass")
        else:
            im = show(ax, map_e, NAMES[method], **style)
            errors[method] = map_e - kappa
            line += (f", correlation with truth {np.corrcoef(map_e.ravel(), kappa.ravel())[0, 1]:.4f}, "
                     f"RMS error {np.sqrt(np.mean(errors[method] ** 2)):.4f} "
                     f"(true peak {kappa.max():.3f})")
        print(line)
    fig.colorbar(im, ax=axes[:3], label="Convergence")
    fig.savefig(OUTPUT / "truth_recovery.png", dpi=120)
    plt.close(fig)
    return kappa, errors

def holes(table):
    """Compare Kaiser-Squires and KS+ inside holes cut in the truth catalog,
    and return the true convergence, the holes and the error of each map.

    Galaxies in the holes get zero weight, so the map grid is unchanged.
    Each map is compared after removing its mean over the observed pixels,
    which shear does not determine.
    """
    rng = np.random.default_rng(3)
    table["weight"] = 1.0
    for ra, dec in zip(rng.uniform(table["ra"].min(), table["ra"].max(), 8),
                       rng.uniform(table["dec"].min(), table["dec"].max(), 8)):
        offset = np.hypot((table["ra"] - ra) * np.cos(np.deg2rad(dec)), table["dec"] - dec)
        table["weight"][offset < 0.02] = 0.0
    kappa = binned_kappa(table).g1
    weights = binned_kappa(table, weight_col="weight").weights
    observed, gaps = weights > 0, weights == 0
    kappa -= kappa[observed].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), layout="constrained")
    show(axes[0], kappa, "True convergence", cmap="magma")
    errors = {}
    for ax, method in zip(axes[1:], ("kaiser_squires", "ks_plus")):
        result = smpy.map_mass(table, method, weight_col="weight", smoothing=0, **TRUTH_SETTINGS)
        errors[method] = result.maps["E"] - result.maps["E"][observed].mean() - kappa
        im = show(ax, errors[method], f"{NAMES[method]} error", cmap="RdBu_r", vmin=-0.04, vmax=0.04)
        print(f"{NAMES[method]}: RMS error in the holes {np.sqrt(np.mean(errors[method][gaps] ** 2)):.4f}")
    for ax in axes:
        ax.contour(gaps, levels=[0.5], colors="limegreen", linewidths=1)
    fig.colorbar(im, ax=axes[1:], label="Map minus true convergence")
    fig.savefig(OUTPUT / "holes.png", dpi=120)
    plt.close(fig)
    return kappa, gaps, errors

def residuals(recovery, filled):
    """Draw the paper's figure from the results of `truth_recovery` and
    `holes`: the true convergence and the error of the Kaiser-Squires and KS+
    maps, smoothed in the top row and unsmoothed with holes in the bottom row.

    Each row's color scale spans the largest error where that row is assessed:
    the whole map in the top row and the holes in the bottom row.
    """
    kappa, errors = recovery
    kappa_holes, gaps, errors_holes = filled
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 3.6), layout="constrained")
    rows = ((axes[0], kappa, errors, np.ones_like(gaps)), (axes[1], kappa_holes, errors_holes, gaps))
    for row, truth, row_errors, assessed in rows:
        fig.colorbar(show(row[0], truth, "True convergence", cmap="magma"), ax=row[0])
        limit = max(np.abs(error[assessed]).max() for error in row_errors.values())
        for ax, (method, error) in zip(row[1:], row_errors.items()):
            im = show(ax, error, f"{NAMES[method]} error", cmap="RdBu_r", vmin=-limit, vmax=limit)
        fig.colorbar(im, ax=row[1:], label="Map minus truth")
    for ax in axes[1]:
        ax.contour(gaps, levels=[0.5], colors="limegreen", linewidths=1)
    fig.savefig(ROOT / "paper" / "residuals.png", dpi=200)
    plt.close(fig)

def snr_calibration(catalogs=10):
    """Show that the SNR of catalogs without lensing signal has unit spread.

    Turning every shear of the example catalog by a random angle removes
    its signal and keeps its noise. The spread of one map varies by about 10%
    between such catalogs, as neighboring pixels are correlated, so the
    SNR of several catalogs is pooled.
    """
    table = Table.read(CATALOG, hdu=1)
    signal = table["g1_Rinv"] + 1j * table["g2_Rinv"]
    settings = dict(g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight", pixel_scale=0.4, create_snr=True)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    bins = np.linspace(-5, 5, 41)
    for method in NAMES:
        snr = []
        for seed in range(catalogs):
            shear = signal * np.exp(2j * np.random.default_rng(seed).uniform(0, np.pi, len(table)))
            table["g1_Rinv"], table["g2_Rinv"] = shear.real, shear.imag
            snr.append(smpy.map_mass(table, method, snr={"seed": seed}, **settings).snr["E"])
        spreads = np.std(snr, axis=(1, 2))
        ax.hist(np.ravel(snr), bins, density=True, histtype="step", linewidth=1.5,
                label=f"{NAMES[method]} (spread {np.std(snr):.2f})")
        print(f"{NAMES[method]}: pooled SNR spread {np.std(snr):.3f}, mean {np.mean(snr):+.3f}; "
              f"spread of single maps {spreads.mean():.3f} +- {spreads.std():.3f}")
    ax.plot(bins, norm.pdf(bins), color="black", linestyle="--", label="Unit normal")
    ax.set_xlabel("E-mode SNR of a catalog without lensing signal")
    ax.set_ylabel("Fraction of pixels per unit SNR")
    ax.legend(loc="upper left")
    fig.savefig(OUTPUT / "snr_calibration.png", dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    residuals(truth_recovery(truth_catalog()), holes(truth_catalog()))
    snr_calibration()
