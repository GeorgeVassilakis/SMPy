"""Time SMPy's methods for the performance page of the documentation.

Run from anywhere with ``python validation/performance.py``. It prints the
timings quoted on the page and writes
``docs/source/_static/performance/scaling.png``.
"""

import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from astropy.table import Table

import smpy

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "source" / "_static" / "performance"
NAMES = {"kaiser_squires": "Kaiser-Squires", "aperture_mass": "Aperture mass", "ks_plus": "KS+"}
SIDES = (64, 128, 256, 512, 1024)
GALAXIES_PER_PIXEL = 10
SETTINGS = dict(coord_system="pixel", downsample_factor=1, coord1_col="x", coord2_col="y")

def catalog(side):
    """Return a catalogue of pure shape noise filling a side x side map."""
    rng = np.random.default_rng(0)
    size = side * side * GALAXIES_PER_PIXEL
    return Table({"x": rng.uniform(0, side, size), "y": rng.uniform(0, side, size),
                  "g1": rng.normal(0, 0.3, size), "g2": rng.normal(0, 0.3, size)})

def seconds(table, method, repeats, **settings):
    """Return the shortest of ``repeats`` run times of `smpy.map_mass`."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        smpy.map_mass(table, method, **SETTINGS, **settings)
        times.append(time.perf_counter() - start)
    return min(times)

if __name__ == "__main__":
    print(f"{platform.platform()}, Python {platform.python_version()}, "
          f"NumPy {np.__version__}, SciPy {scipy.__version__}")
    times = {method: [] for method in NAMES}
    for side in SIDES:
        table = catalog(side)
        for method in NAMES:
            times[method].append(seconds(table, method, 1 if method == "ks_plus" and side > 128 else 3))
        print(f"{side}x{side}, {len(table):,} galaxies: "
              + ", ".join(f"{NAMES[method]} {times[method][-1]:.3g} s" for method in NAMES))

    # An SNR map costs one map and one gridding of the catalogue per shuffle
    side, shuffles = 256, 20
    table = catalog(side)
    for method, repeats in (("kaiser_squires", 3), ("aperture_mass", 3), ("ks_plus", 1)):
        total = seconds(table, method, repeats, create_snr=True, snr={"num_shuffles": shuffles})
        single = times[method][SIDES.index(side)]
        print(f"SNR, {side}x{side}, {NAMES[method]}: {(total - single) / shuffles:.3g} s per shuffle, "
              f"{single:.3g} s per map")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for method in NAMES:
        ax.loglog(np.square(SIDES), times[method], marker="o", label=NAMES[method])
    ax.set_xlabel("Map pixels")
    ax.set_ylabel("Time per map (s)")
    ax.legend()
    fig.savefig(OUTPUT / "scaling.png", dpi=120)
