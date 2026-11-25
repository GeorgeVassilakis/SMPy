# Copyright (c) 2023-2026 Georgios N. Vassilakis
# SPDX-License-Identifier: MIT

"""Benchmark SMPy's NumPy and JAX backends on the same catalogs.

From the checkout root, run this file in a fresh process for each backend.
Put the checkout first on ``PYTHONPATH``. For example::

    PYTHONPATH=. python validation/jax_performance.py --backend numpy
    PYTHONPATH=. python validation/jax_performance.py --backend jax-cpu
    PYTHONPATH=. python validation/jax_performance.py --backend jax-gpu

Use the same CPU affinity for the NumPy and JAX CPU runs and make one GPU
visible for the JAX GPU run. Use ``--case smoke`` to check the harness, and
``--revision`` to record the source revision of the package being timed.

Notes
-----
Catalog preparation is reported separately. The ``map_mass`` measurement
starts with an in-memory table and ends with host E/B maps. The ``map_grid``
measurement starts with placed arrays from the same binned catalog and ends
with synchronized device E/B maps. Array placement and mapper construction
are timed separately. Optional SNR timing includes the signal map, all
shuffles and host results. Optional gradient timing includes the value and
gradient of the mean squared E/B maps with respect to both shear grids.

The first call in each JAX timing series follows an in-process cache clear,
with the persistent compilation cache disabled. It includes compilation and
execution after backend initialization. The full catalog runs use SMPy's
default KS+ iteration counts. The smoke case uses four inpainting iterations
and two reduced-shear iterations.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from functools import partial
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from astropy.table import Table

CLUSTER_FITS = Path(__file__).resolve().parents[1] / "examples" / "data" / "forecast_lum_annular.fits"
"""Example cluster catalog used in SMPy's documentation (`pathlib.Path`)."""

SYNTHETIC_CASES = {
    "euclid": (5, 16.2, 125, "https://doi.org/10.1051/0004-6361/202557573"),
    "survey": (10, 30, 126, "https://arxiv.org/abs/1910.03106"),
    "smoke": (1, 4, 127, None),
}
"""Field widths in degrees, densities, seeds and references (`dict`)."""

METHODS = ("kaiser_squires", "aperture_mass", "ks_plus")
"""Public SMPy mapping method names (`tuple` of `str`)."""

def _load_case(name, cluster_fits):
    """Return an in-memory table, mapping keywords and workload metadata."""
    if name == "cluster":
        columns = ["ra", "dec", "g1_Rinv", "g2_Rinv", "weight"]
        table = Table.read(cluster_fits, hdu=1)[columns]
        settings = dict(coord_system="radec", pixel_scale=0.4, coord1_col="ra", coord2_col="dec",
                        g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")
        return table, settings, {"fits_path": str(cluster_fits.resolve()), "fits_hdu": 1,
                                  "pixel_scale_arcmin": 0.4}

    side_deg, density, seed, reference = SYNTHETIC_CASES[name]
    side_arcmin = 60 * side_deg
    size = round(density * side_arcmin**2)
    rng = np.random.Generator(np.random.PCG64(seed))
    x, y = rng.random(size), rng.random(size)
    shape_noise = 0.26
    g1 = rng.normal(0, shape_noise, size)
    g2 = rng.normal(0, shape_noise, size)

    # A convergence Fourier mode gives shear with the same phase and a
    # spin-2 direction set by its wave vector.
    modes = ((0.020, 2, 1), (0.012, 1, -3), (0.008, 5, 4))
    for amplitude, kx, ky in modes:
        mode = amplitude * np.cos(2 * np.pi * (kx * x + ky * y))
        k_squared = kx**2 + ky**2
        g1 += (kx**2 - ky**2) / k_squared * mode
        g2 += 2 * kx * ky / k_squared * mode
        del mode

    weight = 1 + 0.5 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    holes = ((0.25, 0.24), (0.69, 0.35), (0.50, 0.72))
    hole_radius = 0.075
    for center_x, center_y in holes:
        weight[(x - center_x)**2 + (y - center_y)**2 < hole_radius**2] = 0.0

    # Use the pixel API with one input coordinate unit per arcminute.
    x *= side_arcmin
    y *= side_arcmin
    table = Table(dict(x=x, y=y, g1=g1, g2=g2, weight=weight), copy=False)
    settings = dict(coord_system="pixel", downsample_factor=2, coord1_col="x", coord2_col="y",
                    g1_col="g1", g2_col="g2", weight_col="weight")
    metadata = {
        "side_deg": side_deg,
        "pixel_scale_arcmin": 2,
        "input_galaxies_per_arcmin2": density,
        "seed": seed,
        "shape_noise_sigma_per_component": shape_noise,
        "hole_centers_field_fraction": holes,
        "hole_radius_field_fraction": hole_radius,
        "convergence_modes_amplitude_kx_ky": modes,
        "reference": reference,
    }
    return table, settings, metadata

def _method_settings(args, case, method):
    """Return method settings shared by both backends."""
    if method == "kaiser_squires":
        return {"smoothing": 2.0}
    if method == "aperture_mass":
        return dict(filter_type="schirmer", filter_scale=10, truncation=1.0, ell=3)
    return dict(
        smoothing=2.0,
        inpainting_iterations=args.inpainting_iterations or (4 if case == "smoke" else 100),
        reduced_shear_iterations=args.reduced_shear_iterations or (2 if case == "smoke" else 3),
        nscales=None,
        extension_size="double",
        use_wavelet_constraints=True,
    )

def _timed(call):
    """Return the result of a blocking call and its elapsed wall seconds."""
    start = perf_counter()
    result = call()
    return result, perf_counter() - start

def _time_series(call, repeats):
    """Time the first call and repeated calls with the same inputs."""
    result, first = _timed(call)
    steady = []
    for _ in range(repeats):
        result, seconds = _timed(call)
        steady.append(seconds)
    return result, {"first_seconds": first, "steady_seconds": steady, "steady_median_seconds": median(steady)}

def _map_mass_host(map_mass, table, method, **settings):
    """Map a table and finish any outstanding transfers of E/B and SNR."""
    result = map_mass(table, method, **settings)
    maps = {mode: np.asarray(result.maps[mode]) for mode in "EB"}
    if result.snr is not None:
        for mode in "EB":
            np.asarray(result.snr[mode])
    return result, maps

def _device_timings(jax, device, make_mapper, grid, host_maps, repeats, gradient):
    """Time synchronized ArrayMapper calls on a fixed device grid."""

    def place_grid():
        return jax.block_until_ready(jax.device_put((grid.g1, grid.g2, grid.weights), device))

    inputs, placement = _timed(place_grid)
    jax.clear_caches()
    mapper, construction = _timed(make_mapper)

    def forward():
        return jax.block_until_ready(mapper.map_grid(*inputs))

    maps, timing = _time_series(forward, repeats)
    for mode in "EB":
        if maps[mode].devices() != {device}:
            raise RuntimeError(f"ArrayMapper {mode} output is on {maps[mode].devices()}, expected {device}")

    timing["grid_placement_seconds"] = placement
    timing["mapper_construction_seconds"] = construction
    timing["input_dtypes"] = [str(array.dtype) for array in inputs]
    timing["map_dtypes"] = {mode: str(maps[mode].dtype) for mode in "EB"}
    timing["max_abs_difference_from_map_mass"] = {}
    for mode in "EB":
        difference = np.abs(np.asarray(maps[mode]) - host_maps[mode])
        error = float(difference.max()) if np.isfinite(difference).all() else None
        timing["max_abs_difference_from_map_mass"][mode] = error

    if gradient:
        def loss(g1, g2, weights):
            maps = mapper.map_grid(g1, g2, weights)
            return (maps["E"]**2 + maps["B"]**2).mean()

        jax.clear_caches()
        value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

        def differentiate():
            return jax.block_until_ready(value_and_grad(*inputs))

        (value, derivatives), gradient_timing = _time_series(differentiate, repeats)
        gradient_timing["loss_finite"] = bool(np.isfinite(np.asarray(value)))
        gradient_timing["gradients_finite"] = all(np.isfinite(np.asarray(array)).all()
                                                  for array in derivatives)
        timing["value_and_grad"] = gradient_timing
    return timing

def _parse_args():
    """Read the backend, workloads and timing options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=("numpy", "jax-cpu", "jax-gpu"), required=True)
    parser.add_argument("--case", nargs="+", choices=("cluster", *SYNTHETIC_CASES),
                        default=["cluster", "euclid", "survey"], help="Catalogs to benchmark")
    parser.add_argument("--method", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--cluster-fits", type=Path, default=CLUSTER_FITS)
    parser.add_argument("--repeats", type=int, default=3, help="Calls after the separately timed first call")
    parser.add_argument("--inpainting-iterations", type=int, help="KS+ override for all selected catalogs")
    parser.add_argument("--reduced-shear-iterations", type=int, help="KS+ override for all selected catalogs")
    parser.add_argument("--snr-shuffles", type=int, default=0, help="Run SNR once per case and method")
    parser.add_argument("--gradient", action="store_true",
                        help="Also time a device-resident value and gradient")
    parser.add_argument("--revision", help="Source revision of the SMPy checkout being benchmarked")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of stdout")
    args = parser.parse_args()
    for name in ("repeats", "inpainting_iterations", "reduced_shear_iterations"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.snr_shuffles != 0 and args.snr_shuffles < 2:
        parser.error("--snr-shuffles must be zero or at least two")
    if args.gradient and args.backend == "numpy":
        parser.error("--gradient requires a JAX backend")
    return args

def main():
    """Run the selected benchmarks and write a JSON timing report."""
    args = _parse_args()
    jax_info = None
    device = None
    backend_settings = {}
    if args.backend == "numpy":
        jax = None
    else:
        import jax

        backend_settings = {"backend": "jax"}
        jax.config.update("jax_platforms", "cpu" if args.backend == "jax-cpu" else "cuda")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_disable_jit", False)
        jax.config.update("jax_enable_compilation_cache", False)
        devices, init_seconds = _timed(jax.devices)
        device = devices[0]
        jax.config.update("jax_default_device", device)
        from smpy.jax import ArrayMapper

        jax_info = {
            "device": str(device),
            "platform": device.platform,
            "device_kind": device.device_kind,
            "backend_init_seconds": init_seconds,
            "enable_x64": jax.config.jax_enable_x64,
            "enable_compilation_cache": jax.config.jax_enable_compilation_cache,
        }

    import smpy

    packages = ["numpy", "scipy", "astropy", "pandas", "pyyaml"]
    if jax is not None:
        packages.extend(["jax", "jaxlib"])
    environment_keys = ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION")
    report = {
        "backend": args.backend,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revision": args.revision,
        "smpy_file": str(Path(smpy.__file__).resolve()),
        "smpy_version": smpy.__version__,
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in packages},
        "host": platform.node(),
        "host_platform": platform.platform(),
        "host_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "environment": {key: os.environ[key] for key in environment_keys if key in os.environ},
        "jax": jax_info,
        "cases": {},
    }
    for case in args.case:
        print(f"{args.backend} {case}: preparing catalog", file=sys.stderr, flush=True)
        (table, settings, metadata), preparation = _timed(partial(_load_case, case, args.cluster_fits))
        metadata.update(
            catalog_preparation_seconds=preparation,
            input_galaxies=len(table),
            positive_weight_galaxies=int(np.count_nonzero(table[settings["weight_col"]] > 0)),
            catalog_settings=settings,
        )
        report["cases"][case] = {"catalog": metadata, "methods": {}}
        for method in args.method:
            print(f"{args.backend} {case} {method}: timing maps", file=sys.stderr, flush=True)
            method_settings = _method_settings(args, case, method)
            call = partial(_map_mass_host, smpy.map_mass, table, method,
                           **settings, **method_settings, **backend_settings)
            if jax is not None:
                jax.clear_caches()
            (result, maps), timing = _time_series(call, args.repeats)
            grid = result.grid
            record = {
                "method_settings": result.settings["methods"][method],
                "grid_shape": list(grid.g1.shape),
                "gridded_galaxies": int(np.asarray(grid.counts).sum()),
                "positive_weight_pixels": int(np.count_nonzero(np.asarray(grid.weights) > 0)),
                "map_dtypes": {mode: str(maps[mode].dtype) for mode in "EB"},
                "map_mass": timing,
            }
            if args.snr_shuffles:
                print(f"{args.backend} {case} {method}: timing SNR", file=sys.stderr, flush=True)
                snr_settings = dict(num_shuffles=args.snr_shuffles, shuffle_type="orientation", seed=0)
                _, seconds = _timed(partial(call, create_snr=True, snr=snr_settings))
                record["map_mass_snr"] = {"settings": snr_settings, "seconds": seconds}
            if jax is not None:
                make_mapper = partial(ArrayMapper, method, **method_settings)
                record["map_grid"] = _device_timings(jax, device, make_mapper, grid, maps,
                                                     args.repeats, args.gradient)
            report["cases"][case]["methods"][method] = record
        del call, table

    output = json.dumps(report, indent=2, allow_nan=False)
    if args.output is None:
        print(output)
    else:
        args.output.write_text(output + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
