Performance
===========

SMPy compiles the mapping kernels with JAX. Repeated calls can reuse the
compiled kernels when their array shapes, precision and static settings
match. Catalog loading, plotting and FITS output run on the host.

XTX benchmarks
--------------

The table gives median warmed `smpy.map_mass` times over three repeats
in float64. Timing starts with an in-memory table and ends with NumPy
E/B maps. The baseline is SMPy at commit ``7e5920b`` on XTX CPUs.
The accelerator measurements use one NVIDIA B200. KS+ uses its default
100 inpainting iterations and three reduced-shear iterations.

.. list-table::
   :header-rows: 1

   * - Catalog
     - Map
     - Method
     - Baseline CPU
     - JAX CPU
     - JAX B200
   * - Cluster, 8,767 galaxies
     - 38 × 57
     - Kaiser–Squires
     - 0.0063 s
     - 0.0051 s
     - 0.0066 s
   * -
     -
     - Aperture mass
     - 0.0093 s
     - 0.0065 s
     - 0.0057 s
   * -
     -
     - KS+
     - 0.71 s
     - 0.43 s
     - 0.13 s
   * - 5° × 5° field, 1.46 million galaxies
     - 150 × 150
     - Kaiser–Squires
     - 0.12 s
     - 0.053 s
     - 0.036 s
   * -
     -
     - Aperture mass
     - 0.13 s
     - 0.054 s
     - 0.037 s
   * -
     -
     - KS+
     - 4.27 s
     - 3.67 s
     - 0.16 s
   * - 10° × 10° survey tile, 10.8 million galaxies
     - 300 × 300
     - Kaiser–Squires
     - 0.99 s
     - 0.50 s
     - 0.26 s
   * -
     -
     - Aperture mass
     - 1.07 s
     - 0.53 s
     - 0.26 s
   * -
     -
     - KS+
     - 20.36 s
     - 29.21 s
     - 0.45 s

First calls on the B200 took about 0.8–3.1 s after backend initialization,
including compilation. For repeated inference on shear arrays already
resident on the B200, a KS+ survey-tile map took about 0.19 s and a
compiled value and gradient of the mean squared E/B maps took about 0.82 s.

The cluster is the bundled example FITS catalog. The field and survey tile
are seeded mocks with shape noise, varying weights, masked holes and
known shear modes. Their flat-patch sizes and input densities follow the
scales used in the Euclid studies below.

* 5° × 5° and 16.2 galaxies per square arcminute.
  `Euclid preparation LXXXV <https://doi.org/10.1051/0004-6361/202557573>`_
* 10° × 10° and 30 galaxies per square arcminute.
  `Pires et al. (2020) <https://arxiv.org/abs/1910.03106>`_

Reproducing benchmarks
----------------------

Run ``validation/jax_performance.py`` in a fresh process per backend. The
JSON reports include cold calls, warm repeats, catalog preparation and
configurations. The JAX reports include device placement and resident
grid times. Use ``--case smoke`` for a small run and ``--snr-shuffles 20``
for a null-map benchmark.

.. code-block:: bash

   PYTHONPATH=/path/to/original/SMPy python validation/jax_performance.py \
       --backend numpy --output baseline.json
   PYTHONPATH=. python validation/jax_performance.py \
       --backend jax-cpu --output cpu.json
   CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python validation/jax_performance.py \
       --backend jax-gpu --gradient --output gpu.json

Signal-to-noise maps
--------------------

The forward SNR calculation streams one shuffled catalog at a time and
accumulates its mean and variance online. Its map storage scales with the
number of pixels. KS+ also processes starlet bands on the padded grid.
Runtime scales with the number of shuffles. For Gaussian null-map samples,
the relative uncertainty in the estimated noise is about 7% at 100
shuffles and 10% at 50.

For the example cluster and 20 orientation shuffles, the first KS+ SNR
call took 14.9 s on the baseline CPU, 9.45 s on the JAX CPU and 2.80 s on
the B200. The signal mapper was warmed before the SNR call.
