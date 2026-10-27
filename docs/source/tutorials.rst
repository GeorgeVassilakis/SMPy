Tutorials
=========

Each tutorial is a notebook, in ``docs/source/tutorials`` of the repository,
that runs in under a minute on the example catalogs in ``examples/data``.

:doc:`tutorials/01_getting_started`
   Convergence maps of a simulated cluster from its shear catalog, their
   signal-to-noise maps and FITS output, and the same run from a YAML file.

:doc:`tutorials/02_coordinates_and_gridding`
   How SMPy projects sky positions, orients the shear and bins galaxies into
   pixels, in sky and in image coordinates.

:doc:`tutorials/03_mass_mapping_methods`
   Kaiser-Squires, KS+ and aperture mass on the same catalog, compared with
   the true convergence of a noiseless simulation, with and without holes.

:doc:`tutorials/04_signal_to_noise_maps`
   Noise from randomized catalogs, its calibration, orientation and spatial
   shuffles, the B mode as a null test, and peak finding.

.. toctree::
   :hidden:

   tutorials/01_getting_started
   tutorials/02_coordinates_and_gridding
   tutorials/03_mass_mapping_methods
   tutorials/04_signal_to_noise_maps
