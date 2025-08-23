Tutorials
=========

00 – One‑liner CLI
------------------

Create a convergence map with a single command.

Prerequisites
^^^^^^^^^^^^^

- SMPy installed and this repository cloned.
- Example data available at ``examples/data/forecast_lum_annular.fits``.
- Ensure the configuration file path is valid for your local system: ``smpy/configs/example_config.yaml``.

Command
^^^^^^^

.. code-block:: bash

   python runner.py -c smpy/configs/example_config.yaml

What happens
^^^^^^^^^^^^

- Loads the example configuration and runs a Kaiser–Squires map by default.
- Uses the input shear catalog defined in the config (update ``general.input_path`` if needed).
- Writes outputs according to the config. If ``save_fits: true`` is set, FITS files are saved under
  ``<output_directory>/<method>/``. Plots are controlled by the ``plotting`` section.

Next steps
^^^^^^^^^^

- Prefer Python over CLI? See the Quickstart notebook for the same run via the high‑level API.

Subpages
--------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials index

   tutorials/01_quickstart
   tutorials/02_configuration
   tutorials/03_coordinates
   tutorials/04_kaiser_squires
   tutorials/05_aperture_mass
   tutorials/06_ks_plus
   tutorials/07_snr_and_validation
