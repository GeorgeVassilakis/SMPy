Plotting
========

Visualization tools for mass mapping results.

Functions for creating publication-quality plots of mass and SNR maps.

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.plotting.plot.plot_mass_map
   smpy.plotting.plot.plot_snr_map

Counts Map
----------

When ``general.create_counts_map: true`` is set, SMPy will generate a per-pixel
object counts map using the same plotting extent as the mass maps and save it as
``<output_base_name>_<method>_counts.png`` in the method's output directory.