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

X-ray Contour Overlay
---------------------

Set ``plotting.xray_contours.ctr_file`` to a DS9 ``.ctr`` file and toggle
overlay behavior independently with:

- ``plotting.xray_contours.show_on_convergence``
- ``plotting.xray_contours.show_on_snr``

Supported contour coordinate declarations are ``fk5``/``wcs`` and
``image``/``physical``.
