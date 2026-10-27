Plotting
========

Visualization tools for mass mapping results.

Functions for creating publication-quality plots of mass and SNR maps. Most
users call `smpy.MassMap.plot`, which passes the ``plotting`` settings to
these functions.

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.plotting.plot.plot_mass_map

Counts Map
----------

`smpy.MassMap.plot_counts` plots the number of galaxies per pixel using the
same extent as the mass maps. With ``general.create_counts_map: true``,
`smpy.run` saves it as ``<output_base_name>_<method>_counts.png`` in the
method's output directory. ``plotting.overlay_counts: true`` writes the counts
on the map plots.

X-ray Contour Overlay
---------------------

Set ``plotting.xray_contours.ctr_file`` to a DS9 ``.ctr`` file and toggle
overlay behavior independently with:

- ``plotting.xray_contours.show_on_convergence``
- ``plotting.xray_contours.show_on_snr``

Supported contour coordinate declarations are ``fk5``/``wcs`` and
``image``/``physical``.
