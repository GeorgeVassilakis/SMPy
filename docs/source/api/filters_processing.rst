Filters & Processing
====================

Filtering and signal processing tools for mass mapping.

This module provides various filtering techniques including aperture mass filters,
starlet wavelet transforms, and smoothing operations.

Aperture Mass Filters
---------------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.filters.processing.schirmer_filter
   smpy.filters.processing.schneider_filter
   smpy.filters.processing.create_filter_kernel
   smpy.filters.processing.apply_filter_convolution
   smpy.filters.processing.apply_aperture_filter

Starlet Wavelet Processing
--------------------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.filters.starlet.b3spline_filter
   smpy.filters.starlet.apply_filter
   smpy.filters.starlet.starlet_transform_2d
   smpy.filters.starlet.inverse_starlet_transform_2d

Plotting Filters
----------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.filters.plotting.apply_filter