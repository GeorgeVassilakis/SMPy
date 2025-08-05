Mass Mapping Methods
====================

Implementation classes for different mass mapping algorithms.

These classes implement the core algorithms for reconstructing mass distributions
from weak lensing shear measurements.

Base Class
----------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.mapping_methods.base.MassMapper

Algorithm Implementations
-------------------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.mapping_methods.kaiser_squires.kaiser_squires.KaiserSquiresMapper
   smpy.mapping_methods.aperture_mass.aperture_mass.ApertureMassMapper
   smpy.mapping_methods.ks_plus.ks_plus.KSPlusMapper