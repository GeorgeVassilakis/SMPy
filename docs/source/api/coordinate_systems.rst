Coordinate Systems
==================

Coordinate system handling for different input data formats.

SMPy supports both celestial (RA/Dec) and pixel-based coordinate systems for
flexible handling of different observational data formats.

Factory Function
----------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.coordinates.get_coordinate_system

Base Class
----------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.coordinates.base.CoordinateSystem

Implementations
---------------

.. autosummary::
   :toctree: ../_autosummary
   :nosignatures:

   smpy.coordinates.radec.RADecSystem
   smpy.coordinates.pixel.PixelSystem