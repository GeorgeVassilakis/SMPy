"""Coordinate system implementations for SMPy mass mapping.

This module provides coordinate system classes for handling RA/Dec and
pixel coordinate transformations and gridding operations used in weak
lensing mass mapping.
"""

from .base import CoordinateSystem
from .radec import RADecSystem
from .pixel import PixelSystem

def get_coordinate_system(system_name):
    """Create the appropriate coordinate system instance.

    Factory function that returns the correct coordinate system implementation
    based on the specified system name.

    Parameters
    ----------
    system_name : `str`
        Name of coordinate system ('radec' or 'pixel').

    Returns
    -------
    coord_system : `CoordinateSystem`
        Instance of RADecSystem or PixelSystem subclass.

    Raises
    ------
    ValueError
        If system_name is not 'radec' or 'pixel'.
    """
    systems = {
        'radec': RADecSystem,
        'pixel': PixelSystem
    }
    
    system = systems.get(system_name.lower())
    if system is None:
        raise ValueError(f"Unknown coordinate system: {system_name}. "
                       f"Available systems: {list(systems.keys())}")
    
    return system()