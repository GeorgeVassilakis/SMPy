from .base import CoordinateSystem
from .radec import RADecSystem
from .pixel import PixelSystem

def get_coordinate_system(system_name):
    """
    Factory function to get the appropriate coordinate system.
    
    Parameters
    ----------
    system_name : str
        Name of the coordinate system ('radec' or 'pixel')
        
    Returns
    -------
    CoordinateSystem
        Instance of the requested coordinate system
        
    Raises
    ------
    ValueError
        If system_name is not recognized
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