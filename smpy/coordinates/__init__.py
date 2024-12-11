from .base import CoordinateSystem
from .radec import RADecSystem
from .pixel import PixelSystem

def get_coordinate_system(system_name):
    """Create the appropriate coordinate system instance.

    Parameters
    ----------
    system_name : `str`
        Name of coordinate system ('radec' or 'pixel')
        
    Returns
    -------
    CoordinateSystem
        Instance of RADecSystem or PixelSystem
        
    Raises
    ------
    ValueError
        If system_name is not 'radec' or 'pixel'
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