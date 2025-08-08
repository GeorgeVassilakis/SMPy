"""SMPy: Shear Mapping in Python.

A mass reconstruction toolkit for weak gravitational lensing analysis that
provides easy-to-use functions for undergraduates while maintaining
robustness for senior researchers.

This package provides mass mapping methods including Kaiser-Squires,
aperture mass, and KS+ algorithms for weak lensing analysis.
"""

# Simple API functions - main entry points for users
from .api import (
    map_aperture_mass,
    map_kaiser_squires,
    map_ks_plus,
    map_mass,
)

# Config API - for power users who want more control
from .config import Config
from .run import run

# Version info
__version__ = "0.5.0"

# Define what gets imported with "from smpy import *"
__all__ = [
    'map_mass',
    'map_kaiser_squires', 
    'map_aperture_mass',
    'map_ks_plus',
    'Config',
    'run',
    '__version__'
]