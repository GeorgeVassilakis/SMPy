"""Aperture mass mapping implementation.

This module provides the aperture mass mapping algorithm, which uses
optimal filter functions (Schirmer or Schneider filters) to reconstruct
mass maps from weak lensing shear data. The method applies convolution
with compact support filters for localized mass reconstruction.
"""

from .aperture_mass import ApertureMassMapper

__all__ = ['ApertureMassMapper']