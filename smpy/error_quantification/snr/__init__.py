"""Signal-to-noise ratio (SNR) computation for mass mapping.

This module provides functionality for computing signal-to-noise ratio
maps from mass reconstruction results using randomized null hypothesis
testing through spatial or orientation shuffling of the input data.
"""

from .run import run

__all__ = ['run']