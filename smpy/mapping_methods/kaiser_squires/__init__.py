"""Kaiser-Squires mass mapping implementation.

This module provides the Kaiser-Squires mass mapping algorithm, a direct
inversion method for reconstructing convergence maps from shear measurements
using Fourier transforms. Includes optional Gaussian smoothing for noise
reduction.
"""

from .kaiser_squires import KaiserSquiresMapper

__all__ = ['KaiserSquiresMapper']