"""Filters package for SMPy.

This package provides various filtering and smoothing operations for mass
mapping including Gaussian smoothing, aperture mass filters (Schirmer and
Schneider), and starlet wavelet transforms used in reconstruction algorithms.
"""

from .plotting import apply_filter
from .processing import apply_aperture_filter

__all__ = [
    'apply_filter',           # From plotting.py
    'apply_aperture_filter',  # From processing.py
]