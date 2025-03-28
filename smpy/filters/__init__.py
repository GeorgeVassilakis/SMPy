"""Filters package for SMPy."""

from .plotting import apply_filter
from .processing import apply_aperture_filter

__all__ = [
    'apply_filter',           # From plotting.py
    'apply_aperture_filter',  # From processing.py
]