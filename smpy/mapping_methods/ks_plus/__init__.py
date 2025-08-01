"""Kaiser-Squires Plus (KS+) mass mapping implementation.

This module provides the KS+ mass mapping algorithm, an enhanced version
of Kaiser-Squires that includes iterative inpainting for missing data,
reduced shear corrections, and wavelet-based constraints for improved
mass reconstruction accuracy.
"""

from .ks_plus import KSPlusMapper

__all__ = ['KSPlusMapper']