"""Array conversion and scoped precision for NumPy-facing calls."""

from functools import wraps

import jax
import numpy as np

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

__all__ = ["enable_x64", "native_array", "numpy_precision", "to_numpy"]

def native_array(array):
    """Convert non-native NumPy byte order before passing data to JAX."""
    if isinstance(array, np.ndarray) and not array.dtype.isnative:
        return array.astype(array.dtype.newbyteorder("="))
    return array

def numpy_precision(function):
    """Run host-array calls in double-precision mode."""
    @wraps(function)
    def call(*args, **kwargs):
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        if any(isinstance(leaf, (jax.Array, jax.core.Tracer)) for leaf in leaves):
            return function(*args, **kwargs)
        if not any(isinstance(leaf, np.ndarray) for leaf in leaves):
            return function(*args, **kwargs)
        with enable_x64(True):
            args, kwargs = jax.tree_util.tree_map(native_array, (args, kwargs))
            return to_numpy(function(*args, **kwargs))

    return call

def to_numpy(tree):
    """Transfer arrays to the host with writable NumPy storage."""
    def writable(array):
        if isinstance(array, np.ndarray) and not array.flags.writeable:
            return array.copy()
        return array

    return jax.tree_util.tree_map(writable, jax.device_get(tree))
