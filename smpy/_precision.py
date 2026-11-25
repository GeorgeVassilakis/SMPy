"""Array conversion and scoped precision for NumPy-facing calls."""

from functools import wraps

import jax
import numpy as np

try:
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

__all__ = ["enable_x64", "native_array", "numpy_double", "numpy_precision", "to_numpy"]

def native_array(array):
    """Convert non-native NumPy byte order before passing data to JAX."""
    if isinstance(array, np.ndarray) and not array.dtype.isnative:
        return array.astype(array.dtype.newbyteorder("="))
    return array

def _double_array(array):
    """Promote real NumPy inputs to native float64."""
    if isinstance(array, np.ndarray) and array.dtype.kind == "f":
        return array.astype(np.float64, copy=False)
    return native_array(array)

def numpy_double(function):
    """Run real NumPy array calls in float64 and preserve JAX dtypes."""
    return _numpy_dispatch(function, _double_array)

def numpy_precision(function):
    """Run host-array calls in double-precision mode."""
    return _numpy_dispatch(function, native_array)

def _numpy_dispatch(function, convert):
    """Adapt NumPy calls while keeping traced JAX calls on the device."""
    @wraps(function)
    def call(*args, **kwargs):
        leaves = jax.tree_util.tree_leaves((args, kwargs))
        if any(isinstance(leaf, (jax.Array, jax.core.Tracer)) for leaf in leaves):
            return function(*args, **kwargs)
        if not any(isinstance(leaf, np.ndarray) for leaf in leaves):
            return function(*args, **kwargs)
        with enable_x64(True):
            args, kwargs = jax.tree_util.tree_map(convert, (args, kwargs))
            return to_numpy(function(*args, **kwargs))

    return call

def to_numpy(tree):
    """Transfer arrays to the host with writable NumPy storage."""
    def writable(array):
        if isinstance(array, np.ndarray) and not array.flags.writeable:
            return array.copy()
        return array

    return jax.tree_util.tree_map(writable, jax.device_get(tree))
