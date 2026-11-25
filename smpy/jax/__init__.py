"""Optional JAX mass mapping interfaces."""

try:
    from .arrays import ArrayMapper, ArrayMassMap
except ModuleNotFoundError as exc:
    if exc.name != "jax":
        raise
    raise ImportError('The JAX backend requires JAX. Run pip install "smpy-lensing[jax]".') from exc

__all__ = ["ArrayMapper", "ArrayMassMap"]
