Differentiable mass mapping
===========================

This page requires the ``jax`` extra. See :doc:`installation`.

`smpy.jax.ArrayMapper` accepts arrays from a JAX simulation or likelihood.
Use `smpy.jax.ArrayMapper.map_grid` for gridded shear, or
`smpy.jax.ArrayMapper.map_catalog` for galaxy positions, shear components
and weights. Both use the JAX kernels for Kaiser-Squires, KS+ and aperture
mass that `smpy.map_mass` uses with ``backend="jax"``. Tests compare the
JAX kernels against the NumPy implementation.

The example differentiates the variance of an E-mode map with respect to
the amplitude of a shear model and the Gaussian smoothing width.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from smpy.jax import ArrayMapper

   jax.config.update("jax_enable_x64", True)
   mapper = ArrayMapper("kaiser_squires", smoothing=3.0)
   mode = jnp.cos(2 * jnp.pi * jnp.arange(33) / 33)
   shear = jnp.broadcast_to(mode, (25, 33))

   def loss(amplitude, sigma):
       maps = mapper.map_grid(amplitude * shear, jnp.zeros_like(shear),
                              smoothing=sigma)
       return jnp.mean(maps["E"] ** 2)

   value_and_gradient = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
   value, (d_amplitude, d_sigma) = value_and_gradient(0.02, 1.0)

In a cosmological workflow, pass the shear or weights predicted by your
forward model to the mapper. JAX propagates the map loss back to any
cosmological or redshift parameters used to compute those inputs.
The mapper operates on a specified flat grid or TAN patch.

Choosing gradients
------------------

* Shear components and catalog weights are continuous inputs to the map.
  A weight of zero can change the KS+ observation mask. Its derivative
  holds the mask fixed.
* Use bilinear gridding for subpixel position gradients. It distributes
  each galaxy's weight among pixel centers, so it defines a different
  binned estimator from the containing-pixel rule used by `smpy.map_mass`.
  Supply ``tangent_point=(ra0, dec0)`` for sky coordinates in degrees,
  including the rotation from local shear into the map frame.
  ``map_catalog`` excludes galaxies outside the grid, with non-finite
  positions or at least 90 degrees from ``tangent_point``. These galaxies
  contribute nothing to shear sums, weight sums or counts. A galaxy's
  contribution changes discontinuously at the grid edge, where its position
  derivative is undefined.
* Set the constructor's smoothing width to bound the traced sigma in a
  KS or KS+ call. For aperture mass, the product of the traced scale and
  cutoff must fit inside the configured kernel radius. Gradients are
  computed within a fixed pixel footprint. The aperture cutoff has zero
  derivative between footprint changes.
* KS+ differentiates the finite inpainting and reduced-shear iterations.
  Hard DCT threshold crossings, discrete choices of wavelet scale and
  changes to the grid shape are outside these derivatives.
* To differentiate a catalog SNR map, pass a JAX random key and a fixed
  number of shuffles to ``map_catalog``. Hold the key fixed when evaluating
  nearby parameter values. Pixels with zero variance have undefined SNR,
  represented by NaNs. `smpy.map_mass` uses integer seeds on both backends.

`smpy.jax.ArrayMapper` computes in float32 by default. Enable
``jax_enable_x64`` before creating arrays to use float64. Use float64 for
KS+ with wavelet constraints on fields that are smooth inside the gaps: a
wavelet band whose spread there is below four rounding units of the largest
map value, about ``5e-7`` of it in float32, is rounding error and is left
unscaled. The JAX backend for `smpy.map_mass` and `smpy.run` always computes
in float64 and returns NumPy maps for plotting and FITS output.

API
---

.. autoclass:: smpy.jax.ArrayMapper
   :members: map_grid, map_catalog

.. autoclass:: smpy.jax.ArrayMassMap
   :no-members:
