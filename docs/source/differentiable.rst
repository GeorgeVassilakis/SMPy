Differentiable mass mapping
===========================

`smpy.ArrayMapper` accepts arrays from a JAX simulation or likelihood.
Use `smpy.ArrayMapper.map_grid` for gridded shear, or
`smpy.ArrayMapper.map_catalog` for galaxy positions, shear components and
weights. Both use the same Kaiser–Squires, KS+ and aperture mass kernels as
`smpy.map_mass`.

The example differentiates the variance of an E-mode map with respect to
the amplitude of a shear model and the Gaussian smoothing width.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from smpy import ArrayMapper

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
* Set the constructor's smoothing width to bound the traced sigma in a
  KS or KS+ call. For aperture mass, the product of the traced scale and
  cutoff must fit inside the configured kernel radius. Gradients are
  computed within a fixed pixel footprint. The aperture cutoff has zero
  derivative between footprint changes.
* KS+ differentiates the finite inpainting and reduced-shear iterations.
  Hard DCT threshold crossings, discrete choices of wavelet scale and
  changes to the grid shape are outside these derivatives.
* To differentiate a catalog SNR map, pass a JAX random key and a fixed
  number of shuffles. Hold the key fixed when evaluating nearby parameter
  values. Pixels with zero variance have undefined SNR, represented by
  NaNs. The legacy catalog API uses its original integer seed streams.

Use ``jax_enable_x64`` to choose double precision for the array API.
`smpy.map_mass` and `smpy.run` compute in double precision and return
NumPy maps for plotting and FITS output.

API
---

.. autoclass:: smpy.ArrayMapper
   :members: map_grid, map_catalog

.. autoclass:: smpy.ArrayMassMap
   :no-members:
