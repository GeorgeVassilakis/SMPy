Validation
==========

SMPy's maps are checked against answers known independently of SMPy: exact
algebra, the known effect of transforming a catalog, and simulated catalogs
whose true convergence is known. The checks run as tests on every change, and
``validation/accuracy.py`` makes the figures and numbers on this page.

What the tests check
--------------------

``tests/test_physics.py`` holds the physics checks:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Check
     - Independent answer
   * - Kaiser-Squires relations
     - The shear of a convergence map matches the textbook Fourier relation
       exactly, and inverting it returns the convergence, on odd and even
       grids.
   * - Rotation
     - Turning every shear by an angle turns E + iB by twice that angle for
       Kaiser-Squires and aperture mass, and for KS+ without its reduced-shear
       correction and wavelet constraint, which treat E and B separately.
   * - Cluster recovery
     - Each method finds the NFW cluster of a noiseless simulated catalog
       where it is, with little B mode, in RA/Dec and in pixel coordinates.
   * - KS+ inpainting
     - Inside a hole, KS+ recovers a field of a few Fourier modes as
       Kaiser-Squires does from the complete shear, and it beats
       Kaiser-Squires inside holes cut in the simulated catalog.
   * - Starlet transform
     - The inverse transform restores the image to rounding error.
   * - Aperture mass
     - A constant shear has no aperture mass, and shears beyond the aperture
       radius add nothing.
   * - Sky projection
     - Tangential shear around a lens 4.7 degrees from the center of a
       12-degree field maps to E with almost no B, and the FITS WCS puts
       every galaxy in the pixel it was binned into.
   * - SNR
     - The noise behind an aperture-mass SNR map equals the exact shape noise
       of each pixel, and KS+ gives the Kaiser-Squires SNR where their maps
       agree.

``tests/test_reference.py`` also compares every output with stored results, so
any change to the numbers is deliberate.

Recovering a simulated cluster
------------------------------

The truth catalog in ``examples/data/forecast_lum_truth.fits`` is the input
catalog of one realization of the SuperBIT image simulations of
`McCleary et al. (2023) <https://doi.org/10.3847/1538-3881/ace7ca>`_, the same
field as the example shape catalog. The cluster is a
`GalSim <https://doi.org/10.1016/j.ascom.2015.02.002>`_ NFW halo with
:math:`M_{\mathrm{200c}} = 4.1 \times 10^{14}\,h^{-1} M_\odot`, concentration
:math:`c = 4` and redshift :math:`z = 0.45`, for :math:`\Omega_\mathrm{m} = 0.3`
and :math:`\Omega_\Lambda = 0.7`, at the center of a :math:`22.5' \times 15.0'`
field. Galaxies are placed at random, with redshifts and fluxes drawn from a
mock SuperBIT catalog built on COSMOS 2015, and GalSim gives each galaxy behind
the cluster the reduced shear and magnification of the halo at its own
redshift. The catalog holds these noiseless values for 34,298 galaxies, 30
cluster galaxies and 376 stars. The stars, the cluster galaxies and the
galaxies in front of the cluster have zero shear and unit magnification. The
maps are made from the 29,545 galaxies behind the cluster, as a shear catalog
would hold them.

The catalog has no convergence column, so each galaxy's convergence
:math:`\kappa` is recovered from its reduced shear :math:`g` and magnification
:math:`\mu` through :math:`\mu = [(1 - \kappa)^2 (1 - |g|^2)]^{-1}`. The maps use 0.4 arcminute pixels and
the default settings, and they are compared with this convergence averaged over
each pixel and smoothed like the maps. Shear does not determine the mean
convergence, so every map is shown and compared with its mean removed.

.. image:: _static/validation/truth_recovery.png
   :alt: The true convergence of the simulated cluster next to the
         Kaiser-Squires, KS+ and aperture-mass maps of its shear.

.. list-table::
   :header-rows: 1

   * - Method
     - Correlation with the truth
     - RMS error (peak 0.145)
     - B-mode / E-mode spread
   * - Kaiser-Squires
     - 0.9711
     - 0.0045
     - 0.167
   * - KS+
     - 0.9996
     - 0.0009
     - 0.021
   * - Aperture mass
     -
     -
     - 0.108

All three maps peak within one pixel of the true peak. By correcting for
reduced shear and the field border, KS+ cuts the error of Kaiser-Squires by a
factor of 5 and its B mode by a factor of 7.8. Aperture mass is a filtered statistic, not a
convergence map, so it is compared only by its peak and B mode.

Filling holes
-------------

Masks around bright stars leave holes in real catalogs. Here every galaxy
within 1.2 arcminutes of eight random points gets zero weight, and the maps,
unsmoothed, are compared with the true convergence. Green lines outline the
holes.

.. image:: _static/validation/holes.png
   :alt: The true convergence with eight holes outlined, and the error of the
         Kaiser-Squires and KS+ maps.

Inside the holes, the RMS error is 0.0132 for Kaiser-Squires and 0.0059 for
KS+. KS+ also removes the Kaiser-Squires errors along the field border.

Calibration of the SNR maps
---------------------------

Turning every shear of the example catalog by a random angle removes its
lensing signal and keeps its noise, so the SNR map of such a catalog should
have zero mean and unit spread. The histogram pools the E-mode SNR maps of ten
such catalogs, each with 100 shuffles.

.. image:: _static/validation/snr_calibration.png
   :alt: Histograms of the SNR of catalogs without lensing signal for the
         three methods, matching a unit normal distribution.

The pooled spread is 1.00 for Kaiser-Squires, 1.04 for KS+ and 1.00 for
aperture mass; with 100 shuffles the expected spread is 1.01. Between
catalogs, the spread of a single map scatters by 5% to 14%, as neighboring
pixels are correlated, so one map alone is a poor check.
