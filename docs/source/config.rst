Configuration Guide
===================

SMPy settings live in one place: ``smpy/configs/default.yaml`` holds every
setting and its default. A YAML file passed to ``smpy -c`` or
`smpy.Config.from_file`, and keyword arguments to `smpy.map_mass` or
`smpy.Config.from_defaults`, only give the settings that differ. Keyword
arguments use the same names as the YAML keys.

Configuration Structure
-----------------------

The configuration has four sections:

.. code-block:: yaml

   general:      # Input catalog, coordinate system, method and outputs
   methods:      # Method-specific parameters (kaiser_squires, aperture_mass, ks_plus)
   plotting:     # Visualization settings
   snr:          # Signal-to-noise ratio calculation settings

How Settings Combine
--------------------

- Settings you give replace the defaults key by key; everything else keeps its
  default. An empty file or section keeps all defaults.
- A setting that does not exist raises an error naming it, so a typo such as
  ``smoothng`` fails instead of being ignored.
- ``validate`` (called by `smpy.run` and `smpy.map_mass`) reports missing
  required settings and invalid values by name.
- Relative paths in a YAML file (``input_path``, ``output_directory`` and
  ``plotting.xray_contours.ctr_file``) resolve from that file's directory.
  `smpy.Config.save` rewrites them so they still point at the same files from
  the new location.
- `smpy.run` writes ``<output_base_name>_<method>_settings.yaml`` next to its
  outputs: every setting used, with absolute paths and the SMPy version. It can
  be passed back to ``smpy -c`` to repeat the run.

.. code-block:: python

   from smpy import Config

   config = Config.from_file('my_config.yaml')        # defaults + file
   config = Config.from_defaults('ks_plus', input_path='catalog.fits',
                                 pixel_scale=0.4, inpainting_iterations=50)
   config.save('my_run.yaml')
   print(config)                                      # every setting as YAML

Parameter Reference
-------------------

.. note::
   Required parameters are marked with an asterisk (*).

General Settings
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - input_path*
     - string
     - ""
     - Input FITS catalog. Required by ``smpy.run``; `smpy.map_mass` takes
       the catalog as its first argument instead
   * - input_hdu
     - integer
     - 1
     - FITS extension holding the catalog
   * - coord_system
     - string
     - "radec"
     - Coordinate system: 'radec' or 'pixel'
   * - g1_col
     - string
     - "g1"
     - Column name for the first shear component
   * - g2_col
     - string
     - "g2"
     - Column name for the second shear component
   * - weight_col
     - string/null
     - null
     - Column name for weights (null for unit weights)
   * - method
     - string
     - "kaiser_squires"
     - Mass mapping method: 'kaiser_squires', 'aperture_mass', or 'ks_plus'
   * - create_snr
     - boolean
     - false
     - Also compute signal-to-noise ratio maps
   * - output_directory
     - string
     - "."
     - Directory for outputs, which go to ``<output_directory>/<method>/``
   * - output_base_name
     - string
     - "smpy_output"
     - Base name for output files
   * - mode
     - list
     - ['E']
     - Modes to write: ['E'], ['B'], or ['E', 'B']. Both modes are always
       computed
   * - save_plots
     - boolean
     - true
     - Write PNG plots
   * - save_fits
     - boolean
     - false
     - Write FITS maps with a WCS header
   * - create_counts_map
     - boolean
     - false
     - Write a plot of galaxies per pixel

Only ``smpy.run`` and ``smpy -c`` write outputs; `smpy.map_mass` ignores the
output settings. A NaN, infinite or masked value in a coordinate, shear or
weight column raises an error naming the column; remove those rows first.

RA/Dec Coordinate Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``general.radec``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - pixel_scale*
     - float
     - null
     - Map pixel size in arcminutes
   * - coord1_col
     - string
     - "ra"
     - RA column name, in degrees
   * - coord2_col
     - string
     - "dec"
     - Dec column name, in degrees

RA and Dec columns without a unit are read as degrees; columns that declare
an angle unit, such as ``rad``, are converted, and other units raise an error.
RA/Dec catalogs are projected onto the plane tangent to the sky at the field
centre (the gnomonic, or TAN, projection), with north up and east to the left,
and binned into square pixels of ``pixel_scale``. Galaxies 90 degrees or more
from the centre raise an error. ``g1`` and ``g2`` are read in
each galaxy's local frame, with the first axis towards the west and the second
towards the north, and are turned into the map frame. FITS files carry the
matching TAN WCS.

Pixel Coordinate Settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Under ``general.pixel``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - downsample_factor*
     - integer
     - null
     - Catalog pixels per map pixel
   * - coord1_col
     - string
     - "X_IMAGE"
     - X coordinate column name
   * - coord2_col
     - string
     - "Y_IMAGE"
     - Y coordinate column name

Pixel catalogs are binned into square pixels of ``downsample_factor`` catalog
pixels. ``g1`` and ``g2`` are read along the catalog's x and y axes.

Method-Specific Settings
~~~~~~~~~~~~~~~~~~~~~~~~

Kaiser-Squires Settings
^^^^^^^^^^^^^^^^^^^^^^^

Under ``methods.kaiser_squires``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - smoothing
     - float
     - 2.0
     - Gaussian smoothing sigma in map pixels; 0 for no smoothing

Aperture Mass Settings
^^^^^^^^^^^^^^^^^^^^^^

Under ``methods.aperture_mass``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - filter_type
     - string
     - "schirmer"
     - Filter type: 'schirmer' or 'schneider'
   * - filter_scale
     - float
     - 60
     - Aperture radius in map pixels
   * - truncation
     - float
     - 1.0
     - Kernel radius in units of ``filter_scale``
   * - ell
     - integer
     - 3
     - Polynomial order of the Schneider filter

KS+ Settings
^^^^^^^^^^^^

Under ``methods.ks_plus``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - smoothing
     - float
     - 2.0
     - Gaussian smoothing sigma in map pixels; 0 for no smoothing
   * - inpainting_iterations
     - integer
     - 100
     - DCT thresholding iterations per reduced-shear step
   * - reduced_shear_iterations
     - integer
     - 3
     - Reconstructions in the reduced-shear loop; 1 reconstructs once,
       without reduced-shear correction
   * - nscales
     - integer/null
     - null
     - Starlet scales for the power-spectrum constraint, at least 2. If set,
       it is clipped to a safe maximum derived from the image size and the
       B3-spline kernel support; if null, the safe maximum is used.
   * - extension_size
     - string/int
     - "double"
     - Field extension: 'double' or pixels added on each side
   * - use_wavelet_constraints
     - boolean
     - true
     - Match the E- and B-mode wavelet power inside the gaps to that
       outside

KS+ follows Algorithm 1 of Pires et al. (2020). Its DCT threshold falls from
the largest coefficient towards zero over ``inpainting_iterations`` as
``1 - erf(2.8 i / inpainting_iterations)`` (Pires et al. 2009).

Plotting Settings
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - figsize
     - list
     - [12, 8]
     - Figure size in inches [width, height]
   * - cmap
     - string
     - "viridis"
     - Matplotlib colormap name
   * - fontsize
     - integer
     - 15
     - Base font size for axis ticks, labels, title, and colorbar
   * - xlabel
     - string/null
     - "auto"
     - X-axis label ('auto', null, or custom string)
   * - ylabel
     - string/null
     - "auto"
     - Y-axis label ('auto', null, or custom string)
   * - plot_title
     - string
     - "Mass Map"
     - Plot title (method name will be appended)
   * - gridlines
     - boolean
     - true
     - Whether to show grid lines
   * - vmin
     - float/null
     - null
     - Minimum value for color scale
   * - vmax
     - float/null
     - null
     - Maximum value for color scale
   * - threshold
     - float/null
     - null
     - Peak detection threshold
   * - verbose
     - boolean/null
     - null
     - Print peak information
   * - cluster_center
     - dict/string/null
     - null
     - Center position: null, 'auto', or a dictionary. For RA/Dec plots use {ra_center: float, dec_center: float}. For pixel plots use {x_center: float, y_center: float}.
   * - axis_reference
     - string
     - "catalog"
     - Pixel plots only: label axes in catalog pixel coordinates ('catalog')
       or map pixel indices ('map')
   * - overlay_counts
     - boolean
     - false
     - Write the number of galaxies in each pixel on map plots
   * - scaling
     - dict/string
     - {type: linear}
     - Color scaling configuration. If a string, treated as {type: <string>}.
   * - xray_contours
     - dict/null
     - null
     - Optional DS9 ``.ctr`` contour overlay configuration. See x-ray
       contour settings below.

.. rubric:: Scaling configuration

The ``plotting.scaling`` setting controls normalization of colormaps. Supported forms:

.. code-block:: yaml

   plotting:
     scaling:
       type: linear   # or 'power' or 'symlog'

Options by type:

- ``type: linear``
  - Uses linear normalization with optional ``vmin``/``vmax``.

- ``type: power``
  - ``gamma``: float (default 2). Example:

    .. code-block:: yaml

       plotting:
         scaling:
           type: power
           gamma: 0.5

- ``type: symlog``
  - ``linthresh`` and ``linscale`` for each kind of map, shown with their defaults:

    .. code-block:: yaml

       plotting:
         scaling:
           type: symlog
           convergence:
             linthresh: 0.1
             linscale: 1.0
           snr:
             linthresh: 5
             linscale: 0.5

.. rubric:: X-ray contour settings

Under ``plotting.xray_contours``:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - ctr_file
     - string/null
     - null
     - Path to DS9 contour file (``.ctr``). If null, contour overlay is
       disabled. Resolved like ``general.input_path``.
   * - show_on_convergence
     - boolean
     - false
     - Whether to draw contours on convergence plots.
   * - show_on_snr
     - boolean
     - false
     - Whether to draw contours on SNR plots.
   * - color
     - string
     - "cyan"
     - Matplotlib line color for contours.
   * - linewidth
     - float
     - 0.8
     - Line width for contour traces.
   * - alpha
     - float
     - 0.7
     - Contour opacity in the range [0, 1].

SNR Settings
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - shuffle_type
     - string
     - "orientation"
     - 'orientation' turns each shear by a random angle; 'spatial' permutes
       the galaxy positions
   * - num_shuffles
     - integer
     - 100
     - Number of shuffled catalogs; at least 2
   * - seed
     - int/string
     - 0
     - Random seed (integer, or 'random' for a seed from the secrets module;
       the seed used is recorded in the settings)
   * - plot_title
     - string
     - "Signal-to-Noise Map"
     - Title for SNR plots; the mode is appended

The SNR map is the map divided by the standard deviation of the maps of
``num_shuffles`` shuffled catalogs, each made with the same method and settings
as the map. It is a detection statistic: it measures how unusual each pixel is
next to maps without lensing signal. Pixels where the shuffled maps do not vary
are NaN.

Orientation shuffles keep every galaxy's position, weight and shear amplitude,
so each pixel keeps its own noise. Spatial shuffles move shears and weights
between positions, so they give the right noise only where it is the same
across the field; where depth, weights or shape noise vary, they mix the noise
of different regions. Both keep each galaxy's shear amplitude, lensing
included, and for KS+ the spread of maps without signal need not equal the
uncertainty of a map with signal.

Example Configuration Files
---------------------------

Basic Kaiser-Squires Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/configs/example_config.yaml
   :language: yaml
   :caption: example_config.yaml

Truth Catalog Example
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/configs/example_config_truth.yaml
   :language: yaml
   :caption: example_config_truth.yaml
