Configuration Guide
===================

SMPy uses YAML configuration files to control all aspects of the mass mapping process.

Configuration Structure
-----------------------

The configuration file has four main sections:

.. code-block:: yaml

   general:      # Input/output paths, coordinate system, analysis settings
   methods:      # Method-specific parameters (kaiser_squires, aperture_mass, ks_plus)
   plotting:     # Visualization settings
   snr:          # Signal-to-noise ratio calculation settings

Loading Configuration
---------------------

Configuration files can be loaded using the Config class:

.. code-block:: python

   from smpy.config import Config
   
   # Load from file
   config = Config.from_file('my_config.yaml')
   
   # Load defaults for a specific method
   config = Config.from_defaults('kaiser_squires')

Parameter Reference
-------------------

.. note::
   The following tables describe all available configuration parameters.
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
     - Path to input FITS file containing shear catalog
   * - input_hdu
     - integer
     - 1
     - FITS extension number to read data from
   * - output_directory
     - string
     - "."
     - Directory to save output files
   * - output_base_name
     - string
     - "smpy_output"
     - Base name for output files
   * - coordinate_system*
     - string
     - "radec"
     - Coordinate system: 'radec' or 'pixel'
   * - method
     - string
     - "kaiser_squires"
     - Mass mapping method: 'kaiser_squires', 'aperture_mass', or 'ks_plus'
   * - g1_col
     - string
     - "g1"
     - Column name for first shear component
   * - g2_col
     - string
     - "g2"
     - Column name for second shear component
   * - weight_col
     - string/null
     - null
     - Column name for weights (null for unit weights)
   * - mode
     - list
     - ['E']
     - Modes to compute: ['E'], ['B'], or ['E', 'B']
   * - create_snr
     - boolean
     - false
     - Whether to create signal-to-noise ratio map
   * - save_fits
     - boolean
     - false
     - Whether to save maps as FITS files
   * - print_timing
     - boolean
     - false
     - Whether to print timing information

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
   * - resolution*
     - float
     - 0.4
     - Grid resolution in arcminutes (API: ``pixel_scale``)
   * - coord1
     - string
     - "ra"
     - RA column name in catalog
   * - coord2
     - string
     - "dec"
     - Dec column name in catalog

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
     - 1
     - Grid reduction factor (API: ``downsample_factor``)
   * - coord1
     - string
     - "X_IMAGE"
     - X coordinate column name
   * - coord2
     - string
     - "Y_IMAGE"
     - Y coordinate column name
   * - pixel_axis_reference
     - string
     - "catalog"
     - Which axes to use for pixel plots: 'catalog' (input-pixel coordinates) or 'map' (map-pixel indices)

Validation Behavior and Error Messages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Validation is strict only when ``general.input_path`` is set to a non-empty value. Loading defaults or inspecting configs without a real input path will not trigger hard errors.
- If ``coordinate_system='radec'`` and the required scale is missing, SMPy raises:

  ``Missing required parameter for coordinate_system='radec'. Provide 'pixel_scale' (API: pixel_scale=..., YAML: general.radec.resolution).``

- If ``coordinate_system='pixel'`` and the required factor is missing, SMPy raises:

  ``Missing required parameter for coordinate_system='pixel'. Provide 'downsample_factor' (API: downsample_factor=..., YAML: general.pixel.downsample_factor).``

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
   * - smoothing.type
     - string/null
     - "gaussian"
     - Smoothing type ('gaussian' or null)
   * - smoothing.sigma
     - float
     - 2.0
     - Gaussian smoothing scale in pixels

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
   * - filter.type*
     - string
     - "schirmer"
     - Filter type: 'schirmer' or 'schneider'
   * - filter.scale*
     - float
     - 60
     - Filter scale in pixels
   * - filter.truncation
     - float
     - 1.0
     - Truncation radius in units of scale
   * - filter.l
     - integer
     - 3
     - Polynomial order for Schneider filter

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
   * - inpainting_iterations
     - integer
     - 100
     - Number of iterations for inpainting algorithm
   * - reduced_shear_iterations
     - integer
     - 3
     - Number of iterations for reduced shear correction
   * - min_threshold_fraction
     - float
     - 0.0
     - Minimum threshold for DCT coefficients
   * - extension_size
     - string/int
     - "double"
     - Field extension: 'double' or number of pixels
   * - use_wavelet_constraints
     - boolean
     - true
     - Apply wavelet-based power spectrum constraints
   * - smoothing.type
     - string/null
     - "gaussian"
     - Smoothing type ('gaussian' or null)
   * - smoothing.sigma
     - float
     - 2.0
     - Gaussian smoothing scale in pixels

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
     - Center position: null, 'auto', or {ra_center: float, dec_center: float}

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
     - "spatial"
     - Randomization type: 'spatial' or 'orientation'
   * - num_shuffles
     - integer
     - 100
     - Number of random realizations
   * - seed
     - int/string
     - 0
     - Random seed (integer or 'random' for secure seed)
   * - smoothing.type
     - string/null
     - "gaussian"
     - Smoothing type for SNR maps
   * - smoothing.sigma
     - float
     - 2.0
     - Smoothing scale for SNR maps
   * - plot_title
     - string
     - "Signal-to-Noise Map"
     - Title for SNR plots

Example Configuration Files
---------------------------

Basic Kaiser-Squires Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../smpy/configs/example_config.yaml
   :language: yaml
   :caption: example_config.yaml

Full Featured Example
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../smpy/configs/example_config_truth.yaml
   :language: yaml
   :caption: example_config_truth.yaml