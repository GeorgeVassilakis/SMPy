[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bb1d59e8b4a143f8a261bc7320861495)](https://app.codacy.com/gh/GeorgeVassilakis/SMPy/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/GeorgeVassilakis/SMPy.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/GeorgeVassilakis/SMPy/stargazers/)

# SMPy (Shear Mapping in Python)

## Docs: 
The latest documentation is available on Read the Docs: [smpy-docs.readthedocs.io](https://smpy-docs.readthedocs.io/en/latest/overview.html) (work in progress).

## Overview
**SMPy (Shear Mapping in Python)** is a mass reconstruction toolkit for weak gravitational lensing analysis, primarily focused on mapping total matter distributions from galaxy shear data. The package implements the Kaiser-Squires inversion technique (Kaiser & Squires 1993) to reconstruct the dimensionless surface mass density (convergence) field from weak lensing shear measurements. This non-parametric reconstruction method enables direct mapping of both dark and baryonic matter distributions from the observed distortions in background galaxy shapes.

Key attributes include:
- Mass reconstruction via Kaiser-Squires inversion in both celestial (RA/Dec) and pixel coordinate systems
- Aperture mass mapping for localized mass measurements
- E/B-mode decomposition for systematic error analysis
- Signal-to-noise ratio quantification through spatial and orientation randomization techniques
- Peak statistics with customizable detection thresholds and significance estimation
- Intuitive and 'Pythonic' repository and code structure

The package provides a robust implementation supporting both weighted and unweighted shear catalogs, with built-in handling of spherical geometry for wide-field observations and flexible gridding schemes for irregular galaxy distributions. This project aims to provide a robust, intuitive, and accessible way to create convergence maps from weak lensing data for astrophysicists of all levels.

## Features
### Convergence Mapping
- **Kaiser-Squires Inversion**: Implementation of the classic [Kaiser & Squires (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJ...404..441K/abstract) method for reconstructing convergence maps from weak lensing shear data
- **KS+**: Enhanced Kaiser-Squires implementation from [Pires et al. (2020)](https://www.aanda.org/articles/aa/abs/2020/06/aa36865-19/aa36865-19.html) that:
  - Corrects for missing data using DCT-domain sparsity priors
  - Reduces field border effects through automatic field extension
  - Iteratively corrects for the reduced shear approximation
  - Preserves statistical properties using wavelet-based power spectrum constraints
- **Aperture Mass Mapping**: Direct measurement of the projected mass within apertures using tangential shear
- **Support for Both E-mode and B-mode**: Generate maps for both E-mode (physical) and B-mode (systematic check) signals
- **Flexible Coordinate Systems**: 
  - RA/Dec celestial coordinates with accurate spherical geometry handling
  - Pixel-based coordinates for direct image analysis
  - Automatic coordinate transformations and scaling

### Signal Processing & Error Analysis
- **Filtering**: Gaussian smoothing with configurable kernel sizes, with additional filters planned
- **Signal-to-Noise Maps**: Generate SNR maps using two different randomization techniques:
  - Spatial shuffling: Randomizes galaxy positions while preserving shear values
  - Orientation shuffling: Randomizes galaxy orientations while preserving positions
- **Peak Detection**: Automated identification of significant peaks in convergence maps with customizable detection thresholds

### Data Handling
- **FITS File Support**: Direct reading of astronomical FITS catalogs
- **Flexible Data Input**: Support for various column naming conventions and data formats
- **Optional Weighting**: Handle weighted and unweighted shear measurements
- **Automatic Grid Generation**: Smart binning of irregular galaxy distributions onto regular grids

### Visualization
- **Customizable Plotting**: 
  - Adjustable color schemes and scaling
  - Optional grid lines and coordinate labels
  - Automatic or manual axis labeling
  - Customizable figure sizes and titles
- **Peak Detection Overlay**: Optional automatic detection and marker overlay of peaks above a configurable threshold
- **Coordinate-aware plotting**: Coordinate-aware plotting for RA/Dec and pixel maps (axis orientation, extents, and tick labeling). WCS headers are included in FITS outputs when saving FITS.

### Configuration & Usability
- **YAML Configuration**: Easy-to-use YAML configuration files for full control over:
  - Input/output paths and formats
  - Mapping parameters and methods
  - Visualization settings
  - SNR calculation parameters
- **Multiple Interfaces**: 
  - Command-line interface using a runner script
  - Python API for notebook integration
- **Modular Design**: Extensible architecture supporting multiple mapping methods including Kaiser-Squires inversion, KS+, and aperture mass

## Installation

1. **Prerequisites**: Ensure you have Python 3.8+ installed on your system. SMPy also requires `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`, and `pyyaml` for numerical computations and visualizations.

2. **Clone the Repository**: Clone the SMPy repository to your local machine using git:

   ```bash
   git clone https://github.com/GeorgeVassilakis/SMPy.git
   ```

3. **Install the Package:** Install SMPy using setup.py:

   ```bash
   pip install .
   ```

## How to Run
### Examples
- Pedagogical explinations are shown in the `SMPy/examples/notebooks` directory.
  - The two notebooks run through the SMPy algorithm on mock observations, along with it's corresponding truth file as a unit test that the algorithm correctly recovers a gaussian shear.

### With runner script
1. **Prepare your configuration file**
   - Copy and modify the example configuration file from `smpy/configs/example_config.yaml`
   - Set your input/output paths and data-specific parameters (coordinate system type, shear column names, etc.)
   - Configure visualization settings like smoothing, color maps, and plot titles

2. **Run the runner.py script:** Use the `-c` or `-config` flag to pass your .yaml file
   
   ```bash
   python runner.py -c /path/to/example_config.yaml
   ```

### With Jupyter Notebook (Quickstart: 2 API Levels)
Use the same example catalog for both API levels:

```python
data_file = "examples/data/forecast_lum_annular.fits"
common = dict(
    data=data_file,
    coord_system="radec",
    pixel_scale=0.4,
    g1_col="g1_Rinv",
    g2_col="g2_Rinv",
    weight_col="weight",
    save_plots=False,
)
```

1. **High-level functional API (quick)**
   ```python
   from smpy import map_kaiser_squires, map_aperture_mass, map_ks_plus

   result_ks = map_kaiser_squires(**common)
   result_am = map_aperture_mass(**common, filter_type="schirmer", filter_scale=60)
   result_ksp = map_ks_plus(**common, inpainting_iterations=5, reduced_shear_iterations=1)
   ```

2. **Configuration-based API (recommended for reproducibility)**
   ```python
   from smpy import run
   from smpy.config import Config

   config = Config.from_defaults("kaiser_squires")
   config.update_from_kwargs(**common, smoothing=2.0, mode=["E", "B"])
   result_cfg = run(config)
   ```

## Contributions
- `SMPy` is built in the spirit of open source, so feel free to fork the repository and create a pull request to contribute! Any help is appreciated :)
- If there are issues or bugs in the software, feel free to raise an issue in GitHub's issues tab or create a GitHub discussion, and request @GeorgeVassilakis for review.
- If you need support or help using `SMPy`, feel free to contact me via my email: gv321 [at] cam [dot] ac [dot] uk

## Example Kaiser Squires Convergence Map
![Kaiser Squires Convergence Map](examples/outputs/kaiser_squires/simulation_testing_kaiser_squires_e_mode.png)
