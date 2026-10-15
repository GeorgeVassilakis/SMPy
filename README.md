[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/bb1d59e8b4a143f8a261bc7320861495)](https://app.codacy.com/gh/GeorgeVassilakis/SMPy/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/GeorgeVassilakis/SMPy.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/GeorgeVassilakis/SMPy/stargazers/)

# SMPy (Shear Mapping in Python)

## Docs: 
The latest documentation is available on Read the Docs: [smpy-docs.readthedocs.io](https://smpy-docs.readthedocs.io/en/latest/overview.html).

## Overview
**SMPy (Shear Mapping in Python)** is a mass reconstruction toolkit for weak gravitational lensing analysis, primarily focused on mapping total matter distributions from galaxy shear data. The package implements three reconstruction methods, Kaiser-Squires (Kaiser & Squires 1993), KS+ (Pires et al. 2020) and aperture mass, to map the dimensionless surface mass density (convergence) field from weak lensing shear measurements. These non-parametric methods enable direct mapping of both dark and baryonic matter distributions from the observed distortions in background galaxy shapes.

Key attributes include:
- Mass reconstruction via Kaiser-Squires inversion in both celestial (RA/Dec) and pixel coordinate systems
- Aperture mass mapping for localized mass measurements
- E/B-mode decomposition for systematic error analysis
- Signal-to-noise ratio quantification through spatial and orientation randomization techniques
- Peak finding on signal-to-noise maps above a configurable threshold
- Intuitive and 'Pythonic' repository and code structure

The package provides a robust implementation supporting both weighted and unweighted shear catalogs, projection of RA/Dec catalogs onto the tangent plane at the field center with each shear rotated into the map frame, and binning of irregular galaxy distributions onto regular grids. This project aims to provide a robust, intuitive, and accessible way to create convergence maps from weak lensing data for astrophysicists of all levels.

## Features
### Convergence Mapping
- **Kaiser-Squires Inversion**: Implementation of the classic [Kaiser & Squires (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJ...404..441K/abstract) method for reconstructing convergence maps from weak lensing shear data
- **KS+**: Enhanced Kaiser-Squires implementation from [Pires et al. (2020)](https://www.aanda.org/articles/aa/abs/2020/06/aa36865-19/aa36865-19.html) that:
  - Corrects for missing data using DCT-domain sparsity priors
  - Reduces field border effects through automatic field extension
  - Iteratively corrects for the reduced shear approximation
  - Preserves statistical properties using wavelet-based power spectrum constraints
- **Aperture Mass Mapping**: A filtered measure of the projected mass around each pixel from the tangential shear (Schneider 1996)
- **Support for Both E-mode and B-mode**: Generate maps for both E-mode (physical) and B-mode (systematic check) signals
- **Flexible Coordinate Systems**: 
  - RA/Dec celestial coordinates, projected onto the tangent plane (gnomonic projection)
  - Pixel-based coordinates for direct image analysis
  - Automatic coordinate transformations and scaling

### Signal Processing & Error Analysis
- **Filtering**: Gaussian smoothing of the maps, with a configurable width
- **Signal-to-Noise Maps**: Generate SNR maps using two different randomization techniques:
  - Orientation shuffling (default): Randomizes galaxy orientations while preserving positions
  - Spatial shuffling: Randomizes galaxy positions while preserving shear values. Assumes the noise is uniform across the field
- **Peak Detection**: Automated identification of significant peaks in convergence maps with customizable detection thresholds

### Data Handling
- **FITS File Support**: Direct reading of astronomical FITS catalogs
- **Flexible Data Input**: Reads FITS catalogs or astropy Tables, with configurable column names
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
  - Command-line interface (`smpy -c config.yaml`)
  - Python API for notebook integration
- **Modular Design**: Extensible architecture supporting multiple mapping methods including Kaiser-Squires inversion, KS+, and aperture mass

## Installation

1. **Prerequisites**: Ensure you have Python 3.10+ installed on your system. SMPy also requires `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`, and `pyyaml` for numerical computations and visualizations.

2. **Clone the Repository**: Clone the SMPy repository to your local machine using git:

   ```bash
   git clone https://github.com/GeorgeVassilakis/SMPy.git
   ```

3. **Install the Package:** Install SMPy from the repository root:

   ```bash
   pip install .
   ```

## How to Run
### Tutorials
- The [tutorials](https://smpy-docs.readthedocs.io/en/latest/tutorials.html) are notebooks in `docs/source/tutorials`
  that run SMPy on a simulated cluster observation and its noiseless truth catalog.

### From the command line
1. **Prepare your configuration file**
   - Copy and modify the example configuration file from `examples/configs/example_config.yaml`
   - Set your input/output paths and data-specific parameters (coordinate system type, shear column names, etc.)
   - Set the method's settings, such as its smoothing, and the plot settings, such as color maps and titles

2. **Run SMPy:** Pass your .yaml file with the `-c` (or `--config`) flag
   
   ```bash
   smpy -c /path/to/example_config.yaml
   ```

### With Jupyter Notebook (Quickstart: 2 API Levels)
Use the same example catalog for both API levels:

```python
data_file = "examples/data/forecast_lum_annular.fits"
columns = dict(g1_col="g1_Rinv", g2_col="g2_Rinv", weight_col="weight")
```

1. **High-level functional API (quick)**: returns a `MassMap` and writes nothing to disk
   ```python
   from smpy import map_kaiser_squires, map_aperture_mass, map_ks_plus

   result_ks = map_kaiser_squires(data_file, pixel_scale=0.4, **columns)
   result_am = map_aperture_mass(data_file, pixel_scale=0.4, filter_type="schirmer", filter_scale=10,
                                 **columns)
   result_ksp = map_ks_plus(data_file, pixel_scale=0.4, inpainting_iterations=5,
                            reduced_shear_iterations=1, **columns)

   result_ks.maps["E"]              # E-mode convergence map as a NumPy array
   fig, ax, im = result_ks.plot("E")
   result_ks.to_fits("ks_e_mode.fits")
   ```

2. **Configuration-based API (recommended for reproducibility)**: `run` writes the outputs the configuration asks for
   ```python
   from smpy import Config, run

   config = Config.from_defaults("kaiser_squires", input_path=data_file, pixel_scale=0.4,
                                 mode=["E", "B"], **columns)
   config.save("my_run.yaml")      # rerun later with: smpy -c my_run.yaml
   result_cfg = run(config)
   ```

## Contributions
- Open an issue for bugs, unclear documentation, or feature requests.
- Open a discussion or issue before starting large changes.
- For private support, contact the maintainer at gv321 [at] cam [dot] ac [dot] uk.

## Testing

Run the test suite from the repository root:

```bash
python -m pytest
```

## Example Mass Maps
![Kaiser-Squires, KS+ and aperture-mass maps and a Kaiser-Squires signal-to-noise map of a simulated galaxy cluster](paper/mass_maps.png)
