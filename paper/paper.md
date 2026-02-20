---
title: 'Shear Mapping in Python (SMPy): Modular, Extensible, and Accessible Dark Matter Mapping'
tags:
  - Python
  - astrophysics
  - gravitational lensing
  - dark matter
  - galaxy shear
  - convergence mapping
authors:
  - name: Georgios N. Vassilakis
    orcid: 0009-0006-2684-2961
    corresponding: true
    affiliation: "1, 2, 3, 4"
  - name: Jacqueline E. McCleary
    orcid: 0000-0002-9883-7460
    corresponding: false
    affiliation: 1
  - name: Maya Amit
    orcid: 0009-0002-2898-7022
    corresponding: false
    affiliation: 1
  - name: Sayan Saha
    orcid: 0000-0002-6044-2164
    corresponding: false
    affiliation: 1
affiliations:
 - name: Department of Physics, Northeastern University, Boston, MA, USA
   index: 1
 - name: Institute of Astronomy, University of Cambridge
   index: 2
 - name: Department of Applied Mathematics and Theoretical Physics, University of Cambridge
   index: 3
 - name: Kavli Institute for Cosmology, University of Cambridge
   index: 4
date: 20 April 2025
bibliography: paper.bib
---

# Summary

Understanding the universe's large-scale distribution of dark matter is a central objective in the era of precision cosmology. A key technique for the study of dark matter is weak gravitational lensing: a phenomenon where light from distant galaxies is sheared as it passes through the gravitational field of a massive object, like a galaxy cluster. This shear, which manifests as a slight (weak) distortion of shapes over thousands of galaxies, allows astrophysicists to infer the distribution of total matter, including both luminous and dark matter.

Obtaining a mass distribution from a catalog of galaxy shears requires an intermediate step. A common tool for this step is the mapping of convergence ($\kappa$), which quantifies how much a gravitational lens converges the light from distant galaxies, resulting in a magnification of their shapes. This value is directly proportional to the projected mass density, enabling easy visualization of the overall mass distribution. For a comprehensive review of weak gravitational lensing, refer to [@Umetsu2020]. 

The **Shear Mapping in Python (SMPy)** package provides a standardized, well-documented, and open-source solution for creating convergence maps from weak lensing galaxy shear measurements. `SMPy` was initially developed to support the Superpressure Balloon-borne Imaging Telescope (SuperBIT), a stratospheric, near-UV to near-IR observing platform which completed its 45-night observing run in spring 2023 with over 30 galaxy cluster observations [@Gill2024; @Sirks2023]. `SMPy` has since evolved into a general-purpose tool suitable for analyzing the weak lensing data from cosmological surveys.

# Statement of need

While mass maps are a key deliverable of many cosmological analyses [@ACTDR62024; @DESY32021; @HSC2017], scientists are often left to make these maps from scratch.

`SMPy` addresses an outstanding need for the lensing community: an accessible, well-documented, and extensible tool to construct publication-quality mass maps from galaxy shear data. Built on standard scientific Python packages, it provides an easy entry point for researchers new to mass mapping, while also being robust for more senior scientific use. There are currently three separate mapping methods implemented into `SMPy`: the classic Kaiser-Squires inversion [@KS1993], aperture mass mapping [@Leonard2012; @McCleary2020], and notably, to our knowledge, the first publicly available implementation of the `KS+` algorithm [@Pires2020]. `KS+` improves reconstruction quality by correcting for systematic effects including missing data, field borders, and reduced shear. `SMPy` also offers specialized and unique features valuable for mass mapping, such as flexible coordinate system support (both celestial and pixel space) and comprehensive signal-to-noise analysis with multiple noise randomization techniques. An example convergence map, created from simulated SuperBIT galaxy cluster observations [@McCleary2023], is shown in Figure \ref{fig:convergence_map}. `SMPy` is, to our knowledge, the first convergence mapping software to prioritize both accessibility and advanced features.

# State of the field

The weak lensing community is served by publicly available mapping tools like `lenspack` and `jax-lensing` [@Remy2022], each with their own strengths. `jax-lensing` excels at neural network-based approaches and deep learning methods, but applying it to survey-specific data products requires substantial additional development by the user. `lenspack` provides well-documented, stand-alone mass-mapping functions, but does not provide an end-to-end convergence-mapping workflow (e.g., configuration management, coordinate handling, noise/SNR analysis, and plotting). In practice, many mass-mapping analyses still rely on bespoke codes built for a specific survey or science case. These gaps, combined with the need for a flexible, configuration-driven, and modular mapping framework, motivated the development of `SMPy` as an accessible and extensible alternative.

# Software design

`SMPy` was built with the following design principles in mind:

1. **Accessibility:** `SMPy` is written entirely in Python and deliberately relies only on widely-used scientific Python packages (`NumPy`, `SciPy`, `Pandas`, `Astropy`, `Matplotlib`, and `PyYAML`). This choice of standard dependencies ensures that users can easily install the packages without complex dependency chains, and that the codebase is maintainable and familiar to the scientific Python community.

2. **Extensibility:** `SMPy`'s modular architecture enables seamless addition of new mass mapping techniques, encouraging open-source contribution. The framework's standardized abstract base class architecture allows for the integration of different mapping methods with minimal effort.

3. **Usability:** Creating convergence maps with `SMPy` requires minimal input—users need to only provide a catalog of galaxies with their associated shear components and coordinates. This straightforward input requirement makes the tool accessible to researchers at all levels. A flexible configuration system is integrated via a single YAML file that defines file paths, convergence map algorithm settings, plotting parameters, and more. With this configuration file, the user can create convergence and SNR maps with one line, either via terminal or within code.

4. **Robustness:** `SMPy` is designed to be mathematically and algorithmically accurate, allowing the user to create convergence maps with any galaxy shear data. The coordinate system abstraction handles both celestial coordinates (with proper spherical geometry approximations) or pixel-based coordinates through a unified interface. To quantify the significance of the weak lensing detection, multiple noise realizations can be generated using either spatial shuffling (randomizing galaxy positions while preserving shear values) or orientation shuffling (randomizing shear orientations while preserving positions). These noise realizations are used to create a signal-to-noise map with the observed convergence.

These design decisions are reflected in `SMPy`'s overall workflow: users provide a shear catalog and a single YAML configuration file, and `SMPy` produces convergence maps, signal-to-noise maps, and publication-quality plots with minimal boilerplate. The mapping framework is implemented through standardized abstract base classes, allowing different reconstruction methods to be interchanged while preserving a consistent interface and coordinate-system handling.

![Example convergence map created with SMPy showing the mass distribution of a simulated galaxy cluster. The map was generated using the Kaiser-Squires inversion method on simulated weak lensing data from SuperBIT. The color scale represents the dimensionless surface mass density (convergence), with brighter regions indicating higher mass concentrations.](KS_convergence_map.png){#fig:convergence_map}

# Research impact statement

`SMPy` was initially developed as part of the weak lensing analysis pipeline for SuperBIT, and it is currently being used to create convergence maps from SuperBIT's 2023 science flight imaging. These convergence maps will be presented in a forthcoming SuperBIT weak-lensing analysis (Saha et al., in preparation, 2026; "Lensing in the Blue IV"). In addition, `SMPy` has been presented at the Jet Propulsion Laboratory, and we have received interest in exploring integration into Euclid weak-lensing analysis workflows.

# AI usage disclosure

Generative AI was used in three ways in this work. First, Claude Sonnet 4 (via the Claude.ai web interface) was used to assist in implementing the `KS+` algorithm in `SMPy`. The author reviewed all AI-assisted code line-by-line against the Algorithm 1 description in Appendix A of [@Pires2020] and validated the implementation by injecting fiducial star masks into data and confirming that the reconstruction behaved as expected. Second, OpenAI's ChatGPT (5.0 and 5.2) models have been used for generating unit tests for the codebase. Lastly, OpenAI's ChatGPT models (5.2) in the Codex harness have been used to review pull requests and catch bugs.

## Software references

`SMPy` is written in Python 3.8+ and uses the following packages:

- `NumPy` [@numpy]
- `SciPy` [@scipy]
- `Pandas` [@pandas]
- `Astropy` [@astropyv3; @astropyv2; @astropyv1]
- `Matplotlib` [@matplotlib]
- `PyYAML` [@pyyaml]

# Acknowledgements

This material is based upon work supported by a Northeastern University Undergraduate Research and Fellowships PEAK Summit Award.

# References
