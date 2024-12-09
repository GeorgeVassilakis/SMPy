---
title: 'Shear Mapping in Python (SMPy): Modular, Extensible, and Accessible Dark Matter Mapping'
tags:
  - Python
  - astronomy
  - gravitational lensing
  - dark matter
  - galaxy shear
  - convergence mapping
authors:
  - name: Georgios N. Vassilakis
    orcid: 0009-0006-2684-2961
    corresponding: true
    affiliation: 1
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
 - name: Department of Physics, Northeastern University, USA
   index: 1
date: 7 December 2024
bibliography: paper.bib
---

# Summary

Understanding the universe's structure, particularly the nature of dark matter, is a central challenge in modern astrophysics. A key approach to this problem is the study of weak gravitational lensing, where light from distant galaxies is bent as it passes though the gravitational field of a massive object, like a galaxy cluster. Measuring this slight (weak) bending of light over thousands of galaxies allows astrophysicists to infer the distribution of matter, including dark matter. 

A common tool for analyzing these distortions on large scales is convergence mapping. Convergence ($\kappa$) quantifies how much light from distant galaxies converge due to lensing, resulting in a magnification or distortion of their images. For a comprehensive review of weak gravitational lensing, please refer to [@Umetsu2020]. By mapping convergence across the sky, astronomers can identify areas with high mass concentration based on observed lensing data. Regions showing significant convergence but little visible matter likely indicate the presence of dark matter causing the lensing effect.

The **Shear Mapping in Python (SMPy)** package provides a standardized, well-documented, and open-source solution for creating convergence maps from weak lensing galaxy shear measurements. `SMPy` was initially developed to support the Superpressure Balloon-borne Imaging Telescope (SuperBIT), which completed its 45-night observing run in spring 2023 with over 30 galaxy cluster observations [@Gill2024]. `SMPy` has evolved into a general-purpose tool suitable for analyzing the weak lensing data from any source of galaxies.

# Statement of Need

Mass maps are a critical and key part of many cosmological analyses [@Atacama2020] [@DESY32021] [@HSC2017]. `SMPy` addresses an outstanding need for the lensing community: A robust, well-documented, and open-source tool to construct publication quality mass maps from galaxy shear data. `SMPy` was built with multiple design directions in mind:

1. **Accessibility:** `SMPy` is written entirely in Python and deliberately relies only on widely-used scientific Python packages (`numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`, and `pyyaml`). This choice of standard dependencies ensures that users can easily install the packages without complex dependency chains, and that the codebase is maintainable and familiar to the scientific Python community.

2. **Extensibility:** `SMPy` is built with a modular architecture that allows for easy implementation of new mass mapping techniques beyond the currently implemented Kaiser-Squires inversion algorithm [@KS1993]. An example convergence map is shown in Figure 1, created from simulated galaxy cluster observations from SuperBIT [@McCleary2023]. Aperture mass mapping [@Leonard2012] and KS+ [@Pires2020] algorithms are currently planned to be added to the codebase. 

3. **Usability:** Creating convergence maps with `SMPy` requires minimal input - users need only provide a catalog of galaxies with their associated shears (`g1` & `g2`) and coordinates. This straightforward input requirement makes the tool accessible to researchers at all levels.  A flexible configuration system is integrated via a single YAML file that defines file paths, convergence map algorithm settings, plotting parameters, and more.

4. **Robustness:** Designed to be mathematically and algorithmically accurate, allowing the user to create convergence maps with any galaxy shear data. The coordinate system abstraction handles both RA/Dec celestial coordinates (with proper spherical geometry approximations) or pixel-based coordinates through a unified interface. Signal-to-noise maps can be generated using either spatial shuffling (randomizing galaxy positions while preserving shear values) or orientation shuffling (randomizing shear orientations while preserving positions) to distinguish real signals from noise.

![Example convergence map created with SMPy showing the mass distribution of a simulated galaxy cluster. The map was generated using the Kaiser-Squires inversion method on simulated weak lensing data from SuperBIT. The color scale represents the dimensionless surface mass density (convergence), with brighter regions indicating higher mass concentrations.](KS_convergence_map.png)

To our knowledge, `SMPy` is the first open-source, well-documented convergence mapping software that can compute convergence in either astrometric or pixel space. [NEEDS EDIT/EXPANSION]

# Software References

`SMPy` requires and uses the following packages:
- `NumPy`
- `SciPy`
- `Pandas`
- `Astropy`
- `Matplotlib`
- `PyYAML`


# Acknowledgements

This material is based upon work supported by a Northeastern University Undergraduate Research and Fellowships PEAK Summit Award.