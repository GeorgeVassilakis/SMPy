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

A common tool for analyzing these distortions on large scales is convergence mapping. Convergence ($\kappa$) quantifies how much light from distant galaxies converge due to lensing, resulting in a magnification or distortion of their images. For a comprehensive review of weak gravitational lensing, please refer to [@Umetsu2020]. By mapping convergence across the sku, astronomers can identify areas with high mass concentration based on observed lensing data. Regions showing significant convergence but little visible matter likely indicate the presence of dark matter causing the lensing effect.

The **Shear Mapping in Python (SMPy)** package provides a standardized, well-documented, and open-source solution for creating convergence maps from weak lensing galaxy shear measurements. SMPy was initially developed to support the Superpressure Balloon-borne Imaging Telescope (SuperBIT), which completed its 45-night observing run in spring 2023 with over 30 galaxy cluster observations [@Gill2024]. SMPy has evolved into a general-purpose tool suitable for analyzing the weak lensing data from any source of galaxies.

# Statement of Need

Initial commit

# Acknowledgements

This material is based upon work supported by a Northeastern University Undergraduate Research and Fellowships PEAK Summit Award.