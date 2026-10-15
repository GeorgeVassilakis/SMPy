"""Utility functions for SMPy mass mapping operations.

This module provides catalog loading, peak detection and tangential shear
utilities used throughout the SMPy package.
"""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table

from smpy.coordinates.radec import unproject

def load_shear_data(catalog, coord1_col, coord2_col, g1_col, g2_col, weight_col=None, hdu=0,
                    coord_unit=None):
    """Load a shear catalog with standardized column names.

    Return the catalog as a pandas DataFrame with standardized column names
    for use in mass mapping operations.

    Parameters
    ----------
    catalog : `str`, `pathlib.Path`, or `astropy.table.Table`
        Catalog table, or the path to a FITS file.
    coord1_col : `str`
        Name of first coordinate column (e.g., 'ra' or 'x').
    coord2_col : `str`
        Name of second coordinate column (e.g., 'dec' or 'y').
    g1_col : `str`
        Name of g1 shear component column.
    g2_col : `str`
        Name of g2 shear component column.
    weight_col : `str`, optional
        Name of weight column. If None, unit weights are assigned.
    hdu : `int`, optional
        HDU number to read when ``catalog`` is a FITS file path.
    coord_unit : `str`, optional
        Unit of the coordinates, such as 'deg'. Coordinate columns that
        declare a unit are converted to it, and columns without one are taken
        to be in it. If None, coordinates are used as they are.

    Returns
    -------
    shear_data : `pandas.DataFrame`
        DataFrame with standardized column names: 'coord1', 'coord2', 'g1',
        'g2', 'weight'.

    Raises
    ------
    IndexError
        If specified HDU is not found in the FITS file.
    KeyError
        If required columns are not found in the catalog.
    ValueError
        If a coordinate column declares a unit that cannot be converted to
        ``coord_unit``, or a used column has NaN, infinite or masked values.
    """
    if isinstance(catalog, Table):
        shear_catalog = catalog
    else:
        try:
            shear_catalog = Table.read(catalog, hdu=hdu)
        except IndexError:
            raise IndexError(f"HDU {hdu} not found in {catalog}")

    # Convert to pandas DataFrame with generic column names
    try:
        shear_df = pd.DataFrame({
            'coord1': shear_catalog[coord1_col],
            'coord2': shear_catalog[coord2_col],
            'g1': shear_catalog[g1_col],
            'g2': shear_catalog[g2_col],
        })
    except KeyError as e:
        raise KeyError(f"Column {e} not found in HDU {hdu}. Available columns: {shear_catalog.colnames}")

    # Add weights (unit weights if not specified)
    if weight_col is None:
        shear_df['weight'] = np.ones(len(shear_df))
    else:
        shear_df['weight'] = shear_catalog[weight_col]

    names = dict(coord1=coord1_col, coord2=coord2_col, g1=g1_col, g2=g2_col, weight=weight_col)
    for key, name in names.items():
        bad = np.count_nonzero(~np.isfinite(shear_df[key]))
        if bad:
            raise ValueError(f"Column '{name}' has {bad} NaN, infinite or masked values")

    if coord_unit is not None:
        for key in ('coord1', 'coord2'):
            unit = shear_catalog[names[key]].unit
            if unit is None:
                continue
            if not unit.is_equivalent(coord_unit):
                raise ValueError(f"Column '{names[key]}' has unit '{unit}', which is not convertible to "
                                 f"{coord_unit}; give it a standard unit, or none for {coord_unit}")
            shear_df[key] = u.Quantity(shear_df[key].to_numpy(), unit).to_value(coord_unit)

    return shear_df

def find_peaks2d(image, threshold=None, verbose=False, true_boundaries=None, scaled_boundaries=None):
    """Identify peaks in a 2D array above a specified threshold.

    Find local maxima in a 2D image where each peak pixel has a value
    greater than all 8 neighboring pixels. Refactored from cosmostat/lenspack.

    Parameters
    ----------
    image : `numpy.ndarray`
        2D input map for peak detection.
    threshold : `float`, optional
        Detection threshold. If None, uses minimum image value.
    verbose : `bool`, optional
        Whether to print peak information.
    true_boundaries : `dict`, optional
        True coordinate boundaries for position conversion.
    scaled_boundaries : `dict`, optional
        Scaled coordinate boundaries for position conversion.

    Returns
    -------
    X : `numpy.ndarray`
        Peak pixel indices in x-direction.
    Y : `numpy.ndarray`
        Peak pixel indices in y-direction.
    heights : `numpy.ndarray`
        Peak values sorted in descending order.
    coords : `list` or None
        Peak coordinates in true coordinate system if boundaries provided,
        otherwise None.
    """
    image = np.atleast_2d(image)
    threshold = threshold if threshold is not None else image.min()

    # Pad the image for border peak checks
    padded_image = np.pad(image, pad_width=1, mode='constant',
                         constant_values=image.min())

    # Check for peaks by comparing to neighbors
    is_peak = (
        (padded_image[1:-1, 1:-1] > padded_image[:-2, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[:-2, 1:-1]) &
        (padded_image[1:-1, 1:-1] > padded_image[:-2, 2:]) &
        (padded_image[1:-1, 1:-1] > padded_image[1:-1, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[1:-1, 2:]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, :-2]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, 1:-1]) &
        (padded_image[1:-1, 1:-1] > padded_image[2:, 2:])
    )

    # Apply threshold
    peaks_mask = is_peak & (image >= threshold)

    # Get peak coordinates and heights
    Y, X = np.nonzero(peaks_mask)
    heights = image[Y, X]

    # Sort peaks by height in descending order
    sort_indices = np.argsort(-heights)
    X = X[sort_indices]
    Y = Y[sort_indices]
    heights = heights[sort_indices]

    # Convert pixel coordinates to true coordinates if boundaries are provided
    coords = None
    if verbose and true_boundaries and scaled_boundaries:
        coords = []
        coord_system = 'pixel' if 'X' in true_boundaries['coord1_name'] else 'radec'

        for x, y in zip(X, Y):
            # Convert pixel indices to scaled coordinates
            scaled_coord1 = scaled_boundaries['coord1_min'] + (x + 0.5) * (
                scaled_boundaries['coord1_max'] - scaled_boundaries['coord1_min']
            ) / image.shape[1]

            scaled_coord2 = scaled_boundaries['coord2_min'] + (y + 0.5) * (
                scaled_boundaries['coord2_max'] - scaled_boundaries['coord2_min']
            ) / image.shape[0]

            # RA/Dec maps lie on the tangent plane; pixel coordinates are not
            # transformed
            if coord_system == 'radec':
                coords.append(unproject(scaled_coord1, scaled_coord2, true_boundaries['coord1_center'],
                                        true_boundaries['coord2_center']))
            else:
                coords.append((scaled_coord1, scaled_coord2))

        # Print peak information with appropriate coordinate labels
        if coord_system == 'radec':
            coord1_label, coord2_label = 'RA', 'Dec'
        else:
            coord1_label, coord2_label = 'X', 'Y'

        print("\nDetected Peaks:")
        print("-" * 60)
        print(f"{'Peak #':<8}{'Value':<12}{coord1_label:<12}{coord2_label:<12}")
        print("-" * 60)
        for i, ((c1, c2), height) in enumerate(zip(coords, heights), 1):
            print(f"{i:<8}{height:.<12.5f}{c1:.<12.5f}{c2:.<12.5f}")
        print("-" * 60)

    return X, Y, heights, coords

def g1g2_to_gt_gc(g1, g2, coord1, coord2, center_coord1, center_coord2, pix_coord1=100):
    """Convert shear components to tangential/cross components.

    Transform Cartesian shear components (g1, g2) to tangential and cross
    shear components relative to a specified center position.

    Parameters
    ----------
    g1 : `numpy.ndarray`
        First shear component.
    g2 : `numpy.ndarray`
        Second shear component.
    coord1 : `numpy.ndarray`
        First coordinates (RA or X).
    coord2 : `numpy.ndarray`
        Second coordinates (Dec or Y).
    center_coord1 : `float`
        Center position in first coordinate.
    center_coord2 : `float`
        Center position in second coordinate.
    pix_coord1 : `int`, optional
        Grid size in first dimension.

    Returns
    -------
    gt : `numpy.ndarray`
        Tangential shear component.
    gc : `numpy.ndarray`
        Cross shear component.
    phi : `numpy.ndarray`
        Polar angle relative to center.
    """
    coord1_max = np.max(coord1)
    coord1_min = np.min(coord1)
    coord2_max = np.max(coord2)
    coord2_min = np.min(coord2)

    aspect_ratio = (coord1_max - coord1_min) / (coord2_max - coord2_min)
    pix_coord2 = int(pix_coord1 / aspect_ratio)

    # Create coordinate grid
    coord1_grid, coord2_grid = np.meshgrid(
        np.linspace(coord1_min, coord1_max, pix_coord1),
        np.linspace(coord2_min, coord2_max, pix_coord2)
    )

    # Calculate polar angle
    phi = np.arctan2(coord2_grid - center_coord2,
                     coord1_grid - center_coord1)

    # Calculate tangential and cross components
    gt = -g1 * np.cos(2 * phi) - g2 * np.sin(2 * phi)
    gc = -g1 * np.sin(2 * phi) + g2 * np.cos(2 * phi)

    return gt, gc, phi
