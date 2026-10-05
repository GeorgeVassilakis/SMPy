"""Plotting functions for SMPy mass and SNR maps.

Provides high-level plotting for pixel and RA/Dec coordinate systems
with consistent styling, scaling and overlays. Helpers live in
``smpy.plotting.utils`` to keep this module focused on public plotting
APIs and orchestration.
"""
# Standard Library
from __future__ import annotations

import matplotlib.patheffects as patheffects

# Third Party
import matplotlib.pyplot as plt

from smpy.coordinates.radec import project
from smpy.plotting.utils import (
    add_colorbar,
    apply_axes_style,
    compute_pixel_extent,
    configure_labels,
    convert_center_to_scaled,
    create_normalization,
    overlay_xray_contours,
    peaks_to_plot_coords,
    propose_ticks,
    set_ticks,
)

# Local
from smpy.utils import find_peaks2d

def plot_mass_map(data, scaled_boundaries, true_boundaries, config, ax=None,
                  map_category="convergence", counts_overlay=None):
    """Plot a mass, SNR or counts map with styling and overlays.

    Parameters
    ----------
    data : `numpy.ndarray`
        2D convergence map data (E or B mode).
    scaled_boundaries : `dict`
        Scaled coordinate boundaries for plotting extent.
    true_boundaries : `dict`
        True coordinate boundaries for tick labels.
    config : `dict`
        Plot configuration settings including figsize, cmap, scaling,
        'coordinate_system', optional 'axis_reference' (pixel only), and
        optional x-ray contour settings under ``xray_contours``.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on. If `None`, a new figure is created.
    map_category : `str`, optional
        Map category used for scaling and overlays. Options: 'convergence',
        'snr', 'counts'.
    counts_overlay : `numpy.ndarray`, optional
        If provided, overlays integer per-pixel counts (using existing counts
        labeling logic) on top of the rendered image.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot. It is neither shown nor closed.
    ax : `matplotlib.axes.Axes`
        Axes the map was drawn on.
    im : `matplotlib.image.AxesImage`
        The map image.
    """
    # Dispatch to coordinate-system specific renderer based on config
    coord_system = config.get("coordinate_system", "radec").lower()
    if coord_system == "radec":
        return _plot_radec(data, scaled_boundaries, true_boundaries, config, ax, map_category,
                           counts_overlay)
    return _plot_pixel(data, scaled_boundaries, true_boundaries, config, ax, map_category, counts_overlay)

def _plot_pixel(data, scaled_boundaries, true_boundaries, config, ax, map_category, counts_overlay):
    """Render pixel-coordinate plot with overlays and colorbar."""
    # Create figure/axes if needed and apply local styling (no global rc
    # changes)
    new_figure = ax is None
    if new_figure:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=config.get("figsize", (12, 8)))
    else:
        fig = ax.figure
    fontsize = int(config.get("fontsize", 15))
    apply_axes_style(ax, fontsize=fontsize)

    # Choose how axes are labeled in pixel mode
    axis_reference = str(config.get("axis_reference", "catalog")).lower()

    # Build colormap normalization from config (percentiles, power, symlog)
    norm = create_normalization(config.get("scaling"), data, vmin=config.get("vmin"),
                                vmax=config.get("vmax"), map_type=map_category)

    # Determine image extent from chosen axis reference
    extent = compute_pixel_extent(data, scaled_boundaries, axis_reference)
    im = ax.imshow(data, cmap=config.get("cmap", "viridis"), norm=norm, extent=extent, origin="lower")

    # Optional: mark cluster center; convert to map pixels if needed
    cx, cy = convert_center_to_scaled(config.get("cluster_center"), scaled_boundaries, true_boundaries,
                                      coord_system_type="pixel")
    if cx is not None:
        if axis_reference == "map":
            # Convert from catalog coordinates to map pixel indices
            height, width = data.shape
            x_min = scaled_boundaries["coord1_min"]
            x_max = scaled_boundaries["coord1_max"]
            y_min = scaled_boundaries["coord2_min"]
            y_max = scaled_boundaries["coord2_max"]
            cx = (cx - x_min) / (x_max - x_min) * width
            cy = (cy - y_min) / (y_max - y_min) * height
        ax.plot(cx, cy, "rx", markersize=10)

    # Optional: overlay peak markers above threshold (disabled for counts maps)
    threshold = config.get("threshold")
    if (threshold is not None) and (str(map_category).lower() != "counts"):
        verbose_peaks = bool(config.get("verbose", False))
        # Detect peaks using 2D local maxima algorithm
        X, Y, _, _ = find_peaks2d(data, threshold=threshold, verbose=verbose_peaks,
                                  true_boundaries=true_boundaries, scaled_boundaries=scaled_boundaries)
        # Convert peak indices to appropriate plotting coordinates
        px, py = peaks_to_plot_coords(X, Y, data, scaled_boundaries, axis_reference)
        ax.scatter(px, py, s=100, facecolors="none", edgecolors="g", linewidth=1.5)

    # Optional: overlay DS9 x-ray contours for convergence and/or SNR maps
    overlay_xray_contours(
        ax=ax,
        data_shape=data.shape,
        scaled_boundaries=scaled_boundaries,
        true_boundaries=true_boundaries,
        config=config,
        map_category=map_category,
        coord_system_type="pixel",
        axis_reference=axis_reference,
    )

    # Overlay integer count labels at pixel centers (for counts map or overlay
    # mode)
    overlay_mode = str(map_category).lower() == "counts"
    overlay_data = data if overlay_mode else counts_overlay
    if overlay_data is not None and (overlay_mode or counts_overlay is not None):
        _overlay_counts_text_pixel(ax, overlay_data, scaled_boundaries, axis_reference, fontsize)

    # Labels, title, optional grid
    configure_labels(ax, config, axis_reference=axis_reference, coord_system_type="pixel", fontsize=fontsize)
    if config.get("gridlines", False):
        ax.grid(color="black")

    # Attach colorbar to the right
    add_colorbar(ax, im, tick_fontsize=fontsize)

    if new_figure:
        fig.tight_layout()
    return fig, ax, im

def _overlay_counts_text_pixel(ax, data, scaled_boundaries, axis_reference, base_fontsize):
    """Draw count labels at pixel centers for pixel-coordinate plots."""
    height, width = data.shape
    axis_reference = str(axis_reference or "catalog").lower()
    # Compute x, y centers in plotting coordinates based on axis reference
    if axis_reference == "map":
        x_centers = [j + 0.5 for j in range(width)]
        y_centers = [i + 0.5 for i in range(height)]
    else:
        x_min = scaled_boundaries["coord1_min"]
        x_max = scaled_boundaries["coord1_max"]
        y_min = scaled_boundaries["coord2_min"]
        y_max = scaled_boundaries["coord2_max"]
        x_centers = [x_min + (j + 0.5) * (x_max - x_min) / width for j in range(width)]
        y_centers = [y_min + (i + 0.5) * (y_max - y_min) / height for i in range(height)]

    count_fontsize = max(6, int(base_fontsize * 0.6))
    outline = [patheffects.withStroke(linewidth=1.8, foreground="black")]
    for i in range(height):
        for j in range(width):
            val = data[i, j]
            label = f"{int(round(val))}"
            ax.text(
                x_centers[j],
                y_centers[i],
                label,
                color="white",
                ha="center",
                va="center",
                fontsize=count_fontsize,
                path_effects=outline,
            )

def _plot_radec(data, scaled_boundaries, true_boundaries, config, ax, map_category, counts_overlay):
    """Render RA/Dec plot with astronomical orientation and ticks."""
    # Create figure/axes if needed and apply local styling (no global rc
    # changes)
    new_figure = ax is None
    if new_figure:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=config.get("figsize", (12, 8)))
    else:
        fig = ax.figure
    fontsize = int(config.get("fontsize", 15))
    apply_axes_style(ax, fontsize=fontsize)

    # Build colormap normalization from config
    norm = create_normalization(config.get("scaling"), data, vmin=config.get("vmin"),
                                vmax=config.get("vmax"), map_type=map_category)

    # Draw image using scaled RA/Dec extents
    im = ax.imshow(
        data,
        cmap=config.get("cmap", "viridis"),
        norm=norm,
        extent=[
            scaled_boundaries["coord1_min"],
            scaled_boundaries["coord1_max"],
            scaled_boundaries["coord2_min"],
            scaled_boundaries["coord2_max"],
        ],
        origin="lower",
    )

    # Optional: overlay peak markers above threshold (disabled for counts maps)
    threshold = config.get("threshold")
    if (threshold is not None) and (str(map_category).lower() != "counts"):
        verbose_peaks = bool(config.get("verbose", False))
        # Detect peaks using 2D local maxima algorithm
        X, Y, _, _ = find_peaks2d(data, threshold=threshold, verbose=verbose_peaks,
                                  true_boundaries=true_boundaries, scaled_boundaries=scaled_boundaries)
        # Convert peak pixel indices to RA/Dec coordinates
        ra_peaks = [
            scaled_boundaries["coord1_min"]
            + (x + 0.5) * (scaled_boundaries["coord1_max"] - scaled_boundaries["coord1_min"]) / data.shape[1]
            for x in X
        ]
        dec_peaks = [
            scaled_boundaries["coord2_min"]
            + (y + 0.5) * (scaled_boundaries["coord2_max"] - scaled_boundaries["coord2_min"]) / data.shape[0]
            for y in Y
        ]
        ax.scatter(ra_peaks, dec_peaks, s=100, facecolors="none", edgecolors="g", linewidth=1.5)

    # Optional: overlay DS9 x-ray contours for convergence and/or SNR maps
    overlay_xray_contours(
        ax=ax,
        data_shape=data.shape,
        scaled_boundaries=scaled_boundaries,
        true_boundaries=true_boundaries,
        config=config,
        map_category=map_category,
        coord_system_type="radec",
    )

    # Overlay integer count labels at pixel centers (for counts map or overlay
    # mode)
    overlay_mode = str(map_category).lower() == "counts"
    overlay_data = data if overlay_mode else counts_overlay
    if overlay_data is not None and (overlay_mode or counts_overlay is not None):
        _overlay_counts_text_radec(ax, overlay_data, scaled_boundaries, fontsize)

    # Optional: mark cluster center in RA/Dec coordinates
    ra_center, dec_center = convert_center_to_scaled(config.get("cluster_center"), scaled_boundaries,
                                                     true_boundaries, coord_system_type="radec")
    if ra_center is not None:
        ax.plot(ra_center, dec_center, "rx", markersize=10)

    # Ticks mark where RA crosses the central Dec and Dec crosses the central
    # RA
    ra_center, dec_center = true_boundaries["coord1_center"], true_boundaries["coord2_center"]
    ra_ticks, _ = propose_ticks(true_boundaries["coord1_min"], true_boundaries["coord1_max"], 5)
    dec_ticks, dec_labels = propose_ticks(true_boundaries["coord2_min"], true_boundaries["coord2_max"], 5)
    x_ticks, _ = project(ra_ticks, dec_center, ra_center, dec_center)
    _, y_ticks = project(ra_center, dec_ticks, ra_center, dec_center)
    set_ticks(ax, x_ticks, y_ticks, [f"{tick % 360:.2f}" for tick in ra_ticks], dec_labels)

    # Labels, title, optional grid
    configure_labels(ax, config, coord_system_type="radec", fontsize=fontsize)
    if config.get("gridlines", False):
        ax.grid(color="black")

    # Attach colorbar to the right
    add_colorbar(ax, im, tick_fontsize=fontsize)

    if new_figure:
        fig.tight_layout()
    return fig, ax, im

def _overlay_counts_text_radec(ax, data, scaled_boundaries, base_fontsize):
    """Draw count labels at pixel centers in scaled RA/Dec coordinates."""
    height, width = data.shape
    x_min = scaled_boundaries["coord1_min"]
    x_max = scaled_boundaries["coord1_max"]
    y_min = scaled_boundaries["coord2_min"]
    y_max = scaled_boundaries["coord2_max"]
    x_centers = [x_min + (j + 0.5) * (x_max - x_min) / width for j in range(width)]
    y_centers = [y_min + (i + 0.5) * (y_max - y_min) / height for i in range(height)]
    count_fontsize = max(6, int(base_fontsize * 0.6))
    outline = [patheffects.withStroke(linewidth=1.8, foreground="black")]
    for i in range(height):
        for j in range(width):
            val = data[i, j]
            label = f"{int(round(val))}"
            ax.text(
                x_centers[j],
                y_centers[i],
                label,
                color="white",
                ha="center",
                va="center",
                fontsize=count_fontsize,
                path_effects=outline,
            )
