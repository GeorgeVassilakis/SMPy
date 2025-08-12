"""Plotting functions for SMPy mass and SNR maps.

Provides high-level plotting for pixel and RA/Dec coordinate systems
with consistent styling, scaling, overlays, and saving. Helpers live in
``smpy.plotting.utils`` to keep this module focused on public plotting
APIs and orchestration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from smpy.utils import find_peaks2d
from smpy.plotting.utils import (
    add_colorbar,
    apply_axes_style,
    apply_ra_orientation,
    compute_pixel_extent,
    configure_labels,
    convert_center_to_scaled,
    create_normalization,
    peaks_to_plot_coords,
    propose_ticks,
    set_ticks,
)


def plot_mass_map(
    data: np.ndarray,
    scaled_boundaries: dict,
    true_boundaries: dict,
    config: dict,
    output_name: Optional[str] = None,
    return_handles: bool = False,
    map_category: str = "convergence",
) -> Optional[Tuple[plt.Figure, plt.Axes, any]]:
    """Plot a mass-like map (E/B mode) with styling and overlays.

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
        'coordinate_system', and optional 'axis_reference' (pixel only).
    output_name : `str`, optional
        Path for saving the plot file.
    return_handles : `bool`, optional
        If ``True``, return ``(fig, ax, im)`` instead of closing.
    map_category : `str`, optional
        Map category used for scaling overrides. Options: 'convergence',
        'snr'.

    Returns
    -------
    handles : tuple, optional
        Returns ``(fig, ax, im)`` if ``return_handles=True``.
    """
    # Dispatch to coordinate-system specific renderer based on config
    coord_system = config.get("coordinate_system", "radec").lower()
    if coord_system == "radec":
        return _plot_radec(
            data,
            scaled_boundaries,
            true_boundaries,
            config,
            output_name,
            return_handles,
            map_category,
        )
    return _plot_pixel(
        data,
        scaled_boundaries,
        true_boundaries,
        config,
        output_name,
        return_handles,
        map_category,
    )

def _plot_pixel(
    data: np.ndarray,
    scaled_boundaries: dict,
    true_boundaries: dict,
    config: dict,
    output_name: Optional[str],
    return_handles: bool,
    map_category: str,
):
    """Render pixel-coordinate plot with overlays and colorbar."""
    # Create figure/axes and apply local styling (no global rc changes)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=config.get("figsize", (12, 8)))
    fontsize = int(config.get("fontsize", 15))
    apply_axes_style(ax, fontsize=fontsize)

    # Choose how axes are labeled in pixel mode
    axis_reference = str(config.get("axis_reference", "catalog")).lower()

    # Build colormap normalization from config (percentiles, power, symlog)
    norm = create_normalization(
        config.get("scaling"), data, vmin=config.get("vmin"), vmax=config.get("vmax"), map_type=map_category
    )

    # Determine image extent from chosen axis reference
    extent = compute_pixel_extent(data, scaled_boundaries, axis_reference)
    im = ax.imshow(data, cmap=config.get("cmap", "viridis"), norm=norm, extent=extent, origin="lower")

    # Optional: mark cluster center; convert to map pixels if needed
    cx, cy = convert_center_to_scaled(
        config.get("cluster_center"), scaled_boundaries, true_boundaries, coord_system_type="pixel"
    )
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

    # Optional: overlay peak markers above threshold
    threshold = config.get("threshold")
    if threshold is not None:
        verbose_peaks = bool(config.get("verbose", False))
        # Detect peaks using 2D local maxima algorithm
        X, Y, _, _ = find_peaks2d(
            data, threshold=threshold, verbose=verbose_peaks, true_boundaries=true_boundaries, scaled_boundaries=scaled_boundaries
        )
        # Convert peak indices to appropriate plotting coordinates
        px, py = peaks_to_plot_coords(X, Y, data, scaled_boundaries, axis_reference)
        ax.scatter(px, py, s=100, facecolors="none", edgecolors="g", linewidth=1.5)

    # Labels, title, optional grid
    configure_labels(
        ax,
        config,
        axis_reference=axis_reference,
        coord_system_type="pixel",
        fontsize=fontsize,
    )
    if config.get("gridlines", False):
        ax.grid(color="black")

    # Attach colorbar to the right
    add_colorbar(ax, im, tick_fontsize=fontsize)

    # Save and/or return figure
    fig.tight_layout()
    if output_name:
        fig.savefig(output_name)
        map_label = "Convergence" if map_category.lower() == "convergence" else "SNR"
        print(f"{map_label} map saved as PNG file: {output_name}")
    
    if return_handles:
        return fig, ax, im
    plt.close(fig)
    return None

def _plot_radec(
    data: np.ndarray,
    scaled_boundaries: dict,
    true_boundaries: dict,
    config: dict,
    output_name: Optional[str],
    return_handles: bool,
    map_category: str,
):
    """Render RA/Dec plot with astronomical orientation and ticks."""
    # Create figure/axes and apply local styling (no global rc changes)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=config.get("figsize", (12, 8)))
    fontsize = int(config.get("fontsize", 15))
    apply_axes_style(ax, fontsize=fontsize)

    # Build colormap normalization from config
    norm = create_normalization(
        config.get("scaling"), data, vmin=config.get("vmin"), vmax=config.get("vmax"), map_type=map_category
    )

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

    # Optional: overlay peak markers above threshold
    threshold = config.get("threshold")
    if threshold is not None:
        verbose_peaks = bool(config.get("verbose", False))
        # Detect peaks using 2D local maxima algorithm
        X, Y, _, _ = find_peaks2d(
            data, threshold=threshold, verbose=verbose_peaks, true_boundaries=true_boundaries, scaled_boundaries=scaled_boundaries
        )
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

    # Optional: mark cluster center in RA/Dec coordinates
    ra_center, dec_center = convert_center_to_scaled(
        config.get("cluster_center"), scaled_boundaries, true_boundaries, coord_system_type="radec"
    )
    if ra_center is not None:
        ax.plot(ra_center, dec_center, "rx", markersize=10)

    # Generate ticks: propose in true coordinate space, then map to scaled plotting space
    ra_ticks_true, ra_labels = propose_ticks(true_boundaries["coord1_min"], true_boundaries["coord1_max"], 5)
    dec_ticks_true, dec_labels = propose_ticks(true_boundaries["coord2_min"], true_boundaries["coord2_max"], 5)
    # Transform tick positions from true to scaled coordinates for plotting
    scaled_x = np.interp(
        ra_ticks_true,
        [true_boundaries["coord1_min"], true_boundaries["coord1_max"]],
        [scaled_boundaries["coord1_min"], scaled_boundaries["coord1_max"]],
    )
    scaled_y = np.interp(
        dec_ticks_true,
        [true_boundaries["coord2_min"], true_boundaries["coord2_max"]],
        [scaled_boundaries["coord2_min"], scaled_boundaries["coord2_max"]],
    )
    set_ticks(ax, scaled_x, scaled_y, ra_labels, dec_labels)

    # Labels, title, optional grid
    configure_labels(ax, config, coord_system_type="radec", fontsize=fontsize)
    if config.get("gridlines", False):
        ax.grid(color="black")

    # Astronomical convention: RA increases to the left
    apply_ra_orientation(ax)

    # Attach colorbar to the right
    add_colorbar(ax, im, tick_fontsize=fontsize)

    # Save and/or return figure
    fig.tight_layout()
    if output_name:
        fig.savefig(output_name)
        map_label = "Convergence" if map_category.lower() == "convergence" else "SNR"
        print(f"{map_label} map saved as PNG file: {output_name}")

    if return_handles:
        return fig, ax, im
    plt.close(fig)
    return None


def plot_snr_map(
    data: np.ndarray,
    scaled_boundaries: dict,
    true_boundaries: dict,
    config: dict,
    output_name: Optional[str] = None,
    return_handles: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes, any]]:
    """Plot an SNR map with styling and overlays.

    Parameters
    ----------
    See :func:`plot_mass_map`.
    """
    # Delegate to mass_map with SNR category for proper scaling overrides
    return plot_mass_map(
        data=data,
        scaled_boundaries=scaled_boundaries,
        true_boundaries=true_boundaries,
        config=config,
        output_name=output_name,
        return_handles=return_handles,
        map_category="snr",
    )
