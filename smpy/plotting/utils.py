"""Plotting utilities shared by SMPy plotting functions.

This module contains small, focused helpers used by the plotting layer.
Helpers avoid mutating global matplotlib state and operate on provided
axes/inputs for predictable behavior.
"""

from __future__ import annotations

import logging
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_normalization(
    scaling: Optional[dict | str],
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    map_type: str = "convergence",
):
    """Create a matplotlib normalization object from config.

    Parameters
    ----------
    scaling : `dict` or `str` or None
        Scaling configuration. If a string, treated as ``{"type": <str>}``.
        Supported types: 'linear', 'power', 'symlog'. Optional keys include
        'gamma' (for power), 'percentile' (pair), and map-specific overrides
        under keys matching ``map_type`` for 'symlog'.
    data : `numpy.ndarray`
        Data array used for percentile-based range computation.
    vmin, vmax : `float`, optional
        Explicit min/max for scaling.
    map_type : `str`, optional
        Map category ('convergence' or 'snr'). Used for symlog overrides.

    Returns
    -------
    norm : `matplotlib.colors.Normalize`
        Normalization instance.
    """
    # Default to linear scaling if no configuration provided
    if scaling is None:
        return colors.Normalize(vmin=vmin, vmax=vmax)

    # Convert string shorthand to dict format
    if isinstance(scaling, str):
        scaling = {"type": scaling}

    scale_type = scaling.get("type", "linear").lower()

    # Handle percentile-based range calculation
    percentile = scaling.get("percentile")
    if percentile is not None:
        if isinstance(percentile, (list, tuple)) and len(percentile) == 2:
            vmin = float(np.percentile(data, percentile[0]))
            vmax = float(np.percentile(data, percentile[1]))
        else:
            warnings.warn(
                "'percentile' should be a two-element list/tuple [min, max]; ignoring.",
                RuntimeWarning,
            )

    # Create appropriate normalization based on type
    if scale_type == "linear":
        return colors.Normalize(vmin=vmin, vmax=vmax)

    if scale_type == "power":
        gamma = float(scaling.get("gamma", 1.0))
        return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    if scale_type == "symlog":
        # Allow map-specific parameter overrides (e.g., different thresholds for SNR vs convergence)
        map_specific_params = scaling.get(map_type, {}) or {}
        linthresh = float(map_specific_params.get("linthresh", scaling.get("linthresh", 0.1)))
        linscale = float(map_specific_params.get("linscale", scaling.get("linscale", 1.0)))
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)

    warnings.warn(f"Unknown scaling type '{scale_type}', falling back to linear.")
    return colors.Normalize(vmin=vmin, vmax=vmax)


def apply_axes_style(ax: plt.Axes, fontsize: int = 15) -> None:
    """Apply consistent, local styling to axes (no global rc mutation).

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    fontsize : `int`, optional
        Base font size for labels and ticks.
    """
    # Configure tick appearance without modifying global matplotlib state
    ax.tick_params(
        which="both",
        direction="in",
        labelsize=fontsize,
        width=1.3,
        length=8,
    )
    # Set consistent spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def configure_labels(
    ax: plt.Axes,
    cfg: dict,
    axis_reference: Optional[str] = None,
    coord_system_type: Optional[str] = None,
    fontsize: Optional[int] = None,
) -> None:
    """Configure x/y labels and title from plotting config.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    cfg : `dict`
        Plot configuration containing 'xlabel', 'ylabel', and 'plot_title'.
    axis_reference : `str`, optional
        For pixel coordinates: 'map' or 'catalog'.
    coord_system_type : `str`, optional
        'radec' or 'pixel'. Used for default labels when 'auto'.
    """
    xlabel = cfg.get("xlabel")
    ylabel = cfg.get("ylabel")

    # Handle automatic labeling based on coordinate system
    if xlabel == "auto":
        if coord_system_type == "radec":
            ax.set_xlabel("Right Ascension (deg)", fontsize=fontsize)
        else:
            # Pixel coordinates can reference map indices or catalog coordinates
            label = "X (map pixels)" if axis_reference == "map" else "X (pixels)"
            ax.set_xlabel(label, fontsize=fontsize)
    elif xlabel:
        ax.set_xlabel(str(xlabel), fontsize=fontsize)

    if ylabel == "auto":
        if coord_system_type == "radec":
            ax.set_ylabel("Declination (deg)", fontsize=fontsize)
        else:
            label = "Y (map pixels)" if axis_reference == "map" else "Y (pixels)"
            ax.set_ylabel(label, fontsize=fontsize)
    elif ylabel:
        ax.set_ylabel(str(ylabel), fontsize=fontsize)

    # Set plot title
    title = cfg.get("plot_title") or ""
    ax.set_title(title, fontsize=fontsize)


def apply_ra_orientation(ax: plt.Axes) -> None:
    """Orient RA to increase leftward by inverting x-axis.

    Notes
    -----
    Prefer axis inversion to data reversal; keeps overlays straightforward.
    """
    ax.invert_xaxis()


def propose_ticks(
    range_min: float, range_max: float, target_count: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Propose tick positions and labels for a range.

    Parameters
    ----------
    range_min, range_max : `float`
        Range bounds in data units.
    target_count : `int`, optional
        Approximate number of ticks.

    Returns
    -------
    ticks : `numpy.ndarray`
        Tick values in data units.
    labels : `numpy.ndarray`
        String labels corresponding to ``ticks``.
    """
    span = float(range_max - range_min)
    if span <= 0:
        return np.array([]), np.array([])

    # Choose step size from predefined candidates based on desired tick density
    candidates = np.array([0.01, 0.05, 0.1, 0.2, 0.5])
    step = candidates[np.abs(span / max(target_count, 1) - candidates).argmin()]

    # Generate tick positions aligned to step boundaries
    start = np.ceil(range_min / step) * step
    stop = np.floor(range_max / step) * step + step / 2
    ticks = np.arange(start, stop, step)
    labels = np.array([f"{x:.2f}" for x in ticks])
    return ticks, labels


def set_ticks(
    ax: plt.Axes,
    x_ticks: Sequence[float],
    y_ticks: Sequence[float],
    x_tick_labels: Sequence[str],
    y_tick_labels: Sequence[str],
) -> None:
    """Apply tick positions and labels to axes."""
    ax.set_xticks(list(x_ticks))
    ax.set_yticks(list(y_ticks))
    ax.set_xticklabels(list(x_tick_labels))
    ax.set_yticklabels(list(y_tick_labels))


def compute_pixel_extent(
    data: np.ndarray, scaled_boundaries: dict, axis_reference: str
) -> List[float]:
    """Compute imshow extent for pixel coordinates.

    Parameters
    ----------
    data : `numpy.ndarray`
        Image data array (for map-pixel extent calculation).
    scaled_boundaries : `dict`
        Scaled coordinate boundaries.
    axis_reference : `str`
        'map' to use map indices; 'catalog' to use scaled boundaries.
    """
    axis_reference = str(axis_reference or "catalog").lower()
    if axis_reference == "map":
        # Use map pixel indices as axis units (0 to width/height)
        height, width = data.shape
        return [0, width, 0, height]
    # Use catalog coordinate values as axis units
    return [
        scaled_boundaries["coord1_min"],
        scaled_boundaries["coord1_max"],
        scaled_boundaries["coord2_min"],
        scaled_boundaries["coord2_max"],
    ]


def convert_center_to_scaled(
    cluster_center: Optional[dict | str],
    scaled_boundaries: dict,
    true_boundaries: dict,
    coord_system_type: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert provided center to scaled coordinates for plotting.

    Parameters
    ----------
    cluster_center : `str` or `dict` or None
        'auto' to use field center; dict with `{ra/x_center, dec/y_center}`;
        or None for no center.
    scaled_boundaries, true_boundaries : `dict`
        Boundary dictionaries.
    coord_system_type : `str`
        'radec' or 'pixel'.
    """
    if cluster_center is None:
        return None, None

    # Use geometric center of the field
    if cluster_center == "auto":
        cx = (scaled_boundaries["coord1_max"] + scaled_boundaries["coord1_min"]) / 2
        cy = (scaled_boundaries["coord2_max"] + scaled_boundaries["coord2_min"]) / 2
        return cx, cy

    if isinstance(cluster_center, dict):
        # Select appropriate keys based on coordinate system
        k1, k2 = ("ra_center", "dec_center") if coord_system_type == "radec" else ("x_center", "y_center")
        if k1 not in cluster_center or k2 not in cluster_center:
            warnings.warn(f"Expected '{k1}' and '{k2}' in cluster_center; ignoring center.")
            return None, None

        # Transform from true coordinates to scaled plot coordinates
        cx = np.interp(
            cluster_center[k1],
            [true_boundaries["coord1_min"], true_boundaries["coord1_max"]],
            [scaled_boundaries["coord1_min"], scaled_boundaries["coord1_max"]],
        )
        cy = np.interp(
            cluster_center[k2],
            [true_boundaries["coord2_min"], true_boundaries["coord2_max"]],
            [scaled_boundaries["coord2_min"], scaled_boundaries["coord2_max"]],
        )
        return float(cx), float(cy)

    warnings.warn("Unrecognized cluster_center format; ignoring center.")
    return None, None


def peaks_to_plot_coords(
    X: Iterable[int],
    Y: Iterable[int],
    data: np.ndarray,
    scaled_boundaries: dict,
    axis_reference: str,
) -> Tuple[List[float], List[float]]:
    """Convert peak index coordinates to plotting coordinates for pixel mode.

    Parameters
    ----------
    X, Y : iterable of `int`
        Peak indices along x and y (image space).
    data : `numpy.ndarray`
        Image array (for shape).
    scaled_boundaries : `dict`
        Boundary dictionary.
    axis_reference : `str`
        'map' or 'catalog'.
    """
    axis_reference = str(axis_reference or "catalog").lower()
    if axis_reference == "map":
        # Center markers within pixel with +0.5 offset
        return [x + 0.5 for x in X], [y + 0.5 for y in Y]

    # Transform peak indices to catalog coordinate space
    width = data.shape[1]
    height = data.shape[0]
    xs = [
        scaled_boundaries["coord1_min"]
        + (x + 0.5) * (scaled_boundaries["coord1_max"] - scaled_boundaries["coord1_min"]) / width
        for x in X
    ]
    ys = [
        scaled_boundaries["coord2_min"]
        + (y + 0.5) * (scaled_boundaries["coord2_max"] - scaled_boundaries["coord2_min"]) / height
        for y in Y
    ]
    return xs, ys


def add_colorbar(
    ax: plt.Axes,
    im: plt.Axes,
    size: str = "5%",
    pad: float = 0.07,
    tick_fontsize: Optional[int] = None,
) -> None:
    """Attach a colorbar axis to the right of the given axes.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Parent axes to attach the colorbar to.
    im : `matplotlib.artist.Artist`
        Image/artist to build the colorbar from.
    size : `str`, optional
        Colorbar width relative to the axes size.
    pad : `float`, optional
        Padding between axes and colorbar in inches.
    tick_fontsize : `int`, optional
        Font size for colorbar tick labels.
    """
    # Create colorbar axis and disable minor ticks locally (no global rc updates)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(im, cax=cax)
    try:
        cb.ax.minorticks_off()
    except Exception:  # pragma: no cover - older mpl
        pass
    if tick_fontsize is not None:
        try:
            cb.ax.tick_params(labelsize=tick_fontsize)
        except Exception:  # pragma: no cover
            pass


