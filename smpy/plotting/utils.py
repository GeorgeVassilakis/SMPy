"""Plotting utilities shared by SMPy plotting functions.

This module contains small, focused helpers used by the plotting layer.
Helpers avoid mutating global matplotlib state and operate on provided
axes/inputs for predictable behavior.
"""

# Standard Library
from __future__ import annotations

# Third Party
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from smpy.coordinates.radec import project

def create_normalization(scaling, data, vmin=None, vmax=None, map_type="convergence"):
    """Create a matplotlib normalization object from config.

    Parameters
    ----------
    scaling : `dict` or None
        Scaling configuration with a 'type' key.
        Supported types: 'linear', 'power', 'symlog'. Optional keys include
        'gamma' (for power), 'percentile' (pair), and map-specific overrides
        under keys matching ``map_type`` for 'symlog'.
    data : `numpy.ndarray`
        Data array used for percentile-based range computation; NaN pixels
        are left out.
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

    scale_type = scaling.get("type", "linear").lower()

    # Handle percentile-based range calculation
    percentile = scaling.get("percentile")
    if percentile is not None:
        vmin = float(np.nanpercentile(data, percentile[0]))
        vmax = float(np.nanpercentile(data, percentile[1]))

    # Create appropriate normalization based on type
    if scale_type == "linear":
        return colors.Normalize(vmin=vmin, vmax=vmax)

    if scale_type == "power":
        gamma = float(scaling.get("gamma", 1.0))
        return colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax)

    if scale_type == "symlog":
        # Allow map-specific parameter overrides (e.g., different thresholds
        # for SNR vs convergence)
        map_specific_params = scaling.get(map_type, {}) or {}
        linthresh = float(map_specific_params.get("linthresh", scaling.get("linthresh", 0.1)))
        linscale = float(map_specific_params.get("linscale", scaling.get("linscale", 1.0)))
        return colors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)

    raise ValueError(f"plotting.scaling.type must be one of linear, power, symlog; got {scale_type!r}")


def apply_axes_style(ax, fontsize=15):
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


def configure_labels(ax, cfg, axis_reference=None, coord_system_type=None, fontsize=None):
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
            # Pixel coordinates can reference map indices or catalog
            # coordinates
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


def propose_ticks(range_min, range_max, target_count=5):
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


def set_ticks(ax, x_ticks, y_ticks, x_tick_labels, y_tick_labels):
    """Apply tick positions and labels to axes."""
    ax.set_xticks(list(x_ticks))
    ax.set_yticks(list(y_ticks))
    ax.set_xticklabels(list(x_tick_labels))
    ax.set_yticklabels(list(y_tick_labels))


def compute_pixel_extent(data, scaled_boundaries, axis_reference):
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


def convert_center_to_scaled(cluster_center, scaled_boundaries, true_boundaries, coord_system_type):
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

    # Pixel plots use catalog coordinates; RA/Dec plots the tangent plane
    if coord_system_type != "radec":
        return float(cluster_center["x_center"]), float(cluster_center["y_center"])
    cx, cy = project(cluster_center["ra_center"], cluster_center["dec_center"],
                     true_boundaries["coord1_center"], true_boundaries["coord2_center"])
    return float(cx), float(cy)


def peaks_to_plot_coords(X, Y, data, scaled_boundaries, axis_reference):
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


def add_colorbar(ax, im, size="5%", pad=0.07, tick_fontsize=None):
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
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(im, cax=cax)
    # Disable minor ticks on the colorbar axis
    cb.ax.minorticks_off()
    if tick_fontsize is not None:
        cb.ax.tick_params(labelsize=tick_fontsize)


def read_ds9_ctr(ctr_path):
    """Read DS9 contour coordinates from a ``.ctr`` text file.

    Parameters
    ----------
    ctr_path : `str`
        Path to the contour file exported from DS9.

    Returns
    -------
    contours : `list` [`numpy.ndarray`]
        List of contours. Each contour is an ``(N, 2)`` array with columns
        ``x`` and ``y``.
    coord_type : `str`
        Coordinate type declared in the file. Supported values are
        ``'fk5'``, ``'wcs'``, ``'image'``, and ``'physical'``. Defaults to
        ``'fk5'`` if not specified in the file.
    """
    contours = []
    current_points = []
    coord_type = None

    with open(ctr_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if (not line) or line.startswith("#"):
                continue

            line_lower = line.lower()
            if line_lower in {"fk5", "wcs", "image", "physical"}:
                coord_type = line_lower
                continue

            if line_lower == "line":
                if current_points:
                    contours.append(np.asarray(current_points, dtype=float))
                    current_points = []
                continue

            try:
                values = [float(value) for value in line.replace(",", " ").split()]
            except ValueError:
                continue

            if len(values) >= 2:
                current_points.append((values[0], values[1]))

    if current_points:
        contours.append(np.asarray(current_points, dtype=float))

    if coord_type is None:
        coord_type = "fk5"

    return contours, coord_type


def overlay_xray_contours(
    ax,
    data_shape,
    scaled_boundaries,
    true_boundaries,
    config,
    map_category,
    coord_system_type,
    axis_reference=None,
):
    """Overlay DS9 x-ray contours onto a map axis if enabled.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    data_shape : `tuple` [`int`, `int`]
        Map shape as ``(height, width)``.
    scaled_boundaries : `dict`
        Scaled coordinate boundaries used for plotting extents.
    true_boundaries : `dict`
        True coordinate boundaries (needed for FK5 conversion).
    config : `dict`
        Plot configuration dictionary.
    map_category : `str`
        Map type for routing toggle logic (``'convergence'`` or ``'snr'``).
    coord_system_type : `str`
        Coordinate system used by the plot (``'radec'`` or ``'pixel'``).
    axis_reference : `str`, optional
        Pixel-axis reference (``'catalog'`` or ``'map'``) for pixel plots.
    """
    xray_cfg = config.get("xray_contours")
    if not isinstance(xray_cfg, dict):
        return

    category = str(map_category or "").lower()
    if category == "convergence":
        enabled = bool(xray_cfg.get("show_on_convergence", False))
    elif category == "snr":
        enabled = bool(xray_cfg.get("show_on_snr", False))
    else:
        enabled = False

    if not enabled:
        return

    ctr_file = xray_cfg.get("ctr_file")
    if not ctr_file:
        return

    contours, coord_type = read_ds9_ctr(ctr_file)

    transformed = _transform_contours_for_plot(
        contours=contours,
        coord_type=coord_type,
        data_shape=data_shape,
        scaled_boundaries=scaled_boundaries,
        true_boundaries=true_boundaries,
        coord_system_type=coord_system_type,
        axis_reference=axis_reference,
    )

    if not transformed:
        return

    color = str(xray_cfg.get("color", "cyan"))
    linewidth = float(xray_cfg.get("linewidth", 0.8))
    alpha = float(xray_cfg.get("alpha", 0.7))

    for contour in transformed:
        if contour.shape[0] < 2:
            continue
        ax.plot(
            contour[:, 0],
            contour[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _transform_contours_for_plot(
    contours,
    coord_type,
    data_shape,
    scaled_boundaries,
    true_boundaries,
    coord_system_type,
    axis_reference,
):
    """Convert contour coordinates to the current plot coordinate system."""
    if not contours:
        return []

    coord_type = str(coord_type or "").lower()
    if coord_type in {"fk5", "wcs"}:
        if str(coord_system_type).lower() != "radec":
            raise ValueError(f"Contours in {coord_type} coordinates need an RA/Dec map; this map is in "
                             "pixel coordinates")
        return _convert_world_contours_to_scaled(contours, true_boundaries)

    if coord_type in {"image", "physical"}:
        return _convert_image_contours_to_plot_coords(
            contours=contours,
            data_shape=data_shape,
            scaled_boundaries=scaled_boundaries,
            coord_system_type=coord_system_type,
            axis_reference=axis_reference,
        )

    raise ValueError(f"Unsupported DS9 contour coordinate type {coord_type!r}; use fk5, wcs, image or "
                     "physical")


def _convert_world_contours_to_scaled(contours, true_boundaries):
    """Project RA/Dec contours onto the map's tangent plane."""
    center = (true_boundaries["coord1_center"], true_boundaries["coord2_center"])
    return [np.column_stack(project(contour[:, 0], contour[:, 1], *center)) for contour in contours]


def _convert_image_contours_to_plot_coords(
    contours,
    data_shape,
    scaled_boundaries,
    coord_system_type,
    axis_reference,
):
    """Convert image/physical contours to active plotting coordinates.

    Notes
    -----
    DS9 image-like coordinates are treated as 1-indexed pixel coordinates.
    """
    height, width = data_shape
    if height <= 0 or width <= 0:
        return []

    axis_reference = str(axis_reference or "catalog").lower()
    coord_system_type = str(coord_system_type).lower()

    x_min = scaled_boundaries["coord1_min"]
    x_max = scaled_boundaries["coord1_max"]
    y_min = scaled_boundaries["coord2_min"]
    y_max = scaled_boundaries["coord2_max"]

    transformed = []
    for contour in contours:
        x_image = contour[:, 0] - 0.5
        y_image = contour[:, 1] - 0.5

        if coord_system_type == "pixel" and axis_reference == "map":
            x_plot = x_image
            y_plot = y_image
        else:
            x_plot = x_min + (x_image / width) * (x_max - x_min)
            y_plot = y_min + (y_image / height) * (y_max - y_min)

        transformed.append(np.column_stack((x_plot, y_plot)))

    return transformed
