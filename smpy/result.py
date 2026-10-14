"""Results of an SMPy mass mapping run."""

import copy
import logging
from dataclasses import dataclass

from astropy.io import fits
from astropy.wcs import WCS

from smpy.config import _check_plotting, _merge
from smpy.coordinates import ShearGrid
from smpy.plotting.plot import plot_mass_map

logger = logging.getLogger(__name__)

@dataclass
class MassMap:
    """Mass maps from one SMPy run, with the grid they were made on.

    Nothing is written to disk until a plot is saved or `to_fits` is called.

    Parameters
    ----------
    maps : `dict` [`str`, `numpy.ndarray`]
        E- and B-mode maps under ``"E"`` and ``"B"``, with the grid's shape.
    snr : `dict` [`str`, `numpy.ndarray`] or `None`
        E- and B-mode signal-to-noise maps, or `None` if not computed.
    grid : `smpy.coordinates.ShearGrid`
        Binned shear, weights and galaxy counts the maps were made from.
    scaled_boundaries : `dict`
        Map extent in the flattened coordinates used for gridding.
    true_boundaries : `dict`
        Map extent in the catalog's coordinates.
    settings : `dict`
        Every setting used, with absolute paths.
    """

    maps: dict
    snr: dict | None
    grid: ShearGrid
    scaled_boundaries: dict
    true_boundaries: dict
    settings: dict

    def plot(self, mode="E", *, snr=False, ax=None, **style):
        """Plot an E- or B-mode map, or its signal-to-noise map.

        Parameters
        ----------
        mode : `str`, optional
            'E' or 'B'.
        snr : `bool`, optional
            If `True`, plot the signal-to-noise map instead.
        ax : `matplotlib.axes.Axes`, optional
            Axes to draw on. If `None`, a new figure is created.
        **style
            Overrides for the ``plotting`` settings, such as ``cmap="magma"``.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot. It is neither shown nor closed.
        ax : `matplotlib.axes.Axes`
            Axes the map was drawn on.
        im : `matplotlib.image.AxesImage`
            The map image.

        Raises
        ------
        ValueError
            Raised if ``snr`` is `True` but no SNR maps were computed, or if a
            style setting does not exist.
        """
        data = self._select(mode, snr)
        config = self._plot_config(style)
        title = config["plot_title"]
        if snr and "plot_title" not in style:
            title = self.settings["snr"]["plot_title"]
        config["plot_title"] = f"{title} ({mode}-mode)"
        counts = self.grid.counts if config["overlay_counts"] else None
        return plot_mass_map(data, self.scaled_boundaries, self.true_boundaries, config, ax=ax,
                             map_category="snr" if snr else "convergence", counts_overlay=counts)

    def plot_counts(self, ax=None, **style):
        """Plot the number of galaxies in each map pixel.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            Axes to draw on. If `None`, a new figure is created.
        **style
            Overrides for the ``plotting`` settings. The color scale is
            always linear.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot. It is neither shown nor closed.
        ax : `matplotlib.axes.Axes`
            Axes the counts were drawn on.
        im : `matplotlib.image.AxesImage`
            The counts image.
        """
        config = self._plot_config(style)
        config["plot_title"] = "Counts Map"
        scaling = dict(config["scaling"]) if isinstance(config["scaling"], dict) else {}
        scaling["type"] = "linear"
        scaling.pop("percentile", None)
        config["scaling"] = scaling
        return plot_mass_map(self.grid.counts, self.scaled_boundaries, self.true_boundaries, config, ax=ax,
                             map_category="counts")

    def to_fits(self, path, mode="E", *, snr=False):
        """Write an E- or B-mode map, or its signal-to-noise map, to FITS.

        Parameters
        ----------
        path : `str` or `pathlib.Path`
            Output file. An existing file is overwritten.
        mode : `str`, optional
            'E' or 'B'.
        snr : `bool`, optional
            If `True`, write the signal-to-noise map instead.

        Raises
        ------
        ValueError
            Raised if ``snr`` is `True` but no SNR maps were computed.
        """
        hdu = fits.PrimaryHDU(self._select(mode, snr))
        for key, value in self._wcs_cards():
            hdu.header[key] = value
        hdu.writeto(path, overwrite=True)
        logger.info("Saved FITS file: %s", path)

    @property
    def wcs(self):
        """World coordinate system of the map pixels (`astropy.wcs.WCS`).

        For RA/Dec maps it is the gnomonic (TAN) projection the catalog was
        gridded in, with east to the left. For pixel maps it maps map pixels
        to catalog pixel coordinates.
        """
        return WCS(fits.Header(self._wcs_cards()))

    def _select(self, mode, snr):
        """Return the requested map, checking that SNR maps exist if asked."""
        if snr and self.snr is None:
            raise ValueError("No SNR maps were computed; use create_snr=True")
        return (self.snr if snr else self.maps)[mode]

    def _plot_config(self, style):
        """Return the plotting settings with ``style`` overrides applied."""
        config = copy.deepcopy(self.settings["plotting"])
        _merge(config, style, "plotting.")
        config["coordinate_system"] = self.settings["general"]["coord_system"]
        _check_plotting(config, config["coordinate_system"])
        return config

    def _wcs_cards(self):
        """Return FITS WCS header cards for the map grid."""
        edges = self.scaled_boundaries
        size = edges["pixel_size"]
        # FITS pixel numbers start at 1 and fall on pixel centres
        crpix1, crpix2 = 0.5 - edges["coord1_min"] / size, 0.5 - edges["coord2_min"] / size
        if self.settings["general"]["coord_system"] == "radec":
            sky = self.true_boundaries
            return [
                ("CTYPE1", "RA---TAN"), ("CUNIT1", "deg"), ("CRVAL1", sky["coord1_center"] % 360),
                ("CRPIX1", crpix1), ("CD1_1", -size), ("CD1_2", 0.0),
                ("CTYPE2", "DEC--TAN"), ("CUNIT2", "deg"), ("CRVAL2", sky["coord2_center"]),
                ("CRPIX2", crpix2), ("CD2_1", 0.0), ("CD2_2", size),
            ]
        return [
            ("CTYPE1", "X"), ("CRPIX1", crpix1), ("CRVAL1", 0.0), ("CDELT1", size),
            ("CTYPE2", "Y"), ("CRPIX2", crpix2), ("CRVAL2", 0.0), ("CDELT2", size),
        ]
