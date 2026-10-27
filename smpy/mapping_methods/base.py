"""Base class for mass mapping methods."""

from abc import ABC, abstractmethod

class MassMapper(ABC):
    """Base class for mass mapping methods.

    A method turns a shear grid into E- and B-mode maps. Subclasses take their
    settings as constructor arguments, whose defaults live in the ``methods``
    section of ``default.yaml``, and implement `create_maps`. A new method is
    registered by name in `smpy.mapping_methods.METHODS`.
    """

    @abstractmethod
    def create_maps(self, grid):
        """Create E- and B-mode maps from a shear grid.

        Parameters
        ----------
        grid : `smpy.coordinates.ShearGrid`
            Binned shear, weights and counts.

        Returns
        -------
        map_e : `numpy.ndarray`
            E-mode map with the grid's shape.
        map_b : `numpy.ndarray`
            B-mode map with the grid's shape.
        """
