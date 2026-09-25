"""Mass mapping methods.

Each method is a `MassMapper` subclass, registered in `METHODS` under the name
used by ``method=...`` and by the ``methods`` section of ``default.yaml``.
"""

from .aperture_mass.aperture_mass import ApertureMassMapper
from .base import MassMapper
from .kaiser_squires.kaiser_squires import KaiserSquiresMapper
from .ks_plus.ks_plus import KSPlusMapper

METHODS = {
    "kaiser_squires": KaiserSquiresMapper,
    "aperture_mass": ApertureMassMapper,
    "ks_plus": KSPlusMapper,
}
"""Mass mapping methods by name (`dict` [`str`, `type`])."""

__all__ = ["METHODS", "MassMapper", "KaiserSquiresMapper", "ApertureMassMapper", "KSPlusMapper"]
