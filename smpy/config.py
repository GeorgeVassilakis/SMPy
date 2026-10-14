"""Configuration for SMPy runs.

Settings start from ``configs/default.yaml``. A YAML file or keyword arguments
override them by name, and a name that is not in the defaults is rejected.
"""

import copy
import os
from pathlib import Path

import yaml

from smpy.mapping_methods import METHODS

DEFAULTS_PATH = Path(__file__).parent / "configs" / "default.yaml"
"""Packaged defaults for every setting (`pathlib.Path`)."""

PATH_SETTINGS = (
    ("general", "input_path"),
    ("general", "output_directory"),
    ("plotting", "xray_contours", "ctr_file"),
)
"""Settings that hold file paths (`tuple` of key sequences)."""

class Config:
    """Settings for an SMPy run: the defaults plus overrides.

    Parameters
    ----------
    overrides : `dict`, optional
        Nested settings that replace defaults, such as the contents of a YAML
        file.
    source_dir : `str` or `pathlib.Path`, optional
        Directory that relative paths in the settings are relative to. If
        `None`, they are relative to the working directory.

    Raises
    ------
    ValueError
        Raised if ``overrides`` contains a setting that does not exist.

    Examples
    --------
    >>> config = Config.from_defaults("ks_plus", input_path="catalog.fits",
    ...                               pixel_scale=0.4)
    >>> config.save("run.yaml")
    >>> result = smpy.run("run.yaml")
    """

    def __init__(self, overrides=None, source_dir=None):
        with open(DEFAULTS_PATH, encoding="utf-8") as stream:
            self._settings = yaml.safe_load(stream)
        _merge(self._settings, overrides or {})
        self.source_dir = Path(source_dir) if source_dir is not None else None

    @classmethod
    def from_file(cls, path):
        """Load settings from a YAML file.

        Parameters
        ----------
        path : `str` or `pathlib.Path`
            YAML file. Relative paths inside it are relative to its directory.

        Returns
        -------
        config : `Config`
            Defaults overridden by the file.
        """
        path = Path(path)
        with open(path, encoding="utf-8") as stream:
            overrides = yaml.safe_load(stream)
        return cls(overrides, source_dir=path.resolve().parent)

    @classmethod
    def from_defaults(cls, method=None, **settings):
        """Build settings from the defaults and keyword arguments.

        Parameters
        ----------
        method : `str`, optional
            Mapping method. The default is 'kaiser_squires'.
        **settings
            Settings by name, as accepted by `smpy.map_mass`, plus the
            ``general`` output settings such as ``output_directory``. `None`
            values leave the default in place.

        Returns
        -------
        config : `Config`
            Defaults overridden by the keyword arguments.

        Raises
        ------
        ValueError
            Raised if a setting does not exist for the chosen method and
            coordinate system.
        """
        config = cls()
        _apply_keywords(config._settings, dict(settings, method=method))
        return config

    def validate(self):
        """Check that required settings are set and values are valid.

        Raises
        ------
        ValueError
            Raised for the first missing or invalid setting, which it names.
        """
        general = self._settings["general"]
        _check_choice("general.method", general["method"], METHODS)
        _check_choice("general.coord_system", general["coord_system"], ("radec", "pixel"))
        scale = "pixel_scale" if general["coord_system"] == "radec" else "downsample_factor"
        if general[general["coord_system"]][scale] is None:
            raise ValueError(f"general.{general['coord_system']}.{scale} is required "
                             f"(map_mass: {scale}=...)")
        if not set(general["mode"]) <= {"E", "B"}:
            raise ValueError(f"general.mode may contain only 'E' and 'B', got {general['mode']}")
        _check_choice("snr.shuffle_type", self._settings["snr"]["shuffle_type"], ("spatial", "orientation"))
        num_shuffles = self._settings["snr"]["num_shuffles"]
        if not isinstance(num_shuffles, int) or num_shuffles < 2:
            raise ValueError(f"snr.num_shuffles must be an integer of at least 2; got {num_shuffles!r}")
        _check_xray_contours(self._settings["plotting"]["xray_contours"])
        METHODS[general["method"]](**self._settings["methods"][general["method"]])

    def resolved(self):
        """Return the settings with relative paths made absolute.

        Returns
        -------
        settings : `dict`
            Copy of the settings, with relative paths joined to
            ``source_dir``, or to the working directory if it is `None`.
        """
        settings = self.to_dict()
        base = self.source_dir or Path.cwd()
        for section, key in _path_settings(settings):
            if section[key]:
                section[key] = os.path.normpath(base / section[key])
        return settings

    def save(self, path):
        """Write the settings to a YAML file.

        Relative paths are rewritten against the new file's directory, so they
        still point at the same files.

        Parameters
        ----------
        path : `str` or `pathlib.Path`
            Destination YAML file.
        """
        path = Path(path)
        settings = self.to_dict()
        old_base, new_base = self.source_dir or Path.cwd(), path.resolve().parent
        for section, key in _path_settings(settings):
            if section[key] and not os.path.isabs(section[key]):
                section[key] = os.path.relpath(old_base / section[key], new_base)
        with open(path, "w", encoding="utf-8") as stream:
            yaml.safe_dump(settings, stream, sort_keys=False)

    def to_dict(self):
        """Return a copy of the settings.

        Returns
        -------
        settings : `dict`
            Nested settings with ``general``, ``methods``, ``plotting`` and
            ``snr`` sections.
        """
        return copy.deepcopy(self._settings)

    def __str__(self):
        return yaml.safe_dump(self._settings, sort_keys=False)

def _merge(base, overrides, prefix=""):
    """Update ``base`` from ``overrides``, rejecting unknown keys."""
    for key, value in overrides.items():
        name = f"{prefix}{key}"
        if key not in base:
            raise ValueError(f"Unknown setting '{name}'. Valid settings here: {', '.join(base)}")
        if isinstance(base[key], dict) and value is None:
            continue  # An empty YAML section keeps its defaults
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{name}.")
        else:
            base[key] = value

def _apply_keywords(settings, keywords):
    """Set settings by name; coordinate settings go to the chosen system."""
    general = settings["general"]
    for key, choices in (("method", settings["methods"]), ("coord_system", ("radec", "pixel"))):
        value = keywords.pop(key, None)
        if value is not None:
            _check_choice(f"general.{key}", value, choices)
            general[key] = value

    targets = (general[general["coord_system"]], general, settings["methods"][general["method"]])
    for key, value in keywords.items():
        if value is None:
            continue
        if key in ("plotting", "snr"):
            _merge(settings[key], value, f"{key}.")
            continue
        target = next((section for section in targets if key in section), None)
        if target is None:
            raise ValueError(f"Unknown setting '{key}' for method '{general['method']}' "
                             f"with coord_system '{general['coord_system']}'")
        target[key] = value

def _path_settings(settings):
    """Yield ``(section, key)`` for each path setting in ``settings``."""
    for *parents, key in PATH_SETTINGS:
        section = settings
        for parent in parents:
            section = section.get(parent) if isinstance(section, dict) else None
        if isinstance(section, dict):
            yield section, key

def _check_choice(name, value, choices):
    """Raise `ValueError` unless ``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}; got {value!r}")

def _check_xray_contours(contours):
    """Raise `ValueError` for invalid ``plotting.xray_contours`` values."""
    prefix = "plotting.xray_contours"
    if not isinstance(contours, dict):
        raise ValueError(f"{prefix} must be a mapping")
    if contours["ctr_file"] is not None and not isinstance(contours["ctr_file"], str):
        raise ValueError(f"{prefix}.ctr_file must be a path or null")
    for flag in ("show_on_convergence", "show_on_snr"):
        if not isinstance(contours[flag], bool):
            raise ValueError(f"{prefix}.{flag} must be true or false")
    if not isinstance(contours["alpha"], (int, float)) or not 0 <= contours["alpha"] <= 1:
        raise ValueError(f"{prefix}.alpha must be a number between 0 and 1")
    if not isinstance(contours["linewidth"], (int, float)) or contours["linewidth"] <= 0:
        raise ValueError(f"{prefix}.linewidth must be a positive number")
