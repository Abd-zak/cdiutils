"""
cdiutils - A Python package for (Bragg) Coherent X-ray Diffraction
Imaging processing, analysis and visualisation workflows.
"""

__version__ = "0.2.0"
__author__ = "Clément Atlan"
__email__ = "clement.atlan@esrf.fr"
__license__ = "MIT"


import importlib

from .utils import (
    energy_to_wavelength,
    make_support,
    get_centred_slices,
    CroppingHandler
)

__submodules__ = {
    "utils",
    "geometry",
    "converter",
    "load",
    "process",
    "pipeline",
    "plot"
}

__class_submodules__ = {
    "Geometry": "geometry",
    "SpaceConverter": "converter",
    "BcdiPipeline": "pipeline",
    "Loader": "load",
}


__all__ = [
    "energy_to_wavelength", "make_support",
    "get_centred_slices", "CroppingHandler"
] + list(__submodules__) + list(__class_submodules__)


def __getattr__(name):
    # Lazy load submodules
    if name in __submodules__:
        return importlib.import_module(f"{__name__}.{name}")

    # Lazy load specific classes
    if name in __class_submodules__:
        submodule = importlib.import_module(
            f"{__name__}.{__class_submodules__[name]}"
        )
        return getattr(submodule, name)

    raise AttributeError(f"module {__name__} has no attribute {name}.")
