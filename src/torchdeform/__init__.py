"""
torchdeform -- differentiable synthetic deformation and atmosphere in PyTorch.

A toolkit for generating synthetic geophysical InSAR data end-to-end and
differentiably, so it can be used both to build training datasets and as layers
inside a model:

* ``sources`` -- analytic deformation source models (Mogi, Okada finite fault,
  penny-shaped crack) producing ground displacement fields.
* ``observation`` -- line-of-sight geometry and the displacement <-> phase
  observation operators (Sentinel-1 conventions by default).
* ``atmosphere`` -- turbulent and topography-correlated atmospheric phase
  screens.
* ``simulation`` -- synthetic DEMs and parameter priors for randomised scenes.
* ``geometry`` / ``core`` -- WGS84 coordinate transforms and the shared
  tensor-backed data structures.

The most commonly used classes and functions are re-exported here at the package
top level; see the submodules for the full API.
"""
from .sources import (
    MogiSource,
    OkadaSource,
    OkadaSourceSimple,
    okada_params_from_fault,
    PennySource,
    PCDMSource,
)
from .observation import (
    # S1_INCIDENCE_RANGE_DEG,
    # S1_HEADING_ASCENDING_DEG,
    # S1_HEADING_DESCENDING_DEG,
    # S1_LOOK_SIDE,
    # sample_s1_geometry,
    los_vector,
    los_vector_per_pixel,
    los_vector_from_center,
    los_vector_from_center_curved,
    los_vector_from_satellite,
)
from .core import (
    Displacement,
    LOSVector,
    ECEF,
    Geodetic,
)
from .geometry import (
    geodetic_to_ecef,
    ecef_to_geodetic,
    ecef_to_local_enu,
    local_enu_to_ecef,
    geodetic_to_local_enu,
    local_enu_to_geodetic,
)

__all__ = [
    "MogiSource",
    "OkadaSource",
    "OkadaSourceSimple",
    "okada_params_from_fault",
    "PennySource",
    "PCDMSource",

    # "S1_INCIDENCE_RANGE_DEG",
    # "S1_HEADING_ASCENDING_DEG",
    # "S1_HEADING_DESCENDING_DEG",
    # "S1_LOOK_SIDE",
    # "sample_s1_geometry",
    "los_vector",
    "los_vector_per_pixel",
    "los_vector_from_center",
    "los_vector_from_center_curved",
    "los_vector_from_satellite",

    "Displacement",
    "LOSVector",
    "ECEF",
    "Geodetic",

    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_local_enu",
    "local_enu_to_ecef",
    "geodetic_to_local_enu",
    "local_enu_to_geodetic",
]

__version__ = "0.1.0"