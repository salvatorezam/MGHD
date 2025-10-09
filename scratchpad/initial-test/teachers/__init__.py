"""Teacher package exposing MWPF/LSD/MWPM helpers and mixers."""

import warnings

from .lsd_teacher import LSDConfig, LSDTeacher
from .mix import MixConfig, TeacherMix
from .mwpf_teacher import MWPFConfig, MWPFTeacher
from .mwpm_fallback import MWPMFallback

try:
    from .erasure_surface_ml import ErasureSurfaceMLTeacher
except Exception as exc:  # pragma: no cover - optional dependency stack
    ErasureSurfaceMLTeacher = None
    warnings.warn(
        f"ErasureSurfaceMLTeacher unavailable ({exc}); import will skip erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .erasure_peeling import ErasureQLDPCPeelingTeacher
except Exception as exc:  # pragma: no cover - optional dependency stack
    ErasureQLDPCPeelingTeacher = None
    warnings.warn(
        f"ErasureQLDPCPeelingTeacher unavailable ({exc}); import will skip erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from .dem_matching import DEMMatchingTeacher
except Exception as exc:  # pragma: no cover - optional dependency stack
    DEMMatchingTeacher = None
    warnings.warn(
        f"DEMMatchingTeacher unavailable ({exc}); install pymatching for DEM decoding.",
        RuntimeWarning,
        stacklevel=2,
    )

__all__ = [
    "LSDConfig",
    "LSDTeacher",
    "MixConfig",
    "TeacherMix",
    "MWPFConfig",
    "MWPFTeacher",
    "MWPMFallback",
]

if ErasureSurfaceMLTeacher is not None:
    __all__.append("ErasureSurfaceMLTeacher")
if ErasureQLDPCPeelingTeacher is not None:
    __all__.append("ErasureQLDPCPeelingTeacher")
if "DEMMatchingTeacher" in globals() and DEMMatchingTeacher is not None:
    __all__.append("DEMMatchingTeacher")
