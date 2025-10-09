"""
Backward compatibility shim for teachers.

This module is deprecated. Please use 'mghd.decoders' instead.
"""
import warnings

warnings.warn(
    "Importing from 'teachers' is deprecated. Use 'from mghd.decoders import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mghd.decoders
from mghd.decoders.lsd_teacher import LSDConfig, LSDTeacher
from mghd.decoders.mix import MixConfig, TeacherMix
from mghd.decoders.mwpf_teacher import MWPFConfig, MWPFTeacher
from mghd.decoders.mwpm_fallback import MWPMFallback

try:
    from mghd.decoders.erasure_surface_ml import ErasureSurfaceMLTeacher
except Exception as exc:
    ErasureSurfaceMLTeacher = None
    warnings.warn(
        f"ErasureSurfaceMLTeacher unavailable ({exc}); import will skip erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from mghd.decoders.erasure_peeling import ErasureQLDPCPeelingTeacher
except Exception as exc:
    ErasureQLDPCPeelingTeacher = None
    warnings.warn(
        f"ErasureQLDPCPeelingTeacher unavailable ({exc}); import will skip erasure teacher.",
        RuntimeWarning,
        stacklevel=2,
    )

try:
    from mghd.decoders.dem_matching import DEMMatchingTeacher
except Exception as exc:
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
