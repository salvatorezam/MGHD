"""
CUDA-Q Backend for Fast Batched Monte-Carlo Trajectories Simulation
Mirroring IQM Garnet Circuit-Level Noise Model
"""

__version__ = "1.0.0"
__author__ = "MGHD Team"

from .garnet_noise import (
    FOUNDATION_DEFAULTS,
    GARNET_COUPLER_F2,
    STUDENT_DEFAULTS,
    GarnetFoundationPriors,
    GarnetStudentCalibration,
    GarnetNoiseModel
)

from .circuits import (
    build_round_repetition,
    build_round_surface,
    build_round_bb
)

from .syndrome_gen import (
    sample_surface_cudaq,
    sample_bb_cudaq,
    sample_repetition_cudaq
)

from .backend_api import (
    cudaq_sample_surface_wrapper,
    cudaq_sample_bb_wrapper,
    cudaq_sample_repetition_wrapper,
    get_backend_info,
    validate_backend_installation,
    sample_surface_foundation,
    sample_surface_student,
    sample_bb_foundation,
    sample_bb_student,
    sample_repetition_foundation,
    sample_repetition_student
)

__all__ = [
    "FOUNDATION_DEFAULTS",
    "GARNET_COUPLER_F2", 
    "STUDENT_DEFAULTS",
    "GarnetFoundationPriors",
    "GarnetStudentCalibration",
    "GarnetNoiseModel",
    "build_round_repetition",
    "build_round_surface", 
    "build_round_bb",
    "sample_surface_cudaq",
    "sample_bb_cudaq",
    "sample_repetition_cudaq",
    "cudaq_sample_surface_wrapper",
    "cudaq_sample_bb_wrapper", 
    "cudaq_sample_repetition_wrapper",
    "get_backend_info",
    "validate_backend_installation",
    "sample_surface_foundation",
    "sample_surface_student",
    "sample_bb_foundation",
    "sample_bb_student",
    "sample_repetition_foundation",
    "sample_repetition_student"
]
