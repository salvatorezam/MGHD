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
    GarnetNoiseModel,
)

from .circuits import build_round_repetition, build_round_surface, build_round_bb

from .syndrome_gen import sample_surface_cudaq, sample_bb_cudaq, sample_repetition_cudaq

# Wrapper functions are exposed via mghd.samplers.cudaq_sampler to reduce coupling.

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
]
