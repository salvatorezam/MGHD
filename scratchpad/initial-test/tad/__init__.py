"""
Backward compatibility shim for tad.

This module is deprecated. Please use 'mghd.tad' instead.
"""
import warnings

warnings.warn(
    "Importing from 'tad' is deprecated. Use 'from mghd.tad import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mghd.tad
from mghd.tad.weighting import logit_weight, schedule_to_weight_maps, feature_vector
from mghd.tad.context import context_vector

__all__ = [
    "logit_weight",
    "schedule_to_weight_maps",
    "feature_vector",
    "context_vector",
]
