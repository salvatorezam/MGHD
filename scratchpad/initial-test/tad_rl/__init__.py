"""
Backward compatibility shim for tad_rl.

This module is deprecated. Please use 'mghd.tad.rl' instead.
"""
import warnings

warnings.warn(
    "Importing from 'tad_rl' is deprecated. Use 'from mghd.tad.rl import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mghd.tad.rl
from mghd.tad.rl.lin_ts import LinTSBandit

__all__ = ["LinTSBandit"]
