"""
Backward compatibility shim for mghd_main.

This module is deprecated. Please use 'mghd.core' instead.
"""
import warnings

warnings.warn(
    "Importing from 'mghd_main' is deprecated. Use 'from mghd.core import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mghd.core
from mghd.core import *
