"""
Backward compatibility shim for cudaq_backend.

This module is deprecated. Please use 'mghd.samplers.cudaq_backend' instead.
"""
import warnings

warnings.warn(
    "Importing from 'cudaq_backend' is deprecated. Use 'from mghd.samplers.cudaq_backend import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from mghd.samplers.cudaq_backend
from mghd.samplers.cudaq_backend import *
