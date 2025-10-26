"""Core MGHD v2 implementation.

This package exposes the consolidated runtime from `core.py`. Other legacy
modules remain available as separate imports but are not pulled in here to
avoid unnecessary dependencies.
"""

from .core import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
