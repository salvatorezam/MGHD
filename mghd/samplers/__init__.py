"""
Sampler backends for generating detection events.

Primary sampler: StimSampler (circuit-level DEM-based sampling via Stim).
"""

from dataclasses import dataclass
from typing import Any, Dict, Callable

import numpy as np


@dataclass
class SampleBatch:
    """Container returned by samplers."""

    dets: np.ndarray  # uint8 [B, D]
    obs: np.ndarray  # uint8 [B, K]
    meta: dict[str, Any]


_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_sampler(name: str, factory: Callable[..., object]) -> None:
    _REGISTRY[name] = factory


def get_sampler(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown sampler '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


try:
    from .stim_sampler import StimSampler
except Exception:
    StimSampler = None  # noqa: N816


__all__ = [
    "get_sampler",
    "register_sampler",
    "StimSampler",
    "SampleBatch",
]
