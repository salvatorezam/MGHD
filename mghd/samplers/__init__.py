"""
Sampler backends for generating detection events.

Priority order:
  1) CudaQSampler (primary, uses true circuit-level noise via CUDA-Q trajectories)
  2) StimSampler (optional, Pauli/twirled approximation for benchmarks)

Note:
  CUDA-Q trajectories simulate general (Kraus/coherent) noise at circuit level.
  Stim's fast path assumes Pauli channels + stabilizer ops; use only for A/B checks.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

import numpy as np


@dataclass
class SampleBatch:
    """Container returned by samplers."""

    dets: np.ndarray  # uint8 [B, D]
    obs: np.ndarray  # uint8 [B, K]
    meta: dict[str, Any]
    erase_data_mask: np.ndarray | None = None  # uint8/bool [B, n]
    erase_det_mask: np.ndarray | None = None  # uint8/bool [B, D]
    p_erase_data: np.ndarray | None = None  # float [B, n]


_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_sampler(name: str, factory: Callable[..., object]) -> None:
    _REGISTRY[name] = factory


def get_sampler(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown sampler '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


from .cudaq_sampler import CudaQSampler  # ensures availability and registers itself

try:
    from .stim_sampler import StimSampler  # optional
except Exception:
    StimSampler = None  # noqa: N816


__all__ = [
    "get_sampler",
    "register_sampler",
    "CudaQSampler",
    "StimSampler",
    "SampleBatch",
]
