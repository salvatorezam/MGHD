"""
Sampler backends for generating detection events.

Priority order:
  1) CudaQSampler (primary, uses true circuit-level noise via CUDA-Q trajectories)
  2) StimSampler (optional, Pauli/twirled approximation for benchmarks)

Note:
  CUDA-Q trajectories simulate general (Kraus/coherent) noise at circuit level.
  Stim's fast path assumes Pauli channels + stabilizer ops; use only for A/B checks.
"""
from .registry import get_sampler, register_sampler
from .cudaq_sampler import CudaQSampler  # ensures availability
try:
    from .stim_sampler import StimSampler  # optional
except Exception:
    StimSampler = None  # noqa: N816

__all__ = ["get_sampler", "register_sampler", "CudaQSampler", "StimSampler"]
