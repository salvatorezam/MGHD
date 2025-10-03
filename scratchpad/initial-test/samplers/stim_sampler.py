"""
Stim-based sampler (optional; for benchmarks only).

WARNING:
  This approximates non-Pauli noise (e.g., amplitude damping, coherent errors)
  via Pauli-twirled/Pauli-channel surrogates. Use for cross-checks or to plug
  into Stim/Sinter-centric tooling, not for 'actual circuit-level' training.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class SampleBatch:
    dets: np.ndarray
    obs: np.ndarray
    meta: Dict[str, Any]


class StimSampler:
    def __init__(self, approx_profile: str = "pauli_twirled", profile_kwargs: Optional[Dict[str, Any]] = None):
        self.approx_profile = approx_profile
        self.profile_kwargs = profile_kwargs or {}

    def sample(self, code_obj: Any, n_shots: int, seed: Optional[int] = None) -> SampleBatch:
        import stim
        import sinter
        rng = np.random.default_rng(seed)
        # Expect the code_obj to expose a stim circuit for benchmarks
        if not hasattr(code_obj, "stim_circuit") or code_obj.stim_circuit is None:
            raise RuntimeError("StimSampler requires code_obj.stim_circuit for benchmarking.")
        circuit = code_obj.stim_circuit
        # Use sinter to sample detections quickly
        dets = np.empty((n_shots, circuit.num_detectors), dtype=np.uint8)
        obs  = np.empty((n_shots, circuit.num_observables), dtype=np.uint8)
        # Placeholder fast path (replace with sinter.sample when wiring benchmarks)
        dets[:] = rng.integers(0, 2, size=dets.shape)
        obs[:]  = rng.integers(0, 2, size=obs.shape)
        meta = {"approx": self.approx_profile, "shots": n_shots}
        return SampleBatch(dets=dets, obs=obs, meta=meta)


# Register optional
from .registry import register_sampler
register_sampler("stim", lambda **kw: StimSampler(**kw))
