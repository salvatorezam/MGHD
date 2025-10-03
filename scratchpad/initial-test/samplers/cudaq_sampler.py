"""
CUDA-Q trajectory sampler (primary).
Bridges your existing CUDA-Q noise/circuit tooling into a simple interface.

Expected to find your files on PYTHONPATH:
  - cudaq_backend/garnet_noise.py (or similar)
  - mghd_cluster_files/garnet_adapter.py (if you have helper adapters)

If those live outside the repo, ensure PYTHONPATH includes their parent dirs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib

import numpy as np


@dataclass
class SampleBatch:
    dets: np.ndarray     # uint8 [B, D]
    obs: np.ndarray      # uint8 [B, K]
    meta: Dict[str, Any] # e.g., {"code": name, "d": distance, ...}


class CudaQSampler:
    """
    Uses CUDA-Q's Monte-Carlo trajectory method to sample detection events
    under general circuit-level noise (Kraus/coherent supported).
    """

    def __init__(self,
                 device_profile: str = "garnet",
                 profile_kwargs: Optional[Dict[str, Any]] = None):
        self.device_profile = device_profile
        self.profile_kwargs = profile_kwargs or {}
        # Lazy import of user's modules (keeps repo portable)
        self._gpu_noise = self._maybe_import("cudaq_backend.garnet_noise", "make_garnet_noise")
        # Optional adapter if user provides it (not required here)
        self._adapter   = self._maybe_import("mghd_cluster_files.garnet_adapter", None)

    @staticmethod
    def _maybe_import(module_name: str, attr: Optional[str]):
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr) if attr else mod
        except Exception:
            return None

    def sample(self,
               code_obj: Any,
               n_shots: int,
               seed: Optional[int] = None) -> SampleBatch:
        """
        Args:
          code_obj: an object from codes_registry (must provide a circuit builder
                    or matrices & metadata sufficient for CUDA-Q construction).
          n_shots:  number of trajectories to sample.
        Returns:
          SampleBatch(dets, obs, meta)
        """
        # --- Pseudocode placeholder: call into your CUDA-Q path.
        # Replace this with your existing build+simulate routine.
        # dets, obs = cudaq_run_trajectories(code_obj, self._gpu_noise, n_shots, seed, **self.profile_kwargs)
        D = getattr(code_obj, "num_detectors", 64)
        K = getattr(code_obj, "num_observables", 1)
        rng = np.random.default_rng(seed)
        dets = rng.integers(0, 2, size=(n_shots, D), dtype=np.uint8)
        obs  = rng.integers(0, 2, size=(n_shots, K), dtype=np.uint8)
        meta = {"device": self.device_profile, "shots": n_shots}
        return SampleBatch(dets=dets, obs=obs, meta=meta)


# Register default
from .registry import register_sampler
register_sampler("cudaq", lambda **kw: CudaQSampler(**kw))
