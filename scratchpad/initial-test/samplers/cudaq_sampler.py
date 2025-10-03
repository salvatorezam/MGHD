"""
CUDA-Q trajectory sampler (primary).
Bridges your existing CUDA-Q noise/circuit tooling into a simple interface.

Expected to find your files on PYTHONPATH:
  - cudaq_backend/* (noise model + circuit wrappers)
  - mghd_cluster_files/garnet_adapter.py (optional helpers)

If those live outside the repo, ensure PYTHONPATH includes their parent dirs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib
import os

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
        self.profile_kwargs = dict(profile_kwargs or {})
        env_mode = os.getenv("MGHD_MODE", "foundation")
        self.mode = str(self.profile_kwargs.pop("mode", env_mode)).lower()
        if self.mode not in {"foundation", "student"}:
            self.mode = "foundation"
        self.rounds = int(self.profile_kwargs.get("rounds", 3))
        # Lazy import of user's modules (keeps repo portable)
        self._gpu_noise = self._maybe_import("cudaq_backend.garnet_noise", "make_garnet_noise")
        # Optional adapter if user provides it (not required here)
        self._adapter = self._maybe_import("mghd_cluster_files.garnet_adapter", None)

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
        if n_shots <= 0:
            raise ValueError("n_shots must be positive")

        rng = np.random.default_rng(seed)
        Hx = getattr(code_obj, "Hx", None)
        Hz = getattr(code_obj, "Hz", None)
        if Hx is None or Hz is None:
            raise ValueError("code_obj must expose Hx and Hz for sampling")
        Hx = np.asarray(Hx, dtype=np.uint8)
        Hz = np.asarray(Hz, dtype=np.uint8)

        meta: Dict[str, Any] = {
            "device": self.device_profile,
            "shots": n_shots,
            "mode": self.mode,
        }
        if hasattr(code_obj, "distance") and getattr(code_obj, "distance") is not None:
            meta["distance"] = int(getattr(code_obj, "distance"))
        meta.update({k: v for k, v in self.profile_kwargs.items() if k not in {"rounds"}})

        try:
            if getattr(code_obj, "name", None) == "surface":
                dets, x_err, z_err = self._sample_surface(code_obj, n_shots, rng)
                meta["sampler"] = "cudaq_surface"
            elif getattr(code_obj, "name", None) == "repetition":
                dets, x_err, z_err = self._sample_repetition(code_obj, n_shots, rng)
                meta["sampler"] = "cudaq_repetition"
            else:
                dets, x_err, z_err = self._synthetic_css(code_obj, n_shots, rng)
                meta["sampler"] = "css_fallback"
        except Exception as exc:
            dets, x_err, z_err = self._synthetic_css(code_obj, n_shots, rng)
            meta["sampler"] = "css_fallback"
            meta["cudaq_error"] = str(exc)

        obs = self._logical_observables(code_obj, x_err, z_err)
        return SampleBatch(dets=dets.astype(np.uint8), obs=obs, meta=meta)

    # ------------------------------------------------------------------
    # CUDA-Q backed sampling helpers
    # ------------------------------------------------------------------

    def _surface_kwargs(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        if "phys_p" in self.profile_kwargs:
            extra["phys_p"] = self.profile_kwargs["phys_p"]
        if "noise_scale" in self.profile_kwargs:
            extra["noise_scale"] = self.profile_kwargs["noise_scale"]
        return extra

    def _sample_surface(self, code_obj: Any, n_shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from cudaq_backend.backend_api import cudaq_sample_surface_wrapper

        layout_info = getattr(code_obj, "layout", {})
        layout_dict = None
        surface_layout = "planar"
        if isinstance(layout_info, dict):
            layout_dict = layout_info.get("layout") or layout_info.get("meta")
            if layout_info.get("rotated"):
                surface_layout = "planar"  # keep planar packing for consistent decoding order

        packed = cudaq_sample_surface_wrapper(
            mode=self.mode,
            batch_size=n_shots,
            T=self.rounds,
            d=int(getattr(code_obj, "distance", 3) or 3),
            layout=layout_dict,
            rng=rng,
            bitpack=False,
            surface_layout=surface_layout,
            **self._surface_kwargs(),
        )

        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        num_x = Hx.shape[0]
        num_z = Hz.shape[0]
        n = Hx.shape[1]

        if packed.shape[1] < num_x + num_z:
            raise ValueError("cudaq surface sampler returned fewer columns than expected")

        synd_block = packed[:, : num_x + num_z]
        err_block = packed[:, num_x + num_z:]
        if err_block.shape[1] < n:
            err_block = np.pad(err_block, ((0, 0), (0, n - err_block.shape[1])), mode="constant")
        elif err_block.shape[1] > n:
            err_block = err_block[:, :n]

        sx = (synd_block[:, :num_x] & 1).astype(np.uint8)
        sz_raw = synd_block[:, num_x:num_x + num_z]
        sz = ((sz_raw >> 1) & 1).astype(np.uint8)

        dets_parts = []
        if num_x:
            dets_parts.append(sx)
        if num_z:
            dets_parts.append(sz)
        dets = np.concatenate(dets_parts, axis=1) if dets_parts else np.zeros((n_shots, 0), dtype=np.uint8)

        x_err = (err_block & 1).astype(np.uint8)
        z_err = ((err_block >> 1) & 1).astype(np.uint8)
        return dets, x_err, z_err

    def _sample_repetition(self, code_obj: Any, n_shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from cudaq_backend.backend_api import cudaq_sample_repetition_wrapper

        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        n_data = int(Hx.shape[1] or Hz.shape[1])
        layout = getattr(code_obj, "layout", None)
        layout_dict = layout if isinstance(layout, dict) else None

        packed = cudaq_sample_repetition_wrapper(
            mode=self.mode,
            batch_size=n_shots,
            n_data=n_data,
            T=self.rounds,
            layout=layout_dict,
            rng=rng,
            bitpack=False,
        )

        num_x = Hx.shape[0]
        num_z = Hz.shape[0]
        error_block = packed[:, -n_data:]
        synd_block = (packed[:, : packed.shape[1] - n_data] & 1).astype(np.uint8)

        if num_x and num_z:
            raise ValueError("Repetition sampler currently supports single-basis codes only")

        if num_x:
            sx = synd_block
            sz = np.zeros((n_shots, num_z), dtype=np.uint8)
        else:
            sx = np.zeros((n_shots, num_x), dtype=np.uint8)
            sz = synd_block

        dets_parts = []
        if num_x:
            dets_parts.append(sx)
        if num_z:
            dets_parts.append(sz)
        dets = np.concatenate(dets_parts, axis=1) if dets_parts else np.zeros((n_shots, 0), dtype=np.uint8)

        x_err = (error_block & 1).astype(np.uint8)
        z_err = ((error_block >> 1) & 1).astype(np.uint8)
        if x_err.shape[1] < Hx.shape[1]:
            pad = Hx.shape[1] - x_err.shape[1]
            x_err = np.pad(x_err, ((0, 0), (0, pad)), mode="constant")
            z_err = np.pad(z_err, ((0, 0), (0, pad)), mode="constant")
        elif Hx.shape[1] and x_err.shape[1] > Hx.shape[1]:
            x_err = x_err[:, :Hx.shape[1]]
            z_err = z_err[:, :Hx.shape[1]]
        return dets, x_err, z_err

    # ------------------------------------------------------------------
    # Synthetic fallback (keeps pipeline alive if CUDA-Q unavailable)
    # ------------------------------------------------------------------

    def _synthetic_css(self, code_obj: Any, n_shots: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        n = Hx.shape[1]
        px = float(self.profile_kwargs.get("p_x", 0.02))
        pz = float(self.profile_kwargs.get("p_z", 0.02))
        x_err = (rng.random((n_shots, n)) < px).astype(np.uint8)
        z_err = (rng.random((n_shots, n)) < pz).astype(np.uint8)

        parts = []
        if Hx.shape[0]:
            sx = (z_err @ (Hx.T % 2)) % 2
            parts.append(sx.astype(np.uint8))
        if Hz.shape[0]:
            sz = (x_err @ (Hz.T % 2)) % 2
            parts.append(sz.astype(np.uint8))
        dets = np.concatenate(parts, axis=1) if parts else np.zeros((n_shots, 0), dtype=np.uint8)
        return dets, x_err, z_err

    @staticmethod
    def _logical_observables(code_obj: Any, x_err: np.ndarray, z_err: np.ndarray) -> np.ndarray:
        parts = []
        Lz = getattr(code_obj, "Lz", None)
        if Lz is not None and np.size(Lz):
            Lz_arr = np.asarray(Lz, dtype=np.uint8)
            if Lz_arr.ndim == 1:
                Lz_arr = Lz_arr[np.newaxis, :]
            z_part = (x_err @ (Lz_arr.T % 2)) % 2
            parts.append(z_part.astype(np.uint8))

        Lx = getattr(code_obj, "Lx", None)
        if Lx is not None and np.size(Lx):
            Lx_arr = np.asarray(Lx, dtype=np.uint8)
            if Lx_arr.ndim == 1:
                Lx_arr = Lx_arr[np.newaxis, :]
            x_part = (z_err @ (Lx_arr.T % 2)) % 2
            parts.append(x_part.astype(np.uint8))

        if parts:
            return np.concatenate(parts, axis=1)
        return np.zeros((x_err.shape[0], 0), dtype=np.uint8)


# Register default
from .registry import register_sampler
register_sampler("cudaq", lambda **kw: CudaQSampler(**kw))
