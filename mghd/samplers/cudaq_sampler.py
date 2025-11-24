"""
CUDA-Q trajectory sampler (primary).
Bridges your existing CUDA-Q noise/circuit tooling into a simple interface.

Expected to find your files on PYTHONPATH:
  - cudaq_backend/* (noise model + circuit wrappers)
  - mghd_cluster_files/garnet_adapter.py (optional helpers)

If those live outside the repo, ensure PYTHONPATH includes their parent dirs.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np
import warnings

from . import SampleBatch
from . import register_sampler


# ---------------------------------------------------------------------------
# Local wrappers bridging CUDA-Q backend to sampler surface
# ---------------------------------------------------------------------------


def _try_cudaq_gpu_surface(
    d: int,
    batch_size: int,
    T: int,
    layout: dict[str, Any] | None,
    *,
    phys_p: float | None,
    noise_scale: float | None,
    profile_json: str | None,
):
    """
    Best-effort GPU sampling via CUDA-Q if target 'nvidia' is available.
    Falls back by raising on any error so the caller can use CPU path.
    """
    try:
        import cudaq
        import cudaq_qec as qec
        cudaq.set_target("nvidia")
    except Exception as exc:
        raise RuntimeError(f"cudaq GPU target unavailable: {exc}") from exc

    # Build code via CUDA-Q QEC helpers
    try:
        code = qec.qecrt.get_code("surface_code", distance=int(d))
    except Exception as exc:
        raise RuntimeError(f"cudaq_qec get_code failed: {exc}") from exc

    # Minimal noise model: map phys_p/noise_scale into depolarizing channels.
    # This is a coarse approximation; if more detailed mapping is needed,
    # extend this section. If no noise params are provided, use noiseless.
    noise = None
    try:
        if phys_p is not None or noise_scale is not None or profile_json is not None:
            p_base = float(phys_p) if phys_p is not None else 0.001
            if noise_scale is not None:
                try:
                    p_base *= float(noise_scale)
                except Exception:
                    pass
            noise = cudaq.NoiseModel()
            # Apply depolarizing channels to 1q and 2q gates as a rough proxy.
            noise.add_all_qubit_channel("x", cudaq.DepolarizationChannel(p_base))
            noise.add_all_qubit_channel("rx", cudaq.DepolarizationChannel(p_base))
            noise.add_all_qubit_channel("cx", cudaq.DepolarizationChannel(min(0.1, 2 * p_base)))
    except Exception:
        # If noise construction fails, fall back to noiseless
        noise = None

    try:
        res = qec.qecrt.sample_memory_circuit(code, int(batch_size), int(T), noise)
    except Exception as exc:
        raise RuntimeError(f"cudaq_qec.sample_memory_circuit failed: {exc}") from exc

    # sample_memory_circuit returns a tuple; we expect first element to be a 2D array
    if not isinstance(res, tuple) or len(res) == 0:
        raise RuntimeError("cudaq_qec.sample_memory_circuit returned unexpected format")
    arr = np.array(res[0])
    if arr.ndim != 2:
        raise RuntimeError(f"cudaq_qec.sample_memory_circuit returned array with ndim={arr.ndim}")
    return arr.astype(np.uint8)


def cudaq_sample_surface_wrapper(
    mode: str,
    batch_size: int,
    T: int = 3,
    d: int = 3,
    layout: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    bitpack: bool = False,
    surface_layout: str = "planar",
    profile_json: str | None = None,
    *,
    phys_p: float | None = None,
    noise_scale: float | None = None,
) -> np.ndarray:
    """Sample one CUDA-Q surface-code round with circuit-level noise.

    Parameters
    - mode: "foundation" or "student" profile selection
    - batch_size: number of trajectories to simulate
    - T: number of syndrome rounds (default 3)
    - d: surface-code distance
    - layout: precomputed layout dictionary (optional)
    - rng: NumPy Generator for reproducibility (optional)
    - bitpack: if True, backend may return packed bits (we normalize to uint8)
    - surface_layout: layout flavor (planar/rotated meta handling)
    - profile_json: optional QPU profile JSON path

    Returns
    - uint8 array [B, A_x + A_z + n] with [X-syndrome | Z-syndrome | data errors]
    """
    if mode not in ["foundation", "student"]:
        raise ValueError("Invalid mode: must be 'foundation' or 'student'")
    if layout is None:
        from mghd.samplers.cudaq_backend.circuits import make_surface_layout_general

        layout = make_surface_layout_general(d)
    if rng is None:
        rng = np.random.default_rng()
    from mghd.samplers.cudaq_backend.syndrome_gen import sample_surface_cudaq

    # Optional GPU path: guard with env to avoid surprising failures.
    use_gpu = os.getenv("MGHD_CUDAQ_GPU", "0") == "1"
    result = None
    if use_gpu:
        try:
            result = _try_cudaq_gpu_surface(
                d=d,
                batch_size=batch_size,
                T=T,
                layout=layout,
                phys_p=phys_p,
                noise_scale=noise_scale,
                profile_json=profile_json,
            )
        except Exception as exc:
            warnings.warn(f"CUDA-Q GPU sampler unavailable, falling back to CPU path ({exc})")
            result = None

    if result is None:
        result = sample_surface_cudaq(
            mode=mode,
            batch_size=batch_size,
            T=T,
            layout=layout,
            rng=rng,
            bitpack=bitpack,
            surface_layout=surface_layout,
            phys_p=phys_p,
            noise_scale=noise_scale,
            profile_json=profile_json,
        )
    expected_x = len(layout.get("ancilla_x", []))
    expected_z = len(layout.get("ancilla_z", []))
    expected_n = len(layout.get("data", []))
    expected_cols = expected_x + expected_z + expected_n
    if result.shape[1] != expected_cols:
        if result.shape[1] < expected_cols:
            pad = np.zeros((batch_size, expected_cols - result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, pad], axis=1)
        else:
            result = result[:, :expected_cols]
    return result.astype(np.uint8)


def cudaq_sample_repetition_wrapper(
    mode: str,
    batch_size: int,
    n_data: int = 5,
    T: int = 3,
    layout: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    bitpack: bool = False,
) -> np.ndarray:
    """Sample a repetition-code round via CUDA-Q backend.

    Returns uint8 [B, A + n] with [syndrome | data errors].
    """
    if mode not in ["foundation", "student"]:
        raise ValueError("Invalid mode: mode")
    if layout is None:
        layout = {
            "data": list(range(n_data)),
            "ancilla": list(range(n_data, n_data + n_data - 1)),
        }
    if rng is None:
        rng = np.random.default_rng()
    from mghd.samplers.cudaq_backend.syndrome_gen import sample_repetition_cudaq

    result = sample_repetition_cudaq(
        mode=mode,
        batch_size=batch_size,
        T=T,
        layout=layout,
        rng=rng,
        bitpack=bitpack,
    )
    expected_a = len(layout.get("ancilla", []))
    expected_n = len(layout.get("data", []))
    expected_cols = expected_a + expected_n
    if result.shape[1] != expected_cols:
        if result.shape[1] < expected_cols:
            pad = np.zeros((batch_size, expected_cols - result.shape[1]), dtype=result.dtype)
            result = np.concatenate([result, pad], axis=1)
        else:
            result = result[:, :expected_cols]
    return result.astype(np.uint8)


def _ensure_css_logicals(code: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Ensure code exposes Lx/Lz arrays, deriving them if necessary."""

    Lx = getattr(code, "Lx", None)
    Lz = getattr(code, "Lz", None)
    if (Lx is None or Lz is None) and hasattr(code, "derive_logicals"):
        try:
            Lx, Lz = code.derive_logicals()
            if Lx is not None:
                code.Lx = Lx
            if Lz is not None:
                code.Lz = Lz
        except Exception:
            pass
    return getattr(code, "Lx", None), getattr(code, "Lz", None)


def _obs_from_data_parities(code: Any, ex: np.ndarray, ez: np.ndarray) -> np.ndarray:
    """Project data-qubit error indicators onto logical observables.

    Multiplies data error indicators by logical operators (Lz with X errors,
    Lx with Z errors) to form a compact logical outcome vector [Z | X].
    """

    Lx, Lz = _ensure_css_logicals(code)
    shots = ex.shape[0]

    obs_z = np.zeros((shots, 0), dtype=np.uint8)
    if Lz is not None and np.size(Lz):
        Lz_arr = np.asarray(Lz, dtype=np.uint8)
        if Lz_arr.ndim == 1:
            Lz_arr = Lz_arr[np.newaxis, :]
        obs_z = (ex @ (Lz_arr.T & 1)) & 1

    obs_x = np.zeros((shots, 0), dtype=np.uint8)
    if Lx is not None and np.size(Lx):
        Lx_arr = np.asarray(Lx, dtype=np.uint8)
        if Lx_arr.ndim == 1:
            Lx_arr = Lx_arr[np.newaxis, :]
        obs_x = (ez @ (Lx_arr.T & 1)) & 1

    if obs_x.size and obs_z.size:
        return np.concatenate([obs_z, obs_x], axis=1).astype(np.uint8)
    if obs_z.size:
        return obs_z.astype(np.uint8)
    if obs_x.size:
        return obs_x.astype(np.uint8)
    return np.zeros((shots, 0), dtype=np.uint8)


class CudaQSampler:
    """CUDA-Q Monte-Carlo sampler for circuit-level noise.

    Generates detection events (Z then X ordering) and optional logical
    observables from user-provided code objects (must expose Hx/Hz and
    minimal layout metadata). Supports optional erasure injection.
    """

    def __init__(
        self,
        device_profile: str = "garnet",
        profile_kwargs: dict[str, Any] | None = None,
    ):
        self.device_profile = device_profile
        self.profile_kwargs = dict(profile_kwargs or {})
        env_mode = os.getenv("MGHD_MODE", "foundation")
        self.mode = str(self.profile_kwargs.pop("mode", env_mode)).lower()
        if self.mode not in {"foundation", "student"}:
            self.mode = "foundation"
        self.rounds = int(self.profile_kwargs.get("rounds", 3))
        self.inject_erasure_frac = float(self.profile_kwargs.pop("inject_erasure_frac", 0.0))
        self.emit_obs = bool(self.profile_kwargs.pop("emit_obs", True))
        # Lazy import of user's modules (keeps repo portable)
        self._gpu_noise = self._maybe_import("cudaq_backend.garnet_noise", "make_garnet_noise")
        # Optional adapter if user provides it (not required here)
        self._adapter = self._maybe_import("mghd_cluster_files.garnet_adapter", None)

    @staticmethod
    def _maybe_import(module_name: str, attr: str | None):
        """Best-effort import of optional backend modules."""
        try:
            mod = importlib.import_module(module_name)
            return getattr(mod, attr) if attr else mod
        except Exception:
            return None

    def sample(
        self,
        code_obj: Any,
        n_shots: int,
        seed: int | None = None,
    ) -> SampleBatch:
        """Sample detection events for a CSS-like code using CUDA-Q.

        Parameters
        - code_obj: code with Hx/Hz (and optional layout/circuit metadata)
        - n_shots: number of trajectories to simulate

        Returns
        - SampleBatch with dets [B, D], logical obs [B, K], and metadata.
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

        meta: dict[str, Any] = {
            "device": self.device_profile,
            "shots": n_shots,
            "mode": self.mode,
        }
        if hasattr(code_obj, "distance") and code_obj.distance is not None:
            meta["distance"] = int(code_obj.distance)
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

        if self.emit_obs:
            obs = _obs_from_data_parities(code_obj, x_err, z_err)
        else:
            obs = np.zeros((n_shots, 0), dtype=np.uint8)
        erase_data_mask, erase_det_mask, p_erase_data = self._build_erasure_masks(
            n_shots,
            dets.shape[1],
            Hx.shape[1],
            rng,
        )
        if self.inject_erasure_frac > 0:
            meta.setdefault("erasure_frac", self.inject_erasure_frac)
        return SampleBatch(
            dets=dets.astype(np.uint8),
            obs=obs,
            meta=meta,
            erase_data_mask=erase_data_mask,
            erase_det_mask=erase_det_mask,
            p_erase_data=p_erase_data,
        )

    def _build_erasure_masks(
        self,
        n_shots: int,
        num_detectors: int,
        num_data: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Construct per-shot erasure masks and probabilities.

        Returns (erase_data_mask [B,n], erase_det_mask [B,D], p_erase_data).
        """

        erase_data_mask = np.zeros((n_shots, num_data), dtype=np.uint8)
        erase_det_mask = np.zeros((n_shots, num_detectors), dtype=np.uint8)
        p_erase_data = None
        if self.inject_erasure_frac > 0:
            mask = rng.random((n_shots, num_data)) < self.inject_erasure_frac
            erase_data_mask = mask.astype(np.uint8)
            p_erase_data = np.full((n_shots, num_data), self.inject_erasure_frac, dtype=np.float32)
        return erase_data_mask, erase_det_mask, p_erase_data

    # ------------------------------------------------------------------
    # CUDA-Q backed sampling helpers
    # ------------------------------------------------------------------

    def _surface_kwargs(self) -> dict[str, Any]:
        """Extract sampler tuning fields for surface-code backend."""
        extra: dict[str, Any] = {}
        if "phys_p" in self.profile_kwargs:
            extra["phys_p"] = self.profile_kwargs["phys_p"]
        if "noise_scale" in self.profile_kwargs:
            extra["noise_scale"] = self.profile_kwargs["noise_scale"]
        return extra

    def _sample_surface(
        self, code_obj: Any, n_shots: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CUDA-Q sampling path for surface codes.

        Returns (dets, x_err, z_err). Detectors are ordered Z checks then X checks.
        """
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
        err_block = packed[:, num_x + num_z :]
        if err_block.shape[1] < n:
            err_block = np.pad(err_block, ((0, 0), (0, n - err_block.shape[1])), mode="constant")
        elif err_block.shape[1] > n:
            err_block = err_block[:, :n]

        sx = (synd_block[:, :num_x] & 1).astype(np.uint8)
        sz_raw = synd_block[:, num_x : num_x + num_z]
        sz = ((sz_raw >> 1) & 1).astype(np.uint8)

        # Canonical detector order: Z-checks first, then X-checks
        dets_parts = []
        if num_z:
            dets_parts.append(sz)
        if num_x:
            dets_parts.append(sx)
        dets = (
            np.concatenate(dets_parts, axis=1)
            if dets_parts
            else np.zeros((n_shots, 0), dtype=np.uint8)
        )

        x_err = (err_block & 1).astype(np.uint8)
        z_err = ((err_block >> 1) & 1).astype(np.uint8)
        return dets, x_err, z_err

    def _sample_repetition(
        self, code_obj: Any, n_shots: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CUDA-Q sampling path for repetition codes.

        Returns (dets, x_err, z_err). Detectors ordered Z then X when both exist.
        """
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

        # Canonical detector order: Z first, then X
        dets_parts = []
        if num_z:
            dets_parts.append(sz)
        if num_x:
            dets_parts.append(sx)
        dets = (
            np.concatenate(dets_parts, axis=1)
            if dets_parts
            else np.zeros((n_shots, 0), dtype=np.uint8)
        )

        x_err = (error_block & 1).astype(np.uint8)
        z_err = ((error_block >> 1) & 1).astype(np.uint8)
        if x_err.shape[1] < Hx.shape[1]:
            pad = Hx.shape[1] - x_err.shape[1]
            x_err = np.pad(x_err, ((0, 0), (0, pad)), mode="constant")
            z_err = np.pad(z_err, ((0, 0), (0, pad)), mode="constant")
        elif Hx.shape[1] and x_err.shape[1] > Hx.shape[1]:
            x_err = x_err[:, : Hx.shape[1]]
            z_err = z_err[:, : Hx.shape[1]]
        return dets, x_err, z_err

    # ------------------------------------------------------------------
    # Synthetic fallback (keeps pipeline alive if CUDA-Q unavailable)
    # ------------------------------------------------------------------

    def _synthetic_css(
        self, code_obj: Any, n_shots: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure-NumPy fallback approximating a CSS channel (Pauli, for tests)."""
        Hx = np.asarray(code_obj.Hx, dtype=np.uint8)
        Hz = np.asarray(code_obj.Hz, dtype=np.uint8)
        n = Hx.shape[1]
        px = float(self.profile_kwargs.get("p_x", 0.02))
        pz = float(self.profile_kwargs.get("p_z", 0.02))
        x_err = (rng.random((n_shots, n)) < px).astype(np.uint8)
        z_err = (rng.random((n_shots, n)) < pz).astype(np.uint8)

        # Canonical detector order: Z checks (from X errors) first, then X checks
        parts = []
        if Hz.shape[0]:
            sz = (x_err @ (Hz.T % 2)) % 2
            parts.append(sz.astype(np.uint8))
        if Hx.shape[0]:
            sx = (z_err @ (Hx.T % 2)) % 2
            parts.append(sx.astype(np.uint8))
        dets = np.concatenate(parts, axis=1) if parts else np.zeros((n_shots, 0), dtype=np.uint8)
        return dets, x_err, z_err

    @staticmethod
    def _logical_observables(code_obj: Any, x_err: np.ndarray, z_err: np.ndarray) -> np.ndarray:
        """Compute logical observables [Z | X] from data error indicators."""
        return _obs_from_data_parities(code_obj, x_err, z_err)


register_sampler("cudaq", lambda **kw: CudaQSampler(**kw))
