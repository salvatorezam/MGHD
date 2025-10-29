"""Stim-based sampler aligned with DEM helper circuits.

Adds CUDA‑Q parity in interface:
- Optional omission of observables (``emit_obs=False``)
- Optional erasure injection on data qubits (``inject_erasure_frac>0``)
  returning ``erase_data_mask`` and ``p_erase_data`` fields in ``SampleBatch``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from . import SampleBatch
from . import register_sampler


def sample_surface_memory(
    *,
    d: int,
    rounds: int,
    shots: int,
    dep: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a rotated surface memory circuit via Stim's generator."""

    import stim  # type: ignore

    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=int(d),
        rounds=int(rounds),
        after_clifford_depolarization=float(dep),
    )
    sampler = circuit.compile_detector_sampler()
    dets, obs = sampler.sample(shots=shots, separate_observables=True)
    return np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8)


class StimSampler:
    """Stim sampler that mirrors the DEM construction for surface codes.

    Parameters
    - rounds: number of memory rounds to generate in the synthetic builder
    - dep: after‑Clifford depolarizing probability for the builder
    - emit_obs: if False, return an empty observables array with shape [B,0]
    - inject_erasure_frac: if >0, inject per‑shot erasures on data qubits and
      return ``erase_data_mask`` (uint8/bool [B,n]) and ``p_erase_data`` (float [B,n]).
    """

    # Families this sampler can build Stim circuits for (extend as you implement more).
    _SUPPORTED = {"surface"}  # add "repetition", "steane", etc. when builders exist

    @classmethod
    def supports_family(cls, family: str) -> bool:
        """Return True iff this sampler can generate a Stim circuit for the family."""
        return family in cls._SUPPORTED

    def __init__(self, *, rounds: int = 5, dep: Optional[float] = None,
                 emit_obs: bool = True,
                 inject_erasure_frac: float = 0.0) -> None:
        self.rounds = int(rounds)
        self.dep = dep
        self.emit_obs = bool(emit_obs)
        self.inject_erasure_frac = float(inject_erasure_frac)

    def sample(self, code_obj: Any, n_shots: int, seed: Optional[int] = None) -> SampleBatch:
        import stim  # type: ignore

        distance = getattr(code_obj, "distance", None)
        code_name = str(getattr(code_obj, "name", "")).lower()
        dep = float(self.dep if self.dep is not None else 0.001)

        if "surface" in code_name and distance is not None:
            dets, obs = sample_surface_memory(
                d=int(distance),
                rounds=self.rounds,
                shots=n_shots,
                dep=dep,
            )
            meta_source = "stim.generated"
        else:
            circuit = getattr(code_obj, "stim_circuit", None)
            if circuit is None:
                raise RuntimeError(
                    "StimSampler requires either a surface-code distance or code_obj.stim_circuit"
                )
            sampler = circuit.compile_detector_sampler()
            dets, obs = sampler.sample(shots=n_shots, separate_observables=True)
            dets = np.asarray(dets, dtype=np.uint8)
            obs = np.asarray(obs, dtype=np.uint8)
            meta_source = "code_obj.stim_circuit"

        meta = {
            "source": meta_source,
            "distance": int(distance) if distance is not None else None,
            "rounds": self.rounds,
            "dep": dep,
            "shots": n_shots,
        }
        # Optionally drop observables for parity with CUDA‑Q paths that may omit them
        if not self.emit_obs:
            obs = np.zeros((int(n_shots), 0), dtype=np.uint8)
        # Optional erasure injection on data qubits
        erase_data_mask = None
        erase_det_mask = None
        p_erase_data = None
        if self.inject_erasure_frac > 0.0:
            Hx = getattr(code_obj, "Hx", None)
            if Hx is not None:
                try:
                    n_data = int(np.asarray(Hx, dtype=np.uint8).shape[1])
                except Exception:
                    n_data = 0
                if n_data > 0:
                    rng = np.random.default_rng(seed)
                    mask = (rng.random((int(n_shots), n_data)) < self.inject_erasure_frac)
                    erase_data_mask = mask.astype(np.uint8)
                    p_erase_data = np.full((int(n_shots), n_data), self.inject_erasure_frac, dtype=np.float32)
                    meta.setdefault("erasure_frac", self.inject_erasure_frac)
        return SampleBatch(
            dets=dets.astype(np.uint8),
            obs=obs.astype(np.uint8),
            meta=meta,
            erase_data_mask=erase_data_mask,
            erase_det_mask=erase_det_mask,
            p_erase_data=p_erase_data,
        )


register_sampler("stim", lambda **kw: StimSampler(**kw))


__all__ = ["StimSampler", "sample_surface_memory"]
