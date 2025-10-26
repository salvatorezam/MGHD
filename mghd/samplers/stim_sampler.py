"""Stim-based sampler aligned with DEM helper circuits."""
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
    """Stim sampler that mirrors the DEM construction for surface codes."""

    # Families this sampler can build Stim circuits for (extend as you implement more).
    _SUPPORTED = {"surface"}  # add "repetition", "steane", etc. when builders exist

    @classmethod
    def supports_family(cls, family: str) -> bool:
        """Return True iff this sampler can generate a Stim circuit for the family."""
        return family in cls._SUPPORTED

    def __init__(self, *, rounds: int = 5, dep: Optional[float] = None) -> None:
        self.rounds = int(rounds)
        self.dep = dep

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
        return SampleBatch(dets=dets, obs=obs, meta=meta)


register_sampler("stim", lambda **kw: StimSampler(**kw))


__all__ = ["StimSampler", "sample_surface_memory"]
