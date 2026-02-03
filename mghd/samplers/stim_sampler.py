"""Stim-based sampler aligned with DEM helper circuits.

Adds CUDA‑Q parity in interface:
- Optional omission of observables (``emit_obs=False``)
- Optional erasure injection on data qubits (``inject_erasure_frac>0``)
  returning ``erase_data_mask`` and ``p_erase_data`` fields in ``SampleBatch``.
- Circuit-level sampling with DEM graph extraction for MGHDCircuit.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple
from functools import lru_cache

import numpy as np

from . import SampleBatch
from . import register_sampler


@lru_cache(maxsize=32)
def _get_surface_circuit(d: int, rounds: int, dep: float):
    """Cache Stim circuit construction (expensive for large distances)."""
    import stim
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=int(d),
        rounds=int(rounds),
        after_clifford_depolarization=float(dep),
    )


def sample_surface_memory(
    *,
    d: int,
    rounds: int,
    shots: int,
    dep: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a rotated surface memory circuit via Stim's generator."""

    circuit = _get_surface_circuit(d, rounds, dep)
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

    def __init__(
        self,
        *,
        rounds: int = 5,
        dep: Optional[float] = None,
        emit_obs: bool = True,
        inject_erasure_frac: float = 0.0,
    ) -> None:
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
                    mask = rng.random((int(n_shots), n_data)) < self.inject_erasure_frac
                    erase_data_mask = mask.astype(np.uint8)
                    p_erase_data = np.full(
                        (int(n_shots), n_data), self.inject_erasure_frac, dtype=np.float32
                    )
                    meta.setdefault("erasure_frac", self.inject_erasure_frac)
        return SampleBatch(
            dets=dets.astype(np.uint8),
            obs=obs.astype(np.uint8),
            meta=meta,
            erase_data_mask=erase_data_mask,
            erase_det_mask=erase_det_mask,
            p_erase_data=p_erase_data,
        )


class StimCircuitSampler:
    """Circuit-level sampler that returns detection events + circuit for DEM extraction.

    This sampler is designed for use with MGHDCircuit, providing:
    - Detection events (temporal XOR of measurements)
    - Ground truth observable flips
    - Access to the underlying Stim circuit for DEM graph construction

    Unlike StimSampler which returns raw detectors, this provides the full
    context needed for circuit-level decoding.
    """

    def __init__(
        self,
        *,
        distance: int,
        rounds: int = 5,
        dep: float = 0.001,
    ) -> None:
        # Store as both public and private for curriculum detection
        self._distance = int(distance)
        self._rounds = int(rounds)
        self._dep = float(dep)
        self.distance = self._distance
        self.rounds = self._rounds
        self.dep = self._dep
        self._circuit = None
        self._sampler = None

    @property
    def circuit(self):
        """Lazily construct and cache the Stim circuit."""
        if self._circuit is None:
            self._circuit = _get_surface_circuit(self.distance, self.rounds, self.dep)
        return self._circuit

    @property
    def dem(self):
        """Get the Detector Error Model for graph construction."""
        return self.circuit.detector_error_model(decompose_errors=True)

    def sample(self, n_shots: int, seed: Optional[int] = None) -> dict:
        """Sample detection events and observable flips.

        Returns:
            dict with:
                - 'dets': [n_shots, num_detectors] binary detection events
                - 'obs': [n_shots, num_observables] ground truth observable flips
                - 'circuit': the Stim circuit (for DEM graph construction)
                - 'meta': sampling metadata
        """
        if self._sampler is None:
            self._sampler = self.circuit.compile_detector_sampler()

        dets, obs = self._sampler.sample(shots=n_shots, separate_observables=True)

        return {
            "dets": np.asarray(dets, dtype=np.uint8),
            "obs": np.asarray(obs, dtype=np.uint8),
            "circuit": self.circuit,
            "meta": {
                "distance": self.distance,
                "rounds": self.rounds,
                "dep": self.dep,
                "shots": n_shots,
                "num_detectors": self.dem.num_detectors,
                "num_observables": self.dem.num_observables,
            },
        }

    def sample_with_dem_graph(self, n_shots: int, seed: Optional[int] = None):
        """Sample and build DEMGraph for MGHDCircuit.

        Returns:
            DEMGraph ready for model forward pass
        """
        from mghd.core.core import build_dem_graph

        samples = self.sample(n_shots, seed)
        return build_dem_graph(
            samples["circuit"],
            samples["dets"],
            samples["obs"],
        )


register_sampler("stim", lambda **kw: StimSampler(**kw))
register_sampler("stim_circuit", lambda **kw: StimCircuitSampler(**kw))


__all__ = ["StimSampler", "StimCircuitSampler", "sample_surface_memory"]
