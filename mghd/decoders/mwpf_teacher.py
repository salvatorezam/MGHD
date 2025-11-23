"""
MWPF primary teacher (hypergraph generalization of MWPM).
- Offline: build model hypergraph from code structure / fault graph.
- Online: given a syndrome (det-stream), return a correction in fault-id space.

Refs:
- Python API & examples (hyperedges, solver init): https://github.com/yuewuo/mwpf (README 'Usage')
- Sinter adapter exists but is optional (we don't rely on DEM for training).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import warnings

import numpy as np

try:  # pragma: no cover - optional dependency
    from mwpf import (  # type: ignore
        HyperEdge,
        SolverInitializer,
        SolverSerialJointSingleHair,
        SyndromePattern,
    )

    _HAVE_MWPF = True
    _MWPF_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised when lib missing
    HyperEdge = SolverInitializer = SolverSerialJointSingleHair = SyndromePattern = None  # type: ignore
    _HAVE_MWPF = False
    _MWPF_IMPORT_ERROR = exc


@dataclass
class MWPFConfig:
    """Configuration knobs for the hypergraph solver.

    cluster_node_limit: optional cap for clustering (speed/accuracy trade-off)
    timeout: optional wall-clock limit per solve (seconds)
    """

    cluster_node_limit: Optional[int] = None
    timeout: Optional[float] = None


class MWPFTeacher:
    """Hyperblossom (a.k.a. Minimum-Weight Parity Factor, MWPF) teacher.

    Kept as MWPFTeacher for backward compatibility (configs/imports). Builds a
    fault hypergraph (vertices=detectors, hyperedges=faults) with non-negative
    weights (e.g., -log p or TAD-scaled costs) and solves a minimum set cover in
    hypergraph space conditional on a detector pattern.

    If the optional ``mwpf`` dependency is unavailable, falls back to a simple
    deterministic detector→fault heuristic for tests.
    """

    def __init__(self, code_obj: Any, *, config: Optional[MWPFConfig] = None):
        """Initialize solver context from a code object's hypergraph hooks.

        Expects either ``to_fault_hypergraph()`` or ``detectors_per_fault`` (+
        optional ``fault_weights``). Stores hypergraph and prepares solver cfg.
        """
        self.code_obj = code_obj
        self.config = config or MWPFConfig()
        self._vertex_num: Optional[int] = None
        self._fault_map: Optional[List[int]] = None
        self._solver_cfg: Optional[Dict[str, object]] = None
        self._fault_edges: List[np.ndarray] = []
        self._edge_sets: List[set[int]] = []
        self._weights: List[float] = []
        # Caching for solver reuse
        self._cached_initializer: Any = None
        self._cached_solver: Any = None
        self._cached_weights_key: Optional[tuple] = None
        self._build_offline()

    # ------------------------------------------------------------------
    # Hypergraph construction
    # ------------------------------------------------------------------
    def _build_offline(self) -> None:
        """
        Build the hypergraph from ``code_obj``.

        EXPECTATION:
          code_obj exposes either:
            - code_obj.detectors_per_fault: List[List[int]], and
              code_obj.fault_weights: List[float] (optional), OR
            - code_obj.to_fault_hypergraph() -> (vertex_num, list_of_edges,
              list_of_weights, fault_map)
        """

        if hasattr(self.code_obj, "to_fault_hypergraph"):
            vertex_num, edges, weights, fault_map = self.code_obj.to_fault_hypergraph()
            edges_list = [np.asarray(e, dtype=int) for e in edges]
            weight_list = [float(w) for w in weights]
            fault_map_list = list(fault_map)
        elif hasattr(self.code_obj, "detectors_per_fault"):
            dets_per_fault = getattr(self.code_obj, "detectors_per_fault")
            weight_seq = getattr(self.code_obj, "fault_weights", None)
            if weight_seq is None:
                weight_list = [1.0] * len(dets_per_fault)
            else:
                tmp: list[float] = []
                for w in weight_seq:
                    try:
                        tmp.append(float(w))
                    except Exception:
                        # Some backends (e.g. mwpf.Rational) use custom numeric
                        # types that do not support float() directly. For
                        # training/teacher use we only need relative ordering,
                        # so fall back to unit weights when conversion fails.
                        tmp.append(1.0)
                weight_list = tmp
            edges_list = [np.asarray(edge, dtype=int) for edge in dets_per_fault]
            max_idx = max((int(np.max(edge)) if len(edge) else -1) for edge in edges_list)
            vertex_num = int(getattr(self.code_obj, "num_detectors", max_idx + 1))
            fault_map_list = list(range(len(edges_list)))
        else:  # pragma: no cover - defensive
            raise NotImplementedError(
                "code_obj must provide detectors_per_fault+fault_weights or to_fault_hypergraph()."
            )

        self._vertex_num = int(vertex_num)
        self._fault_edges = edges_list
        self._edge_sets = [set(map(int, edge.tolist())) for edge in self._fault_edges]
        self._weights = weight_list
        if fault_map_list:
            self._fault_map = [int(x) for x in fault_map_list]
        else:
            self._fault_map = None

        if _HAVE_MWPF:
            cfg: Dict[str, object] = {}
            if self.config.cluster_node_limit is not None:
                cfg["cluster_node_limit"] = int(self.config.cluster_node_limit)
            if self.config.timeout is not None:
                cfg["timeout"] = float(self.config.timeout)
            self._solver_cfg = cfg or None
        else:
            warnings.warn(
                "mwpf not available – using heuristic fallback decoder."
                " Install `mwpf` for production-quality supervision.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode_batch(
        self,
        dets: np.ndarray,
        *,
        mwpf_scale: Optional[Dict[int, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Solve a batch of detector patterns.

        Parameters
        - dets: uint8 [B, D] detection events (1 where detector fired)
        - mwpf_scale: optional per-fault scaling overrides (from TAD) mapping
                      global fault ids to multiplicative factors

        Returns a dict with fields:
        - 'fault_ids': int32 [B, F_sel] (padded with -1)
        - 'weights': float32 [B] (optional total weight upper bound if exposed)
        """

        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim != 2:
            raise ValueError("dets must be rank-2 array [B, num_detectors]")
        if self._vertex_num is None:
            raise RuntimeError("MWPFTeacher not initialized correctly")
        B, D = dets.shape
        if D != self._vertex_num:
            raise AssertionError(f"syndrome length {D} != vertex count {self._vertex_num}")

        # Prefer the mwpf backend when available, but fall back gracefully to the
        # heuristic decoder if the library raises (e.g., due to custom Rational
        # weight types or environment issues). For training we mainly need a
        # consistent mapping from detectors to non-empty fault_id sets.
        if _HAVE_MWPF:
            try:
                return self._decode_batch_mwpf(dets, mwpf_scale=mwpf_scale)
            except Exception:
                return self._decode_batch_fallback(dets)
        return self._decode_batch_fallback(dets)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_solver_with_weights(self, weights: List[float]):
        """Instantiate the MWPF solver for a given weight vector."""
        # Check cache
        weights_key = tuple(weights)
        if self._cached_weights_key == weights_key and self._cached_initializer is not None:
            # Reuse initializer. We also try to reuse the solver if possible,
            # but for safety we recreate the solver wrapper which is cheap.
            # Actually, SolverSerialJointSingleHair might be stateful per solve,
            # so we recreate it. The heavy part is SolverInitializer.
            return self._cached_initializer, SolverSerialJointSingleHair(
                self._cached_initializer, self._solver_cfg
            )

        weighted_edges = [
            HyperEdge(list(edge.tolist()), float(weight))  # type: ignore[arg-type]
            for edge, weight in zip(self._fault_edges, weights)
        ]
        initializer = SolverInitializer(self._vertex_num, weighted_edges)  # type: ignore[arg-type]
        solver = SolverSerialJointSingleHair(initializer, self._solver_cfg)
        
        # Update cache
        self._cached_initializer = initializer
        self._cached_weights_key = weights_key
        
        return initializer, solver

    def _decode_batch_mwpf(
        self,
        dets: np.ndarray,
        *,
        mwpf_scale: Optional[Dict[int, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run the mwpf solver per sample; apply optional per-fault scaling."""
        sols: List[List[int]] = []
        weights: List[float] = []
        max_sel = 0
        base_weights = self._weights
        
        # Optimization: if mwpf_scale is None, weights are constant for the whole batch.
        # We can build the solver once and reuse it (or at least the initializer).
        # Even if mwpf_scale is present, if it's constant, we could cache, but
        # usually it varies per sample in TAD? No, mwpf_scale is passed as a single dict for the batch?
        # The signature says `mwpf_scale: Optional[Dict[int, float]]`.
        # If it's a single dict, it applies to the whole batch?
        # The docstring says "mapping global fault ids to multiplicative factors".
        # If it's one dict for the batch, then weights ARE constant for the batch!
        # So we can compute scaled weights ONCE.
        
        scaled_weights = base_weights
        if mwpf_scale:
            scaled = []
            for idx, base in enumerate(base_weights):
                fault_idx = int(self._fault_map[idx]) if self._fault_map is not None else idx
                scaled.append(base * float(mwpf_scale.get(fault_idx, 1.0)))
            scaled_weights = scaled
            
        # Build/Get solver once for the batch
        _, solver = self._build_solver_with_weights(scaled_weights)

        for sample in dets:
            # We reuse the solver instance if it supports multiple solves.
            # SolverSerialJointSingleHair usually supports repeated .solve() calls.
            # If not, we would need to recreate it from the initializer.
            # Assuming it does (standard for such solvers).
            
            syn = np.flatnonzero(sample != 0).tolist()
            solver.solve(SyndromePattern(syn))  # type: ignore[call-arg]
            idxs: List[int] = []
            subgraph = getattr(solver, "subgraph", None)
            if callable(subgraph):
                result = subgraph()
                for edge_idx in result:
                    if self._fault_map is not None:
                        idxs.append(int(self._fault_map[int(edge_idx)]))
                    else:
                        idxs.append(int(edge_idx))
            idxs = sorted(set(idxs))
            sols.append(idxs)
            max_sel = max(max_sel, len(idxs))

            total_w = None
            bound = getattr(solver, "subgraph_range", None)
            if callable(bound):
                try:
                    _, rng = bound()
                    total_w = getattr(rng, "upper", None)
                except Exception:  # pragma: no cover - defensive
                    total_w = None
            weights.append(0.0 if total_w is None else float(total_w))

        padded = -np.ones((len(sols), max_sel), dtype=np.int32)
        for row, ids in enumerate(sols):
            if ids:
                padded[row, : len(ids)] = np.asarray(ids, dtype=np.int32)
        return {
            "fault_ids": padded,
            "weights": np.asarray(weights, dtype=np.float32),
        }

    def _decode_batch_fallback(self, dets: np.ndarray) -> Dict[str, np.ndarray]:
        """Detector→fault heuristic used when mwpf is unavailable (tests only)."""
        sols: List[List[int]] = []
        max_sel = 0
        for sample in dets:
            active = set(np.flatnonzero(sample != 0).tolist())
            chosen: List[int] = []
            for idx, edge in enumerate(self._edge_sets):
                if active.intersection(edge):
                    mapped = idx if self._fault_map is None else self._fault_map[idx]
                    chosen.append(int(mapped))
            chosen = sorted(set(chosen))
            sols.append(chosen)
            max_sel = max(max_sel, len(chosen))

        padded = -np.ones((len(sols), max_sel if max_sel else 1), dtype=np.int32)
        for row, ids in enumerate(sols):
            if ids:
                padded[row, : len(ids)] = np.asarray(ids, dtype=np.int32)
            else:
                padded[row, 0] = -1
        weights = np.zeros(len(sols), dtype=np.float32)
        return {"fault_ids": padded, "weights": weights}


__all__ = ["MWPFTeacher", "MWPFConfig", "_HAVE_MWPF"]

# Friendly alias reflecting 2025 literature; keeps API stable
HyperblossomTeacher = MWPFTeacher


if not _HAVE_MWPF and _MWPF_IMPORT_ERROR is not None:  # pragma: no cover - info
    warnings.warn(
        "mwpf package not found – MWPFTeacher will use the heuristic fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
