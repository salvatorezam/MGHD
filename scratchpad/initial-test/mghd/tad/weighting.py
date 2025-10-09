"""Utilities for schedule-aware weighting and context features."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def logit_weight(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log((1.0 - p) / p)


def schedule_to_weight_maps(
    schedule_ir: List[Tuple[int, str, Tuple[int, ...], Any]],
    profile,
    n_qubits: int,
) -> Dict[str, Dict[int, Dict[Any, float]]]:
    """Convert schedule and calibration profile to per-qubit/pair weight maps."""

    w_qubit: Dict[int, Dict[int, float]] = {}
    w_pair: Dict[int, Dict[Tuple[int, int], float]] = {}

    for t, gate, qubits, _ in schedule_ir:
        if not qubits:
            continue
        if len(qubits) == 1:
            p = profile.gate_error.p_1q.get(gate, 0.0)
            idx = int(qubits[0])
            w_qubit.setdefault(t, {})
            w_qubit[t][idx] = w_qubit[t].get(idx, 0.0) + logit_weight(p)
        elif len(qubits) == 2:
            pair = tuple(sorted(int(q) for q in qubits))
            default = profile.gate_error.p_2q.get(gate, {}).get("default", 0.0)
            p = profile.gate_error.p_2q.get(gate, {}).get(str(tuple(pair)), default)
            w_pair.setdefault(t, {})
            w_pair[t][pair] = w_pair[t].get(pair, 0.0) + logit_weight(p)
    return {"w_qubit": w_qubit, "w_pair": w_pair}


def feature_vector(
    schedule_ir: List[Tuple[int, str, Tuple[int, ...], Any]],
    window: int = 8,
) -> Dict[str, Any]:
    from collections import Counter

    tail = schedule_ir[-window:] if window > 0 else schedule_ir
    gate_hist = Counter(g for _, g, _, _ in tail)
    n_two_qubit = sum(1 for _, _, qs, _ in tail if len(qs) == 2)
    return {"gate_hist": dict(gate_hist), "n_2q_recent": n_two_qubit}


__all__ = ["logit_weight", "schedule_to_weight_maps", "feature_vector"]
