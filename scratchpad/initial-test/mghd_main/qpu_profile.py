"""QPU profile loader for transpilation-aware decoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import json
import pathlib


@dataclass
class GateError:
    p_1q: Dict[str, float]
    p_2q: Dict[str, Dict[str, float]]
    p_meas: float
    t1_us: Dict[str, float]
    t2_us: Dict[str, float]
    idle_us: float
    crosstalk_pairs: Dict[str, float]


@dataclass
class QPUProfile:
    name: str
    n_qubits: int
    coupling: List[Tuple[int, int]]
    gate_error: GateError
    meta: Dict[str, Any]


def load_qpu_profile(path: str | pathlib.Path) -> QPUProfile:
    """Load a QPU calibration profile from JSON."""

    data = json.loads(pathlib.Path(path).read_text())
    ge = data.get("gate_error", {})
    gate_error = GateError(
        p_1q=ge.get("p_1q", {}),
        p_2q=ge.get("p_2q", {}),
        p_meas=float(ge.get("p_meas", 0.0)),
        t1_us=ge.get("t1_us", {}),
        t2_us=ge.get("t2_us", {}),
        idle_us=float(ge.get("idle_us", 0.0)),
        crosstalk_pairs=ge.get("crosstalk_pairs", {}),
    )
    coupling = [tuple(edge) for edge in data.get("coupling", [])]
    return QPUProfile(
        name=data.get("name", "unknown"),
        n_qubits=int(data.get("n_qubits", 0)),
        coupling=coupling,
        gate_error=gate_error,
        meta=data.get("meta", {}),
    )


__all__ = ["GateError", "QPUProfile", "load_qpu_profile"]
