"""QPU profile loader for transpilation‑aware decoding.

This module defines a typed view of calibration/architecture information used
by TAD (Teacher‑Assisted Decoding) to bias supervision and to provide compact
context vectors to the model. Profiles are typically authored as JSON files
checked into the repo under ``mghd/qpu/profiles/*.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import json
import pathlib


@dataclass
class GateError:
    """Per‑gate error and timing information.

    Fields
    - p_1q: per‑qubit single‑qubit error rates (e.g., {'q0': 1e-3, ...})
    - p_2q: nested error rates for pairs (e.g., {'cx': {'0-1': 2e-3, ...}})
    - p_meas: average readout error probability
    - t1_us/t2_us: per‑qubit relaxation/dephasing time estimates (microseconds)
    - idle_us: typical idle duration per scheduling tick (microseconds)
    - crosstalk_pairs: optional coupling weights for noisy pairs
    """

    p_1q: Dict[str, float]
    p_2q: Dict[str, Dict[str, float]]
    p_meas: float
    t1_us: Dict[str, float]
    t2_us: Dict[str, float]
    idle_us: float
    crosstalk_pairs: Dict[str, float]


@dataclass
class QPUProfile:
    """Unified profile object used by samplers and TAD.

    Fields
    - name: human‑readable device name/identifier
    - n_qubits: number of physical qubits
    - coupling: undirected coupling graph as a list of (u, v) edges
    - gate_error: GateError container with error/timing metadata
    - meta: freeform extras (e.g., vendor, date, calibration notes)
    """

    name: str
    n_qubits: int
    coupling: List[Tuple[int, int]]
    gate_error: GateError
    meta: Dict[str, Any]


def load_qpu_profile(path: str | pathlib.Path) -> QPUProfile:
    """Load a QPU calibration profile from a JSON file.

    The JSON schema is intentionally flexible; unknown fields are stored under
    ``meta`` for downstream consumers. Known top‑level keys include
    ``name``, ``n_qubits``, ``coupling``, and a nested ``gate_error`` object.
    """

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
