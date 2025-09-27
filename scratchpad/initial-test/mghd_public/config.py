"""Public MGHD configuration dataclass for inference builds."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class MGHDConfig:
    """Minimal configuration needed to rebuild the S-profile MGHD model."""

    gnn: Dict[str, Any]
    mamba: Dict[str, Any]
    profile: str = "S"
    dist: int = 3
    n_qubits: int = 9
    n_checks: int = 8
    n_node_inputs: int = 9
    n_node_outputs: int = 2  # binary head for rotated d=3

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
