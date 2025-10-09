"""Context feature helpers for bandit weighting."""
from __future__ import annotations

import numpy as np
from typing import Dict


def context_vector(features: Dict[str, any], gate_vocab: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(gate_vocab) + 1, dtype=np.float32)
    for gate, count in features.get("gate_hist", {}).items():
        if gate in gate_vocab:
            vec[gate_vocab[gate]] = float(count)
    vec[-1] = float(features.get("n_2q_recent", 0))
    return vec


__all__ = ["context_vector"]
