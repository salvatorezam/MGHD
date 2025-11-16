"""Minimal bench helpers for tests; prefer mghd.tools for real usage.

Exposes `run_distance` and a `sample_round` symbol (overridden by tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sample_round(d: int, p: float, seed: int, profile_path: str | None = None) -> dict[str, Any]:
    """Default sampler hooked for tests; overridden via monkeypatch in tests.

    In production, prefer mghd.qpu.adapters.garnet_adapter.sample_round.
    """
    from mghd.qpu.adapters.garnet_adapter import sample_round as _sr

    return _sr(d=d, p=p, seed=seed, profile_path=profile_path)


def run_distance(
    d: int,
    p: float,
    decoder: Any,
    Hx: Any,
    Hz: Any,
    rng: np.random.Generator,
    args: Any,
) -> dict[str, Any]:
    """Return simple stats for a distance sweep.

    This is a small test-oriented harness: it treats each shot as producing
    two logical checks (Z and X), so total shots=2*args.shots. The failure
    count is injected deterministically via `args.inject_ler_rate` for unit
    testing of downstream metrics.
    """
    shots = int(getattr(args, "shots", 1))
    total = 2 * shots
    rate = float(getattr(args, "inject_ler_rate", 0.0))
    failures = int(round(total * rate))
    return {"shots": total, "failures": failures}
