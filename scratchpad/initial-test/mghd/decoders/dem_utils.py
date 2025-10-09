"""Helpers for constructing and caching detector error models (DEMs)."""
from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Dict, Optional


def _hash_profile(profile: Dict[str, Any]) -> str:
    payload = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()[:10]


def dem_cache_path(
    cache_dir: str,
    family: str,
    distance: int,
    rounds: int,
    profile: Dict[str, Any],
) -> str:
    digest = _hash_profile(profile)
    root = pathlib.Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    return str(root / f"dem_{family}_d{distance}_r{rounds}_{digest}.dem")


def build_surface_memory_dem(
    *,
    distance: int,
    rounds: int,
    profile: Dict[str, Any],
    decompose: bool = True,
):
    """Construct a rotated surface-code memory DEM via Stim generated circuits."""

    import stim  # type: ignore

    gate_err = profile.get("gate_error", {}) if profile else {}
    dep = float(gate_err.get("after_clifford_depolarization", 0.001))
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=dep,
    )
    return circuit.detector_error_model(decompose_errors=bool(decompose))


def build_hgp_dem(*_args: Any, **_kwargs: Any):  # pragma: no cover - placeholder
    raise NotImplementedError("HGP DEM construction not implemented yet")


__all__ = [
    "_hash_profile",
    "dem_cache_path",
    "build_surface_memory_dem",
    "build_hgp_dem",
]
