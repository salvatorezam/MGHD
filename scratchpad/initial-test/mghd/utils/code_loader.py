"""
Tiny, defensive code loader. We try a few common entry points in codes_registry:
  - get_code(name, **kw)
  - registry dict lookups
  - build_<family>(d=...)
Returned object should expose, when available:
  - Hx, Hz (uint8)
  - num_detectors, num_observables (for MWPF/det-stream)
  - optional mapping: detectors_per_fault / to_fault_hypergraph()
  - optional: stim_circuit (benchmarks only)
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import importlib


def load_code(family: str, distance: int, **kw) -> Any:
    cr = importlib.import_module("codes_registry")
    # 1) get_code
    if hasattr(cr, "get_code"):
        try:
            raw = cr.get_code(family, distance=distance, **kw)
            return _wrap_code(raw, family)
        except Exception:
            pass
    # 2) registry dicts
    for attr in ("REGISTRY", "registry", "CODES"):
        if hasattr(cr, attr):
            reg = getattr(cr, attr)
            if isinstance(reg, dict) and family in reg:
                builder = reg[family]
                try:
                    raw = builder(distance=distance, **kw)
                    return _wrap_code(raw, family)
                except Exception:
                    pass
    # 3) build_<family>
    cand = f"build_{family.lower()}"
    if hasattr(cr, cand):
        raw = getattr(cr, cand)(distance=distance, **kw)
        return _wrap_code(raw, family)
    # 4) Extended builder names
    for alt in (
        f"build_{family.lower()}_rotated_H",
        f"build_{family.lower()}_H",
        f"build_{family.lower()}_code",
    ):
        if hasattr(cr, alt):
            builder = getattr(cr, alt)
            try:
                raw = builder(distance)
            except TypeError:
                raw = builder(distance=distance)
            return _wrap_code(raw, family)
    # 5) Spec helpers (e.g., surface_rotated_spec)
    spec_name = f"{family.lower()}_spec"
    if hasattr(cr, spec_name):
        raw = getattr(cr, spec_name)(distance)
        return _wrap_code(raw, family)
    raise RuntimeError(
        f"Could not load code family='{family}' distance={distance}. "
        "Ensure codes_registry exposes get_code or build_<family>."
    )


def _wrap_code(raw: Any, family: str) -> Any:
    """Normalize raw outputs into an object exposing Hx/Hz."""

    if raw is None:
        return raw
    if isinstance(raw, tuple) and len(raw) >= 2:
        hx, hz = raw[:2]
        meta = raw[2] if len(raw) > 2 else {}
        return SimpleNamespace(name=getattr(raw, "name", family), Hx=hx, Hz=hz, meta=meta)

    if hasattr(raw, "Hx") and hasattr(raw, "Hz"):
        return raw

    hx = getattr(raw, "hx", None)
    hz = getattr(raw, "hz", None)
    if hx is not None and hz is not None:
        wrapped = SimpleNamespace(
            name=getattr(raw, "name", family),
            Hx=hx,
            Hz=hz,
            meta=getattr(raw, "meta", {}),
        )
        for attr in (
            "num_detectors",
            "num_observables",
            "detectors_per_fault",
            "fault_weights",
            "to_fault_hypergraph",
        ):
            if hasattr(raw, attr):
                setattr(wrapped, attr, getattr(raw, attr))
        return wrapped
    return raw


__all__ = ["load_code"]
