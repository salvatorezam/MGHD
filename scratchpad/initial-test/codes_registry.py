"""Compatibility wrapper for legacy `codes_registry` imports.

We lazy-load the canonical implementation from `mghd_main.codes_registry` to
keep optional dependencies away from import-time in light-weight CI checks."""
from __future__ import annotations
import importlib
from types import ModuleType
from typing import Any

_lazy_mod: ModuleType | None = None


def _load() -> ModuleType:
    global _lazy_mod
    if _lazy_mod is None:
        _lazy_mod = importlib.import_module("mghd_main.codes_registry")
    return _lazy_mod


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough
    return getattr(_load(), name)


def __dir__() -> list[str]:  # pragma: no cover - passthrough
    return sorted(set(dir(_load())))
