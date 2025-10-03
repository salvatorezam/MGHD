"""Compatibility wrapper for legacy `core` imports.

This module defers loading the heavy `mghd_main.core` implementation until one
of its attributes is accessed. That keeps import-time dependencies (e.g. torch)
optional for lightweight checks like pytest smoke tests."""
from __future__ import annotations
import importlib
from types import ModuleType
from typing import Any

_lazy_mod: ModuleType | None = None


def _load() -> ModuleType:
    global _lazy_mod
    if _lazy_mod is None:
        _lazy_mod = importlib.import_module("mghd_main.core")
    return _lazy_mod


def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough
    return getattr(_load(), name)


def __dir__() -> list[str]:  # pragma: no cover - passthrough
    return sorted(set(dir(_load())))
