from __future__ import annotations

import sys
from types import ModuleType

import pytest


class _BlockPyMatching(ModuleType):
    def __getattr__(self, name: str):  # pragma: no cover - triggered on failure
        raise AssertionError("PyMatching should not be imported during CUDA-Q smoke")


@pytest.mark.skipif(sys.platform.startswith("win"), reason="CUDA-Q smoke relies on posix fallbacks")
def test_cudaq_smoke_runs_without_pymatching(monkeypatch):
    stub = _BlockPyMatching("pymatching")
    monkeypatch.setitem(sys.modules, "pymatching", stub)
    monkeypatch.setitem(sys.modules, "PyMatching", stub)

    from tools import train_core

    argv = [
        "train_core",
        "--families",
        "surface",
        "--distances",
        "5",
        "--sampler",
        "cudaq",
        "--shots-per-batch",
        "2",
        "--batches",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    train_core.main()

    assert sys.modules["pymatching"] is stub
