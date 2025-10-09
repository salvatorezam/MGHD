"""Tests for 4.8.8 color-code providers and caches."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


def _css_ok(Hx: np.ndarray, Hz: np.ndarray) -> bool:
    return not ((Hx.astype(np.uint8) @ Hz.astype(np.uint8).T) % 2).any()


def _build_or_skip(distance: int = 3):
    try:
        provider = importlib.import_module("mghd_main.codes_external_488")
    except ImportError:
        pytest.skip("codes_external_488 not available")
    try:
        return provider.build_color_488(distance)
    except ImportError:
        pytest.skip("No 4.8.8 provider installed (install panqec or quantum-pecos)")


def test_color_488_builder_css():
    Hx, Hz, n, layout = _build_or_skip()
    assert Hx.shape[1] == n
    assert Hz.shape[1] == n
    assert layout.get("tiling") == "4.8.8"
    assert _css_ok(Hx, Hz)


@pytest.mark.skipif(
    not (Path("color_cache/color_488_d3.npz").exists() or Path("data/color_488_d3.npz").exists()),
    reason="color_488 cache missing",
)
def test_color_488_cache_css_and_shape():
    path = Path("data/color_488_d3.npz")
    if not path.exists():
        path = Path("color_cache/color_488_d3.npz")
    data = np.load(path)
    Hx = data["Hx"]
    Hz = data["Hz"]
    assert Hx.shape[1] == Hz.shape[1]
    assert _css_ok(Hx, Hz)
