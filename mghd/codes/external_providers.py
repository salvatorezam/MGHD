"""Optional external builders for CSS codes (requires extra dependencies)."""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


def build_color_666_qecsim(distance: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, object]]:
    """Return (Hx, Hz, n, layout) for triangular 6.6.6 color code using qecsim."""

    try:
        from qecsim.models.color import Color666Code
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("qecsim is required to build color_666 codes. Install `qecsim`." ) from exc

    code = Color666Code(distance)
    n, k, d = code.n_k_d
    if d != distance:
        raise RuntimeError(f"Unexpected distance from qecsim ({d}) for color_666 with distance={distance}.")

    stabilizers = code.stabilizers.astype(np.uint8)
    num_checks = len(code._plaquette_indices)
    Hx = stabilizers[:num_checks, :n].copy()
    Hz = stabilizers[num_checks:, n:].copy()
    layout = {
        "tiling": "6.6.6",
        "distance": distance,
        "n_formula": (3 * distance * distance + 1) // 4,
        "source": "qecsim",
    }
    return Hx, Hz, n, layout


def build_color_488_qecsim(distance: int) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, object]]:
    """Placeholder for triangular 4.8.8 color code generation."""

    raise RuntimeError(
        "color_488 generation requires cached matrices. Run `python -m tools.precompute_color_codes` "
        "after installing a library capable of producing triangular 4.8.8 color codes."
    )
