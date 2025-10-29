"""Optional CC-style distance metrics for DEM-based clustering.

Implements:
  - 3D Manhattan distance (phenomenological noise)
  - 4D L1 embedding for circuit-level surfaces with hook edges (Eq. (E4))

References
- Ben Barber et al., "Real-time decoder ... Collision Clustering" (Supplementary),
  Appendix E: coordinate embedding and distance function (Eq. E4).
"""
from __future__ import annotations

from typing import Tuple


def manhattan_3d(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    """Return 3D Manhattan distance between lattice-time coordinates.

    Parameters
    - a, b: tuples (x1, x2, t) on the rotated planar space–time grid
    """
    (x1a, x2a, ta), (x1b, x2b, tb) = a, b
    return abs(x1a - x1b) + abs(x2a - x2b) + abs(ta - tb)


def cc_embed_4d(x1: int, x2: int, t: int) -> Tuple[float, float, float, float]:
    """Riverlane-style 4D embedding for circuit-level surfaces (Eq. E4).

    Maps (x1, x2, t) → (x1/2, x2/2, (x1+t)/2, (x2+t)/2).
    """

    return (0.5 * x1, 0.5 * x2, 0.5 * (x1 + t), 0.5 * (x2 + t))


def l1_4d(a4: Tuple[float, float, float, float], b4: Tuple[float, float, float, float]) -> float:
    """L1 distance in the embedded 4D space."""

    return sum(abs(ai - bi) for ai, bi in zip(a4, b4))


def cc_distance_4d(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Circuit-level unweighted distance per Eq. (E4)."""

    return l1_4d(cc_embed_4d(*a), cc_embed_4d(*b))


def distance_to_boundary_min_x_or_y(x1: int, x2: int, d: int) -> int:
    """Heuristic distance to the nearest planar boundary along x1 or x2.

    For rotated planar codes with square patch of nominal distance ``d``.
    Adjust as needed for your specific boundary labeling.
    """

    return min(x1, d - x1, x2, d - x2)


__all__ = [
    "manhattan_3d",
    "cc_embed_4d",
    "cc_distance_4d",
    "distance_to_boundary_min_x_or_y",
]

