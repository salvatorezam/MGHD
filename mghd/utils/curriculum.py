"""Simple curriculum utilities."""

from __future__ import annotations

from typing import List


def parse_distances(spec: str) -> List[int]:
    """
    Accept:
      - comma list: '3,5,7'
      - range: '3-31'
      - range with step: '3-31:2'
    """

    spec = spec.strip()
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    if "-" in spec:
        if ":" in spec:
            rng, step = spec.split(":")
            lo, hi = [int(x) for x in rng.split("-")]
            st = int(step)
        else:
            lo, hi = [int(x) for x in spec.split("-")]
            st = 2
        return list(range(lo, hi + 1, st))
    return [int(spec)]


__all__ = ["parse_distances"]
