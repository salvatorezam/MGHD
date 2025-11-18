#!/usr/bin/env python
"""Cache triangular color‑code CSS matrices for odd distances.

This CLI attempts to construct 6.6.6 and/or 4.8.8 triangular color codes for
odd distances up to ``--max-d`` and saves compressed ``.npz`` files under
``color_cache/`` for fast, reproducible reuse.

Providers
- 6.6.6: qecsim (optional) via mghd.codes.external_providers
- 4.8.8: panqec or quantum‑pecos (optional) via mghd.codes.external_color_488

When providers are missing, the script logs a warning and skips those entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

DATA_DIR = Path("color_cache")


def _ensure_data_dir() -> None:
    """Create the output cache directory if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_npz(
    kind: str, distance: int, Hx: np.ndarray, Hz: np.ndarray, n: int, layout: dict
) -> None:
    """Write a color code instance to ``color_cache/color_{kind}_d{distance}.npz``.

    Stores uint8 Hx/Hz, the data‑qubit count ``n``, and a JSON‑encoded layout
    metadata dict under key ``meta``.
    """
    path = DATA_DIR / f"color_{kind}_d{distance}.npz"
    np.savez_compressed(
        path,
        Hx=Hx.astype(np.uint8),
        Hz=Hz.astype(np.uint8),
        n=int(n),
        meta=json.dumps(layout),
    )
    print(f"[cache] wrote {path}")


def main(argv: List[str] | None = None) -> int:
    """Entry point for color‑code precomputation.

    Arguments
    - argv: optional list of CLI arguments for programmatic invocation.
    """
    parser = argparse.ArgumentParser(description="Precompute triangular color-code parity checks")
    parser.add_argument(
        "--max-d", type=int, default=31, help="Maximum odd distance to precompute (>=3)."
    )
    parser.add_argument(
        "--which",
        choices=["666", "488", "both"],
        default="both",
        help="Which tilings to precompute.",
    )
    args = parser.parse_args(argv)

    wanted = {"666", "488"} if args.which == "both" else {args.which}
    _ensure_data_dir()
    failures: List[Tuple[str, int, str]] = []
    warned_488 = False

    for d in range(3, args.max_d + 1, 2):
        for kind in sorted(wanted):
            if kind == "666":
                try:
                    # Providers for 6.6.6 live under mghd.codes.external_providers
                    from mghd.codes import external_providers as cx  # optional provider package(s)
                except ImportError:
                    cx = None
                builder = getattr(cx, "build_color_666_qecsim", None) if cx else None
            else:
                # 4.8.8 provider lives under mghd.codes.external_color_488
                try:
                    from mghd.codes.external_color_488 import build_color_488 as builder  # type: ignore
                except Exception:
                    builder = None
            if builder is None:
                if kind == "488":
                    if not warned_488:
                        print(
                            "[color_488] Missing provider (install panqec or quantum-pecos); skipping cache generation."
                        )
                        failures.append((kind, d, "provider unavailable"))
                        warned_488 = True
                else:
                    failures.append((kind, d, "external_providers missing builder"))
                continue
            try:
                Hx, Hz, n, layout = builder(d)
            except ImportError as exc:  # pragma: no cover - informative logging
                if kind == "488":
                    if not warned_488:
                        print(
                            "[color_488] Missing provider (install panqec or quantum-pecos); skipping cache generation."
                        )
                        failures.append((kind, d, str(exc)))
                        warned_488 = True
                else:
                    failures.append((kind, d, str(exc)))
                continue
            except Exception as exc:  # pragma: no cover - informative logging
                failures.append((kind, d, str(exc)))
                continue
            _save_npz(kind, d, Hx, Hz, n, layout)
    if failures:
        print("[warn] failed entries:")
        for kind, d, msg in failures:
            print(f"  color_{kind} d={d}: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
