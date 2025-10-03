#!/usr/bin/env python
"""Cache triangular color-code CSS matrices for odd d (<= max_d)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

DATA_DIR = Path("color_cache")


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_npz(kind: str, distance: int, Hx: np.ndarray, Hz: np.ndarray, n: int, layout: dict) -> None:
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
    parser = argparse.ArgumentParser(description="Precompute triangular color-code parity checks")
    parser.add_argument("--max-d", type=int, default=31, help="Maximum odd distance to precompute (>=3).")
    parser.add_argument("--which", choices=["666", "488", "both"], default="both",
                        help="Which tilings to precompute.")
    args = parser.parse_args(argv)

    try:
        from mghd_main import codes_external as cx
    except ImportError:
        import importlib
        cx = importlib.import_module("codes_external")

    wanted = {"666", "488"} if args.which == "both" else {args.which}
    _ensure_data_dir()
    failures: List[Tuple[str, int, str]] = []

    for d in range(3, args.max_d + 1, 2):
        for kind in sorted(wanted):
            builder_name = f"build_color_{kind}_qecsim"
            if not hasattr(cx, builder_name):
                failures.append((kind, d, f"codes_external missing {builder_name}"))
                continue
            builder = getattr(cx, builder_name)
            try:
                Hx, Hz, n, layout = builder(d)
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
    try:
        import qecsim  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        print("qecsim is not installed; install it (e.g. `pip install qecsim`) before running this script.",
              file=sys.stderr)
        sys.exit(2)
    sys.exit(main())
