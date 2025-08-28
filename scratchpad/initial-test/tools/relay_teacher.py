#!/usr/bin/env python3
"""
Relay teacher utilities for labels under rotated surface d=3.

Primary mode uses a 256-entry LUT (MWPF proxy at d=3). MWPM fallback may use
the same LUT for this setup. Provides both CLI and Python API.

CLI:
  echo "0 20 84" | python tools/relay_teacher.py --packed

Python:
  from tools.relay_teacher import teacher_labels
  labels_x, labels_z = teacher_labels(synd_bytes, mode="mwpf")
"""

from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))


def _load_lut() -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    from fastpath import load_rotated_d3_lut_npz  # type: ignore
    return load_rotated_d3_lut_npz()


def _decode_lut(synd_bytes: np.ndarray) -> np.ndarray:
    from fastpath import decode_bytes, load_rotated_d3_lut_npz  # type: ignore
    lut16, *_ = load_rotated_d3_lut_npz()
    return decode_bytes(synd_bytes, lut16)


def teacher_labels(synd_bytes: np.ndarray, mode: str = "mwpf") -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels_x, labels_z) both [B,9] uint8.
    For d=3 we use LUT for all modes for stability and speed.
    """
    labels = _decode_lut(np.asarray(synd_bytes, dtype=np.uint8))
    return labels.copy(), labels.copy()


def _stdin_bytes() -> np.ndarray:
    data = sys.stdin.read().strip()
    if not data:
        return np.zeros((0,), dtype=np.uint8)
    toks = [int(x) for x in data.split()]  # whitespace-separated integers 0..255
    arr = np.array(toks, dtype=np.uint8)
    return arr


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Relay teacher (LUT-based for d=3)")
    ap.add_argument("--mode", choices=["mwpf", "mwpm", "relay"], default="mwpf")
    ap.add_argument("--packed", action="store_true", help="Read uint8 syndromes from stdin")
    ap.add_argument("--json", action="store_true", help="Emit JSON with labels")
    args = ap.parse_args()

    if not args.packed:
        print("Use --packed and pipe a whitespace-separated list of bytes (0..255) via stdin")
        sys.exit(2)

    synd = _stdin_bytes()
    if synd.size == 0:
        print("[]")
        sys.exit(0)

    lx, lz = teacher_labels(synd, mode=args.mode)
    if args.json:
        out = {"labels_x": lx.tolist(), "labels_z": lz.tolist()}
        print(json.dumps(out))
    else:
        # print as rows of 9 bits
        for i in range(lx.shape[0]):
            print(" ".join(str(int(b)) for b in lx[i]))
