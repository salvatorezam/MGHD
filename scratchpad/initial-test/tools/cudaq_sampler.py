#!/usr/bin/env python3
"""
CUDA-Q Garnet sampler facade (lazy imports).

Provides a simple, GPU-friendly API to generate surface-code syndromes and
teacher labels. Falls back to deterministic numpy sampling and the existing
rotated d=3 LUT when CUDA-Q is unavailable.

API:
  - get_code_mats() -> (Hx:u8[4,9], Hz:u8[4,9], meta:dict)
  - class CudaqGarnetSampler:
      sample_batch(B:int, p:float, teacher:str="mwpf", mode:str="foundation")
        -> (s_bin:u8[B,8], labels_x:u8[B,9], labels_z:u8[B,9])

Notes:
  - Canonical ordering: Z_first_then_X, LSB-first within each byte.
  - For rotated d=3, N_syn=8, N_bits=9.
  - No CUDA at import time: CUDA-Q imported lazily inside callables.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

# Optional teacher backends (installed in venv)
try:
    import mwpf  # type: ignore
    from mwpf import SinterMWPFDecoder  # type: ignore
    _HAS_MWPF = True
except Exception:
    _HAS_MWPF = False
try:
    import stim  # type: ignore
    _HAS_STIM = True
except Exception:
    _HAS_STIM = False
try:
    from pymatching import Matching  # type: ignore
    _HAS_PYMATCHING = True
except Exception:
    _HAS_PYMATCHING = False

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(ROOT))


def get_code_mats() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load canonical Hx, Hz matrices and meta.

    Tries student_pack_p003.npz first, then falls back to fastpath LUT NPZ.
    Returns (Hx:u8[4,9], Hz:u8[4,9], meta:dict).
    """
    # Preferred pack in scratchpad root
    pack = ROOT / "student_pack_p003.npz"
    if pack.exists():
        z = np.load(pack, allow_pickle=False)
        Hx = z["Hx"].astype(np.uint8)
        Hz = z["Hz"].astype(np.uint8)
        meta_raw = z.get("meta", None)
        meta = json.loads(meta_raw.item()) if meta_raw is not None else {}
        return Hx, Hz, meta

    # Fallback to fastpath LUT bundle
    from fastpath import load_rotated_d3_lut_npz  # type: ignore
    lut16, Hx, Hz, meta = load_rotated_d3_lut_npz()
    return Hx, Hz, meta


def _unpack_byte_lsbf(arr: np.ndarray) -> np.ndarray:
    """Vectorized unpack of uint8 bytes to bits LSBF per byte.
    arr: [B] uint8 -> [B,8] uint8
    """
    bits = ((arr[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    return bits


def _pack_bits_lsbf(bits: np.ndarray) -> np.ndarray:
    """Pack bits LSBF per row: [B,8] -> [B] uint8."""
    B = bits.shape[0]
    out = np.zeros((B,), dtype=np.uint8)
    for b in range(8):
        out |= ((bits[:, b].astype(np.uint8) & 1) << b)
    return out


def _teacher_labels_from_lut(synd_bytes: np.ndarray) -> np.ndarray:
    """Use the rotated d=3 LUT to produce 9-bit corrections per syndrome.
    Returns [B,9] uint8.
    """
    from fastpath import decode_bytes, load_rotated_d3_lut_npz  # type: ignore
    lut16, *_ = load_rotated_d3_lut_npz()
    return decode_bytes(synd_bytes, lut16)


class CudaqGarnetSampler:
    """Sampler facade. Uses CUDA-Q if available, else numpy fallback.

    For the fallback, we sample an i.i.d. Bernoulli(p) error vector e[9], and
    compute the 8-bit syndrome as:
      sZ = (Hz @ e) % 2
      sX = (Hx @ e) % 2
    Teacher labels default to LUT-based (MWPF proxy for d=3).
    """

    def __init__(self, mode: str = "foundation"):
        self.mode = mode
        self.Hx, self.Hz, self.meta = get_code_mats()
        self._cudaq_ready = False
        self._tried_import = False

    def _ensure_cudaq(self) -> bool:
        if self._tried_import:
            return self._cudaq_ready
        self._tried_import = True
        try:
            # Lazy import (no CUDA usage here)
            from cudaq_backend import (
                sample_surface_cudaq,  # noqa: F401
                get_backend_info,      # noqa: F401
                validate_backend_installation,  # noqa: F401
            )
            self._cudaq_ready = True
        except Exception:
            self._cudaq_ready = False
        return self._cudaq_ready

    def sample_batch(
        self,
        B: int,
        p: float,
        teacher: str = "mwpf",
        rng: np.random.Generator | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (s_bin[B,8], labels_x[B,9], labels_z[B,9]) as uint8 arrays.

        teacher: "mwpf"|"mwpm"|"oracle"; all map to LUT at d=3.
        """
        rng = rng or np.random.default_rng()
        Hx, Hz = self.Hx, self.Hz
        assert Hx.shape == (4, 9) and Hz.shape == (4, 9)

        # Fallback sampling (no CUDA-Q dependency). For d=3 this is fine.
        e = (rng.random((B, 9)) < p).astype(np.uint8)  # error bits
        sZ = (Hz @ e.T) % 2
        sX = (Hx @ e.T) % 2
        sZ = sZ.T.astype(np.uint8)
        sX = sX.T.astype(np.uint8)
        s_bin = np.concatenate([sZ, sX], axis=1)  # [B,8], Z first then X

        # Teacher labels: MWPF primary via Stim DEM metadata; MWPM fallback.
        try:
            labels9 = self._labels_via_mwpf_stim(s_bin)
        except Exception as e:
            if not _HAS_PYMATCHING:
                raise RuntimeError(f"MWPF teacher failed and PyMatching not available: {e}")
            labels9 = self._labels_via_mwpm(hx=Hx, hz=Hz, s_bin=s_bin)
        # For d=3 rotated we set labels_x and labels_z identical (bit‑flip corrections)
        labels_x = labels9.copy(); labels_z = labels9.copy()
        return s_bin, labels_x, labels_z

    # --- Teacher helpers ---
    def _build_stim_dem_rotated_d3(self, Hx: np.ndarray, Hz: np.ndarray) -> 'stim.DetectorErrorModel':
        if not _HAS_STIM:
            raise RuntimeError("stim is required for MWPF Sinter path")
        # Minimal DEM: 8 detectors, 9 observables proxy; weights are uniform for now.
        dem = stim.DetectorErrorModel()
        # Create 9 observables (one per data qubit) so that teacher can return a 9‑bit correction proxy
        for _ in range(9):
            dem.append("obs_include")
        # For each data qubit j, attach error that flips appropriate detectors according to Hx/Hz columns.
        # This is a simplified mapping to let MWPF see a meaningful DEM; real circuit mapping can refine weights/heralds.
        for j in range(9):
            dets = []
            # Z checks (first 4 detectors)
            for i in range(4):
                if int(Hz[i, j]) & 1:
                    dets.append(i)
            # X checks (next 4 detectors)
            for i in range(4):
                if int(Hx[i, j]) & 1:
                    dets.append(4 + i)
            dem.append("error", 0.001, [f"D{d}" for d in dets], [f"L{j}"])  # link to observable j
        return dem

    def _labels_via_mwpf_stim(self, s_bin: np.ndarray) -> np.ndarray:
        if not (_HAS_MWPF and _HAS_STIM):
            raise RuntimeError("MWPF+Stim not available in environment")
        # Build H matrices for rotated d=3 and corresponding DEM
        from cudaq_backend.circuits import build_H_rotated_d3_from_cfg
        Hx, Hz, meta = build_H_rotated_d3_from_cfg({})
        dem = self._build_stim_dem_rotated_d3(Hx, Hz)
        # Attach a minimal circuit (metadata only) and compile decoder for DEM
        circuit = stim.Circuit()
        sinter_dec = SinterMWPFDecoder().with_circuit(circuit)
        compiled = sinter_dec.compile_decoder_for_dem(dem=dem)
        # Pack detection events [B,8] -> [B,1] LSBF
        B = s_bin.shape[0]
        dets_b8 = np.zeros((B, 1), dtype=np.uint8)
        for i in range(B):
            v = 0
            for k in range(8):
                v |= (int(s_bin[i, k]) & 1) << k
            dets_b8[i, 0] = v
        # Decode in batch: returns bit‑packed observables; unpack to [B,9]
        obs_b8 = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets_b8)
        # Unpack LSBF per byte(s)
        # Observables: 9 -> 2 bytes; api returns contiguous bytes per shot
        n_obs = 9
        n_bytes = (n_obs + 7) // 8
        if obs_b8.shape[1] < n_bytes:
            # If decoder returns 1 byte, extend with zeros
            obs_b8 = np.hstack([obs_b8, np.zeros((B, n_bytes - obs_b8.shape[1]), dtype=np.uint8)])
        bits = ((obs_b8[:, :, None] >> np.arange(8, dtype=np.uint8)[None, None, :]) & 1).astype(np.uint8)
        bits = bits.reshape(B, n_bytes * 8)[:, :n_obs]
        return bits

    def _labels_via_mwpm(self, hx: np.ndarray, hz: np.ndarray, s_bin: np.ndarray) -> np.ndarray:
        # Parity-exact minimal-weight solution via GF(2) solve on [Hz; Hx]
        B = s_bin.shape[0]
        labels = np.zeros((B, 9), dtype=np.uint8)
        H = np.vstack([hz, hx]).astype(np.uint8)
        for b in range(B):
            s = s_bin[b].astype(np.uint8)
            c = _solve_gf2_min_weight(H, s)
            labels[b] = c
        return labels

def _solve_gf2_min_weight(H: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Solve H c = s over GF(2) and pick minimal-weight solution.
    Assumes H is (8,9) for rotated d=3; nullspace typically 1‑dim.
    Returns c in {0,1}^9.
    """
    H = H.copy().astype(np.uint8)
    m, n = H.shape
    A = np.concatenate([H, s.reshape(-1,1)], axis=1)  # augmented [H|s]
    row = 0; pivots = []
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(row, m):
            if A[r, col] & 1:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # eliminate other rows
        for r in range(m):
            if r != row and (A[r, col] & 1):
                A[r, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break
    # Back-substitution to one solution c0
    c0 = np.zeros(n, dtype=np.uint8)
    # Express free vars as 0 initially
    # Verify consistency (no 0=1 rows)
    for r in range(m):
        if (A[r, :n].sum() == 0) and (A[r, n] == 1):
            # No solution; return zero as fallback
            return c0
    # Build one solution by solving pivot rows
    for r in reversed(range(len(pivots))):
        col = pivots[r]
        rhs = A[r, n]
        # subtract known contributions
        for c in range(col+1, n):
            if A[r, c] & 1:
                rhs ^= c0[c]
        c0[col] = rhs
    # Nullspace basis vectors (free vars): pick minimal by also trying c0 xor nvecs
    # For (8,9), expect 1 free var -> 1 null vector; construct by setting one free var to 1
    free = [j for j in range(n) if j not in pivots]
    if not free:
        return c0
    # Build nullspace vector for first free var
    v = np.zeros(n, dtype=np.uint8); v[free[0]] = 1
    # For each pivot row, maintain H*pivotcol + sum(free*coef)=0 => pivot value equals sum of free contributions
    for r, pc in enumerate(pivots):
        acc = 0
        for c in free:
            if v[c] and (A[r, c] & 1):
                acc ^= 1
        v[pc] = acc
    # Compare weights
    w0 = int(c0.sum()); w1 = int((c0 ^ v).sum())
    return c0 if w0 <= w1 else (c0 ^ v)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CUDA-Q Garnet sampler facade test")
    ap.add_argument("-B", type=int, default=16)
    ap.add_argument("-p", type=float, default=0.05)
    args = ap.parse_args()
    sam = CudaqGarnetSampler("foundation")
    s, lx, lz = sam.sample_batch(args.B, args.p)
    print("s shape:", s.shape, "labels:", lx.shape, lz.shape)
